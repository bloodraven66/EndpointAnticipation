#!/usr/bin/env python3
"""
Offline inference script for the endpoint anticipation model.

Processes one or two audio files (user stream + optional system stream) through
the Mimi encoder and anticipation transformer, and outputs per-frame endpoint
probabilities.

By default, the pretrained checkpoint and config are downloaded automatically
from HuggingFace (viks66/endpoint-anticipation).

Usage:
    # Using pretrained checkpoint from HuggingFace (default)
    python infer.py --user_audio user.wav --system_audio system.wav --plot out.png

    # Dual-stream with a local checkpoint
    python infer.py --config configs/forecasting/mimi/fc960_transformer_mimi_12.5hz_loss1-01_m3.yaml \
                    --checkpoint /path/to/best_val_acc.pt \
                    --user_audio user.wav \
                    --system_audio system.wav

    # Single-stream (zero-filled system stream)
    python infer.py --user_audio audio.wav --output predictions.json

Dependencies:
    pip install torch torchaudio moshi numpy matplotlib huggingface_hub pyyaml
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import yaml
from huggingface_hub import hf_hub_download
from moshi.models import loaders
from moshi.modules.transformer import StreamingTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MIMI_HF_REPO = "kyutai/mimi"
ANTICIPATION_HF_REPO = "viks66/endpoint-anticipation"
MIMI_CHUNK_SIZE = 1920   # 80ms at 24kHz
MAX_CONTEXT_STEPS = 240  # ~19.2 seconds of rolling context


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class AnticipationModel(nn.Module):
    """Dual-stream causal transformer for endpoint anticipation."""

    def __init__(self, hidden_size: int, num_heads: int, num_layers: int,
                 dim_feedforward: int, context: int, positional_embedding: str,
                 max_period: float, output_size: int = 1, **_: Any):
        super().__init__()
        transformer_kwargs = dict(
            d_model=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            causal=True,
            context=context,
            positional_embedding=positional_embedding,
            max_period=max_period,
        )
        self.model1 = StreamingTransformer(**transformer_kwargs)
        self.model2 = StreamingTransformer(**transformer_kwargs)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (2, feat_size, T)
        x1 = x[0].unsqueeze(0).permute(0, 2, 1)   # (1, T, feat)
        x2 = x[1].unsqueeze(0).permute(0, 2, 1)
        x1 = self.model1(x1)
        x2 = self.model2(x2)
        return torch.sigmoid(self.linear(torch.cat([x1, x2], dim=2)))  # (1, T, output_size)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def resolve_from_hf(config_path: str | None, checkpoint_path: str | None) -> tuple[str, str]:
    if config_path is None:
        logger.info("Downloading config from %s", ANTICIPATION_HF_REPO)
        config_path = hf_hub_download(repo_id=ANTICIPATION_HF_REPO, filename="config.yaml")
    if checkpoint_path is None:
        logger.info("Downloading checkpoint from %s", ANTICIPATION_HF_REPO)
        checkpoint_path = hf_hub_download(repo_id=ANTICIPATION_HF_REPO, filename="best_val_acc.pt")
    return config_path, checkpoint_path


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_mimi(device: torch.device):
    logger.info("Loading Mimi encoder from %s", MIMI_HF_REPO)
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(MIMI_HF_REPO)
    mimi = checkpoint_info.get_mimi(device=device)
    mimi.streaming_forever(batch_size=1)
    logger.info("Mimi encoder loaded")
    return mimi


def load_model(cfg: dict, checkpoint_path: str, device: torch.device) -> AnticipationModel:
    mp = cfg["model_params"]
    n_intervals = len(cfg["data"]["label_params"]["forecast_intervals_ms"])
    model = AnticipationModel(output_size=n_intervals, **mp)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device).eval()
    logger.info("Anticipation model loaded from %s", checkpoint_path)
    return model


def load_audio(path: str, target_sr: int) -> np.ndarray:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0).numpy()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def encode_chunk(mimi, chunk: np.ndarray, device: torch.device) -> torch.Tensor:
    audio = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        codes = mimi.encode(audio)
        return mimi.quantizer.decode(codes)  # (1, feat_size, T_frames)


def get_zero_stream(mimi, device: torch.device) -> torch.Tensor:
    zero_audio = torch.zeros(1, 1, MIMI_CHUNK_SIZE, device=device)
    with torch.no_grad():
        codes = mimi.encode(zero_audio)[:, :, :1]
        return mimi.quantizer.decode(codes)  # (1, feat, 1)


def run_inference(
    mimi,
    model: AnticipationModel,
    user_audio: np.ndarray,
    system_audio: np.ndarray | None,
    device: torch.device,
    threshold: float,
    target_sr: int,
) -> list[dict]:
    n_chunks = len(user_audio) // MIMI_CHUNK_SIZE
    if n_chunks == 0:
        raise ValueError("Audio too short — needs at least one Mimi chunk (1920 samples at 24 kHz).")

    user_cache: torch.Tensor | None = None
    sys_cache: torch.Tensor | None = None
    zero_embed: torch.Tensor | None = None
    predictions: list[dict] = []

    for i in range(n_chunks):
        s, e = i * MIMI_CHUNK_SIZE, (i + 1) * MIMI_CHUNK_SIZE

        user_embed = encode_chunk(mimi, user_audio[s:e], device)
        user_cache = torch.cat([user_cache, user_embed], dim=-1) if user_cache is not None else user_embed

        if system_audio is not None:
            sys_embed = encode_chunk(mimi, system_audio[s:e], device)
        else:
            if zero_embed is None:
                zero_embed = get_zero_stream(mimi, device)
            sys_embed = zero_embed.repeat(1, 1, user_embed.shape[-1])
        sys_cache = torch.cat([sys_cache, sys_embed], dim=-1) if sys_cache is not None else sys_embed

        if user_cache.shape[-1] > MAX_CONTEXT_STEPS:
            user_cache = user_cache[..., -MAX_CONTEXT_STEPS:]
            sys_cache = sys_cache[..., -MAX_CONTEXT_STEPS:]

        model_input = torch.cat([user_cache, sys_cache], dim=0)  # (2, feat, T)

        with torch.no_grad():
            output = model(model_input)  # (1, T, output_size)

        chunk_frames = user_embed.shape[-1]
        frame_probs = output[0, -chunk_frames:, -1]  # last horizon, latest frames

        for f in range(chunk_frames):
            frame_time = (s + f * MIMI_CHUNK_SIZE / chunk_frames) / target_sr
            prob_val = float(frame_probs[f].item())
            predictions.append({
                "time_s": round(frame_time, 4),
                "probability": round(prob_val, 4),
                "endpoint_detected": prob_val >= threshold,
            })

    return predictions


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_predictions(
    predictions: list[dict],
    user_audio: np.ndarray,
    system_audio: np.ndarray | None,
    target_sr: int,
    threshold: float,
    save_path: str,
) -> None:
    times = np.array([p["time_s"] for p in predictions])
    probs = np.array([p["probability"] for p in predictions])
    audio_times = np.arange(len(user_audio)) / target_sr

    fig, ax_audio = plt.subplots(figsize=(14, 3))

    ax_audio.plot(audio_times, user_audio, color="#2196F3", alpha=0.45, linewidth=0.6, label="User")
    if system_audio is not None:
        ax_audio.plot(audio_times, system_audio, color="#FF5722", alpha=0.45, linewidth=0.6, label="System")
    ax_audio.set_xlabel("Time (s)")
    ax_audio.set_ylabel("Amplitude", color="#555555")
    ax_audio.tick_params(axis="y", labelcolor="#555555")
    ax_audio.set_xlim(audio_times[0], audio_times[-1])

    ax_prob = ax_audio.twinx()
    ax_prob.plot(times, probs, color="#4CAF50", linewidth=1.4, label="Anticipation prob", zorder=3)
    ax_prob.axhline(threshold, color="#4CAF50", linewidth=0.8, linestyle="--", alpha=0.6)
    ax_prob.set_ylabel("Anticipation probability", color="#4CAF50")
    ax_prob.tick_params(axis="y", labelcolor="#4CAF50")
    ax_prob.set_ylim(0, 1)

    lines_a, labels_a = ax_audio.get_legend_handles_labels()
    lines_p, labels_p = ax_prob.get_legend_handles_labels()
    ax_audio.legend(lines_a + lines_p, labels_a + labels_p, loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Plot saved to %s", save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Endpoint anticipation offline inference")
    parser.add_argument("--config", default=None, help="Model config YAML. Downloaded from HuggingFace if omitted.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path (.pt). Downloaded from HuggingFace if omitted.")
    parser.add_argument("--user_audio", required=True, help="User audio file (WAV)")
    parser.add_argument("--system_audio", default=None, help="System audio file (WAV). Zero stream used if omitted.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for endpoint detection")
    parser.add_argument("--output", default=None, help="Path to save predictions JSON. Prints to stdout if omitted.")
    parser.add_argument("--plot", default=None, help="Path to save prediction plot (PNG).")
    parser.add_argument("--device", default=None, help="Device override (cuda / cpu). Auto-detected if omitted.")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Using device: %s", device)

    config_path, checkpoint_path = resolve_from_hf(args.config, args.checkpoint)
    cfg = load_config(config_path)
    target_sr = cfg["data"]["audio_params"]["target_sr"]

    mimi = load_mimi(device)
    model = load_model(cfg, checkpoint_path, device)

    logger.info("Loading user audio from %s", args.user_audio)
    user_audio = load_audio(args.user_audio, target_sr)

    system_audio = None
    if args.system_audio:
        logger.info("Loading system audio from %s", args.system_audio)
        system_audio = load_audio(args.system_audio, target_sr)
        n = min(len(user_audio), len(system_audio))
        user_audio, system_audio = user_audio[:n], system_audio[:n]

    logger.info("Running inference on %.2f seconds of audio", len(user_audio) / target_sr)
    predictions = run_inference(mimi, model, user_audio, system_audio, device, args.threshold, target_sr)

    detected = [p for p in predictions if p["endpoint_detected"]]
    logger.info("Inference complete: %d frames, %d endpoint detections", len(predictions), len(detected))

    result = {
        "config": config_path,
        "checkpoint": checkpoint_path,
        "threshold": args.threshold,
        "audio_duration_s": round(len(user_audio) / target_sr, 3),
        "n_frames": len(predictions),
        "n_detections": len(detected),
        "predictions": predictions,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Predictions saved to %s", args.output)
    else:
        print(json.dumps(result, indent=2))

    if args.plot:
        plot_predictions(predictions, user_audio, system_audio, target_sr, args.threshold, args.plot)


if __name__ == "__main__":
    main()
