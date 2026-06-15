#!/usr/bin/env python3
"""
Offline inference script for the endpoint anticipation model.

Processes one or two audio files (user stream + optional system stream) through
the Mimi encoder and anticipation transformer, and outputs per-frame endpoint
probabilities.

Usage:
    # Dual-stream (user + system audio separate)
    python infer.py --config configs/forecasting/mimi/fc2560_transformer_mimi_12.5hz_loss1-01_m3.yaml \
                    --checkpoint /path/to/checkpoints/<run_name>/best_val_acc.pt \
                    --user_audio user.wav \
                    --system_audio system.wav \
                    --output predictions.json

    # Single-stream (zero-filled system stream)
    python infer.py --config configs/forecasting/mimi/fc2560_transformer_mimi_12.5hz_loss1-01_m3.yaml \
                    --checkpoint /path/to/best_val_acc.pt \
                    --user_audio audio.wav \
                    --output predictions.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from moshi.models import loaders

from src.utils.common import load_config, fc_base_transformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MIMI_HF_REPO = "kyutai/mimi"
MIMI_CHUNK_SIZE = 1920   # 80ms at 24kHz
MAX_CONTEXT_STEPS = 240  # ~19.2 seconds of rolling context


def load_audio(path: str, target_sr: int) -> np.ndarray:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0).numpy()


def load_mimi(device: torch.device):
    logger.info("Loading Mimi encoder from %s", MIMI_HF_REPO)
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(MIMI_HF_REPO)
    mimi = checkpoint_info.get_mimi(device=device)
    mimi.streaming_forever(batch_size=1)
    logger.info("Mimi encoder loaded")
    return mimi


def load_model(cfg, checkpoint_path: str, device: torch.device):
    logger.info("Building model from config")
    model = fc_base_transformer(cfg, feat_extractor=None)
    logger.info("Loading checkpoint from %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info("Model loaded")
    return model


def encode_chunk(mimi, chunk: np.ndarray, device: torch.device) -> torch.Tensor:
    """Encode one MIMI_CHUNK_SIZE audio chunk → embedding tensor (1, feat, T)."""
    audio = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        codes = mimi.encode(audio)
        embeddings = mimi.quantizer.decode(codes)
    return embeddings  # (1, feat_size, T_frames)


def build_zero_stream(mimi, device: torch.device, feat_size: int, n_steps: int) -> torch.Tensor:
    """Zero-filled system stream (used when no system audio is provided)."""
    zero_audio = torch.zeros(1, 1, MIMI_CHUNK_SIZE, device=device)
    with torch.no_grad():
        zero_codes = mimi.encode(zero_audio)[:, :, :1]
        zero_embed = mimi.quantizer.decode(zero_codes)  # (1, feat, 1)
    return zero_embed.repeat(1, 1, n_steps)  # (1, feat, n_steps)


def run_inference(
    mimi,
    model,
    user_audio: np.ndarray,
    system_audio: np.ndarray | None,
    device: torch.device,
    threshold: float,
) -> list[dict]:
    """
    Chunk audio into MIMI_CHUNK_SIZE windows, encode, and run the anticipation model.
    Returns a list of per-frame predictions.
    """
    n_chunks = len(user_audio) // MIMI_CHUNK_SIZE
    if n_chunks == 0:
        raise ValueError("Audio too short — needs at least one Mimi chunk (1920 samples at 24kHz).")

    user_cache: torch.Tensor | None = None
    sys_cache: torch.Tensor | None = None
    zero_stream_template: torch.Tensor | None = None

    predictions = []

    for i in range(n_chunks):
        start = i * MIMI_CHUNK_SIZE
        end = start + MIMI_CHUNK_SIZE

        # Encode user chunk
        user_embed = encode_chunk(mimi, user_audio[start:end], device)
        user_cache = torch.cat([user_cache, user_embed], dim=-1) if user_cache is not None else user_embed

        # Encode system chunk (or use zero stream)
        if system_audio is not None:
            sys_embed = encode_chunk(mimi, system_audio[start:end], device)
        else:
            if zero_stream_template is None:
                zero_stream_template = build_zero_stream(mimi, device, user_embed.shape[1], 1)
            sys_embed = zero_stream_template
        sys_cache = torch.cat([sys_cache, sys_embed], dim=-1) if sys_cache is not None else sys_embed

        # Trim to rolling context window
        if user_cache.shape[-1] > MAX_CONTEXT_STEPS:
            user_cache = user_cache[..., -MAX_CONTEXT_STEPS:]
            sys_cache = sys_cache[..., -MAX_CONTEXT_STEPS:]

        # model input: (2, feat_size, T)
        model_input = torch.cat([user_cache, sys_cache], dim=0)

        with torch.no_grad():
            # model expects (batch, channels, feat, T) — wrap accordingly
            x = model_input.unsqueeze(0)  # (1, 2, feat, T)
            # Dummy label tensor (not used during inference)
            dummy_label = torch.zeros(1, x.shape[-1], 1, device=device)
            output = model(x, dummy_label)
            probs = output.squeeze(0).squeeze(-1)  # (T,) or (T, n_horizons)

        # Take only the frames from the latest chunk
        chunk_frames = user_embed.shape[-1]
        frame_probs = probs[-chunk_frames:]

        time_offset = i * MIMI_CHUNK_SIZE / 24000.0  # seconds

        for f in range(chunk_frames):
            prob_val = float(frame_probs[f].item()) if frame_probs.dim() == 1 else float(frame_probs[f, -1].item())
            frame_time = time_offset + f * (MIMI_CHUNK_SIZE / chunk_frames) / 24000.0
            predictions.append({
                "time_s": round(frame_time, 4),
                "probability": round(prob_val, 4),
                "endpoint_detected": prob_val >= threshold,
            })

    return predictions


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

    # Waveforms (left y-axis)
    ax_audio.plot(audio_times, user_audio, color="#2196F3", alpha=0.45, linewidth=0.6, label="User")
    if system_audio is not None:
        ax_audio.plot(audio_times, system_audio, color="#FF5722", alpha=0.45, linewidth=0.6, label="System")
    ax_audio.set_xlabel("Time (s)")
    ax_audio.set_ylabel("Amplitude", color="#555555")
    ax_audio.tick_params(axis="y", labelcolor="#555555")
    ax_audio.set_xlim(audio_times[0], audio_times[-1])

    # Anticipation probabilities (right y-axis)
    ax_prob = ax_audio.twinx()
    ax_prob.plot(times, probs, color="#4CAF50", linewidth=1.4, label="Anticipation prob", zorder=3)
    ax_prob.axhline(threshold, color="#4CAF50", linewidth=0.8, linestyle="--", alpha=0.6)
    ax_prob.set_ylabel("Anticipation probability", color="#4CAF50")
    ax_prob.tick_params(axis="y", labelcolor="#4CAF50")
    ax_prob.set_ylim(0, 1)

    # Combined legend
    lines_audio, labels_audio = ax_audio.get_legend_handles_labels()
    lines_prob, labels_prob = ax_prob.get_legend_handles_labels()
    ax_audio.legend(lines_audio + lines_prob, labels_audio + labels_prob, loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Plot saved to %s", save_path)


def main():
    parser = argparse.ArgumentParser(description="Endpoint anticipation offline inference")
    parser.add_argument("--config", required=True, help="Model config YAML (e.g. configs/forecasting/mimi/fc2560_...yaml)")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--user_audio", required=True, help="User audio file (WAV)")
    parser.add_argument("--system_audio", default=None, help="System audio file (WAV). If omitted, zero stream is used.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for endpoint detection")
    parser.add_argument("--output", default=None, help="Path to save predictions JSON. Prints to stdout if omitted.")
    parser.add_argument("--plot", default=None, help="Path to save prediction plot (e.g. predictions.png).")
    parser.add_argument("--device", default=None, help="Device override (cuda / cpu). Auto-detected if omitted.")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Using device: %s", device)

    cfg = load_config([args.config])
    target_sr = cfg.data.audio_params.target_sr

    mimi = load_mimi(device)
    model = load_model(cfg, args.checkpoint, device)

    logger.info("Loading user audio from %s", args.user_audio)
    user_audio = load_audio(args.user_audio, target_sr)

    system_audio = None
    if args.system_audio:
        logger.info("Loading system audio from %s", args.system_audio)
        system_audio = load_audio(args.system_audio, target_sr)
        min_len = min(len(user_audio), len(system_audio))
        user_audio = user_audio[:min_len]
        system_audio = system_audio[:min_len]

    logger.info("Running inference on %.2f seconds of audio", len(user_audio) / target_sr)
    predictions = run_inference(mimi, model, user_audio, system_audio, device, args.threshold)

    detected = [p for p in predictions if p["endpoint_detected"]]
    logger.info("Inference complete: %d frames, %d endpoint detections", len(predictions), len(detected))

    result = {
        "config": args.config,
        "checkpoint": args.checkpoint,
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
