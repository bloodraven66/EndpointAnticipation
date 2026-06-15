#!/usr/bin/env python3
"""
Anticipator Inference Server v2

Streams per-frame endpoint anticipation probabilities over WebSocket.
The anticipation model and config are downloaded automatically from HuggingFace
(viks66/endpoint-anticipation) on first run.

Start with:
    python -m uvicorn dockerless.anticipator_inference_server_v2:app --host 127.0.0.1 --port 8093
Or use start_anticipator_v2.sh.
"""

from __future__ import annotations

import argparse
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional

import msgpack
import numpy as np
import torch
import torch.nn as nn
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download
from moshi.models import loaders
from moshi.modules.transformer import StreamingTransformer

os.environ["TORCHDYNAMO_DISABLE"] = "1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000
BASE_FRAME_SIZE = 960       # 40ms at 24kHz
MIMI_CHUNK_SIZE = 1920      # 80ms at 24kHz
ENDPOINTER_OUTPUT_RATE = 12.5
ACCEPTED_FRAME_SIZES = {960, 1920}
MAX_AUDIO_SAMPLES_PER_MESSAGE = SAMPLE_RATE * 5

ANTICIPATION_HF_REPO = "viks66/endpoint-anticipation"
MIMI_HF_REPO = "kyutai/stt-1b-en_fr"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class AnticipationModel(nn.Module):
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
        x1 = x[0].unsqueeze(0).permute(0, 2, 1)
        x2 = x[1].unsqueeze(0).permute(0, 2, 1)
        return torch.sigmoid(self.linear(torch.cat([self.model1(x1), self.model2(x2)], dim=2)))


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

@dataclass
class SessionStats:
    messages_received: int = 0
    samples_received: int = 0
    frames_received: int = 0
    outputs_sent: int = 0
    mismatched_packets: int = 0


class AnticipatorSession:
    def __init__(self, mimi_model: Any, anticipator_model: AnticipationModel,
                 device: torch.device, max_context_steps: int = 240):
        self.mimi = mimi_model
        self.anticipator = anticipator_model
        self.device = device
        self.max_context_steps = max_context_steps

        self.base_frame_fifo = np.zeros(0, dtype=np.float32)
        self.mimi_chunk_fifo = np.zeros(0, dtype=np.float32)

        self.zero_stream_cache: torch.Tensor | None = None
        self.cached_embeds: torch.Tensor | None = None

        self.stats = SessionStats()

    def _coerce_audio_payload(self, pcm: np.ndarray) -> np.ndarray:
        if pcm.ndim != 1:
            pcm = pcm.reshape(-1)
        if pcm.dtype != np.float32:
            pcm = pcm.astype(np.float32)
        if len(pcm) == 0:
            return pcm
        if len(pcm) > MAX_AUDIO_SAMPLES_PER_MESSAGE:
            raise ValueError(f"Audio packet too large: {len(pcm)} samples")
        if not np.isfinite(pcm).all():
            pcm = np.nan_to_num(pcm, nan=0.0, posinf=1.0, neginf=-1.0)
        if len(pcm) not in ACCEPTED_FRAME_SIZES:
            self.stats.mismatched_packets += 1
            logger.warning("Unexpected packet size %d; rechunking.", len(pcm))
        return pcm

    async def process_audio_payload(self, pcm: np.ndarray) -> list[tuple[float, int]]:
        pcm = self._coerce_audio_payload(pcm)
        if len(pcm) == 0:
            return []

        self.stats.messages_received += 1
        self.stats.samples_received += len(pcm)
        self.base_frame_fifo = np.concatenate([self.base_frame_fifo, pcm])

        predictions: list[tuple[float, int]] = []
        while len(self.base_frame_fifo) >= BASE_FRAME_SIZE:
            frame = self.base_frame_fifo[:BASE_FRAME_SIZE]
            self.base_frame_fifo = self.base_frame_fifo[BASE_FRAME_SIZE:]
            self.stats.frames_received += 1
            prob = await self._process_base_frame(frame)
            if prob is not None:
                self.stats.outputs_sent += 1
                predictions.append((prob, self.stats.frames_received))

        return predictions

    async def _process_base_frame(self, frame: np.ndarray) -> Optional[float]:
        self.mimi_chunk_fifo = np.concatenate([self.mimi_chunk_fifo, frame])
        if len(self.mimi_chunk_fifo) < MIMI_CHUNK_SIZE:
            return None

        chunk = self.mimi_chunk_fifo[:MIMI_CHUNK_SIZE]
        self.mimi_chunk_fifo = self.mimi_chunk_fifo[MIMI_CHUNK_SIZE:]

        audio_tensor = torch.from_numpy(chunk).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            codes = self.mimi.encode(audio_tensor.unsqueeze(0))
            embeddings = self.mimi.quantizer.decode(codes)

        current_chunk_steps = embeddings.shape[-1]
        if self.cached_embeds is not None:
            embeddings = torch.cat([self.cached_embeds, embeddings], dim=-1)
        if embeddings.shape[-1] > self.max_context_steps:
            embeddings = embeddings[..., -self.max_context_steps:]
        self.cached_embeds = embeddings.clone()

        if self.zero_stream_cache is None:
            zero_audio = torch.zeros_like(audio_tensor)
            with torch.no_grad():
                zero_codes = self.mimi.encode(zero_audio.unsqueeze(0))[:, :, :1]
                self.zero_stream_cache = self.mimi.quantizer.decode(zero_codes)

        zero_stream = self.zero_stream_cache.repeat(1, 1, embeddings.shape[-1])
        model_input = torch.cat([embeddings, zero_stream], dim=0)

        with torch.no_grad():
            output = self.anticipator(model_input)
            probs = output.squeeze(0).squeeze(-1)[-current_chunk_steps:]

        return float(probs[-1].item())

    def get_stats(self) -> dict:
        return {
            "messages_received": self.stats.messages_received,
            "samples_received": self.stats.samples_received,
            "frames_received": self.stats.frames_received,
            "outputs_sent": self.stats.outputs_sent,
            "mismatched_packets": self.stats.mismatched_packets,
            "expected_output_rate_hz": ENDPOINTER_OUTPUT_RATE,
            "accepted_frame_sizes": sorted(ACCEPTED_FRAME_SIZES),
        }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

mimi_model = None
anticipator_model = None
device = None


def load_models() -> None:
    global mimi_model, anticipator_model, device

    logger.info("Loading models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    logger.info("Loading Mimi from %s", MIMI_HF_REPO)
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(MIMI_HF_REPO)
    mimi_model = checkpoint_info.get_mimi(device=device)
    mimi_model.streaming_forever(batch_size=1)

    logger.info("Downloading anticipation model config from %s", ANTICIPATION_HF_REPO)
    config_path = hf_hub_download(repo_id=ANTICIPATION_HF_REPO, filename="config.yaml")
    checkpoint_path = hf_hub_download(repo_id=ANTICIPATION_HF_REPO, filename="best_val_acc.pt")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    mp = cfg["model_params"]
    n_intervals = len(cfg["data"]["label_params"]["forecast_intervals_ms"])
    anticipator_model = AnticipationModel(output_size=n_intervals, **mp)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    anticipator_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    anticipator_model.to(device).eval()
    logger.info("Anticipation model loaded")

    # GPU warmup
    logger.info("Running warmup forward pass...")
    with torch.no_grad():
        dummy = torch.zeros(1, 1, MIMI_CHUNK_SIZE, device=device)
        _codes = mimi_model.encode(dummy)
        _embeds = mimi_model.quantizer.decode(_codes)
        _ = anticipator_model(torch.cat([_embeds, _embeds], dim=0))
    logger.info("Warmup done.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Anticipator Inference Server v2")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_models()
    yield


app.router.lifespan_context = lifespan


@app.get("/api/build_info")
async def build_info():
    return JSONResponse({
        "service": "anticipator",
        "version": "2.0.0",
        "hf_repo": ANTICIPATION_HF_REPO,
        "sample_rate": SAMPLE_RATE,
        "base_frame_size": BASE_FRAME_SIZE,
        "mimi_chunk_size": MIMI_CHUNK_SIZE,
        "output_rate_hz": ENDPOINTER_OUTPUT_RATE,
        "device": str(device),
    })


@app.websocket("/api/endpointer_stream")
async def endpointer_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    session = AnticipatorSession(mimi_model, anticipator_model, device)

    ready_msg = msgpack.packb(
        {"type": "Ready", "service": "anticipator", "version": "2.0.0",
         "accepted_frame_sizes": sorted(ACCEPTED_FRAME_SIZES),
         "base_frame_size": BASE_FRAME_SIZE},
        use_bin_type=True,
    )
    await websocket.send_bytes(ready_msg)

    try:
        while True:
            message_bytes = await websocket.receive_bytes()
            message = msgpack.unpackb(message_bytes, raw=False)
            msg_type = message.get("type")

            if msg_type == "Audio":
                pcm = np.array(message.get("pcm", []), dtype=np.float32)
                for prob, frame_count in await session.process_audio_payload(pcm):
                    response = msgpack.packb(
                        {"type": "Prediction", "user_end_probability": prob,
                         "frame_count": frame_count},
                        use_bin_type=True, use_single_float=True,
                    )
                    await websocket.send_bytes(response)

            elif msg_type == "GetStats":
                await websocket.send_bytes(
                    msgpack.packb({"type": "Stats", **session.get_stats()}, use_bin_type=True)
                )
            else:
                logger.warning("Unknown message type: %r", msg_type)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as exc:
        logger.error("Error in WebSocket handler: %r", exc, exc_info=True)
        try:
            await websocket.send_bytes(
                msgpack.packb({"type": "Error", "message": str(exc)}, use_bin_type=True)
            )
        except Exception:
            pass
    finally:
        logger.info("Session ended. Stats: %s", session.get_stats())


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, ws="websockets",
                log_level="info", timeout_keep_alive=300,
                ws_ping_interval=60, ws_ping_timeout=60)
