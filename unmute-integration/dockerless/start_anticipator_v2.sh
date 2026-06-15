#!/bin/bash
# Starts the endpoint anticipation inference server (v2).
# Serves anticipation probabilities over WebSocket on port 8093.
set -ex

# Set your GPU
export CUDA_VISIBLE_DEVICES=0

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

cd "$(dirname "$0")/.."

python -m uvicorn dockerless.anticipator_inference_server_v2:app --host 127.0.0.1 --port 8093
