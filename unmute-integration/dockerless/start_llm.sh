#!/bin/bash
# Starts the vLLM server with Gemma 3 1B.
# Requires: vllm installed (pip install vllm)
set -ex

# Set your GPU
export CUDA_VISIBLE_DEVICES=0

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

cd "$(dirname "$0")/.."

vllm serve google/gemma-3-1b-it \
  --max-model-len=8192 \
  --dtype=bfloat16 \
  --gpu-memory-utilization=0.7 \
  --host=127.0.0.1 \
  --port=8091
