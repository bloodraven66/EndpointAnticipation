#!/bin/bash
# Starts the Kyutai moshi-server TTS worker.
# Requires: Rust + moshi-server installed (see install.sh)
set -ex

# Set your GPU
export CUDA_VISIBLE_DEVICES=0

export CUDA_HOME=/usr/local/cuda
export CARGO_HOME="$HOME/.cargo"
export RUSTUP_HOME="$HOME/.rustup"
export PATH="$CARGO_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

cd "$(dirname "$0")/.."

moshi-server worker --config services/moshi-server/configs/tts.toml --addr 127.0.0.1 --port 8089
