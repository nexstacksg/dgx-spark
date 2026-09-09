#!/usr/bin/env bash
# Agent/vision model server — Qwen3.6-35B-A3B (official FP8) on :8000
# See README "Known constraints" before changing max-model-len / max-num-seqs /
# gpu-memory-utilization: the three together are the memory + bandwidth budget.
set -euo pipefail

VENV="$HOME/Documents/GitHub/dgx-spark/.venv"
MODEL="$HOME/models/Qwen3.6-35B-A3B-FP8"

# torchcodec needs FFmpeg shared libs; no system ffmpeg (no sudo), so we ship
# BtbN's FFmpeg 7.1 shared build in ~/.local/ffmpeg-shared
export LD_LIBRARY_PATH="$HOME/.local/ffmpeg-shared/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# vLLM's compile step subprocesses `ninja`; under systemd the venv is not
# activated, so its bin dir must be on PATH explicitly
export PATH="$VENV/bin:$PATH"

exec "$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name qwen3.6-35b-a3b \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 65536 \
  --max-num-seqs 10 \
  --gpu-memory-utilization 0.55 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3
