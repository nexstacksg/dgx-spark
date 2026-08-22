#!/usr/bin/env bash
# Voice model server — Qwen3-Omni-30B-A3B on :8091
# No official FP8 repo exists, so vLLM quantizes the BF16 weights to FP8 at load
# time (--quantization fp8) — ~30GB in memory instead of ~66GB.
# NOTE: vLLM serves the Omni *thinker*: audio/image/video IN -> text out.
# Speech synthesis (talker) is handled downstream — see scripts/voice_demo.py.
set -euo pipefail

VENV="$HOME/Documents/GitHub/agentic/.venv"
MODEL="$HOME/models/Qwen3-Omni-30B-A3B-Instruct"

# torchcodec needs FFmpeg shared libs; no system ffmpeg (no sudo), so we ship
# BtbN's FFmpeg 7.1 shared build in ~/.local/ffmpeg-shared
export LD_LIBRARY_PATH="$HOME/.local/ffmpeg-shared/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# vLLM's compile step subprocesses `ninja`; under systemd the venv is not
# activated, so its bin dir must be on PATH explicitly
export PATH="$VENV/bin:$PATH"

exec "$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name qwen3-omni-30b \
  --host 0.0.0.0 --port 8091 \
  --quantization fp8 \
  --max-model-len 32768 \
  --max-num-seqs 3 \
  --gpu-memory-utilization 0.30 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --chat-template "$MODEL/chat_template.jinja"
