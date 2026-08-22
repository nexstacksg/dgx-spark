#!/usr/bin/env bash
# Agent/vision model server — Qwen3.8-27B (Unsloth Dynamic NVFP4) on :8000
#
# Replaces Qwen3.6-35B-A3B. This is a DENSE 27B, not a 3B-active MoE: every
# token reads the full 23.4 GB of weights, so single-stream decode is ~10-12
# tok/s (bandwidth-bound at ~273 GB/s). Aggregate throughput across concurrent
# users is fine (~134 tok/s at 16-way) because batching amortises weight reads.
#
# See README "Known constraints" before changing max-model-len / max-num-seqs /
# gpu-memory-utilization: the three together are the memory + bandwidth budget.
set -euo pipefail

# vLLM nightly venv (2026-08-22): needed for DFlash2 speculative decoding
# (DFlash2DraftModel is in no stable release yet). Fallbacks kept on disk:
#   .venv-v0271 — vLLM 0.27.1, works with the MTP speculative config
#   .venv       — vLLM 0.25.0, still serves qwen-omni; do not touch
VENV="$HOME/Documents/GitHub/agentic/.venv-nightly"
MODEL="$HOME/models/Qwen3.8-27B-NVFP4"

# torchcodec needs FFmpeg shared libs; no system ffmpeg (no sudo), so we ship
# BtbN's FFmpeg 7.1 shared build in ~/.local/ffmpeg-shared
export LD_LIBRARY_PATH="$HOME/.local/ffmpeg-shared/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# vLLM's compile step subprocesses `ninja`; under systemd the venv is not
# activated, so its bin dir must be on PATH explicitly
export PATH="$VENV/bin:$PATH"

# GB10 is sm_121; CUTLASS DSL kernels need the arch spelled with the 'a' suffix
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_121a}"

exec "$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 131072 \
  --max-num-seqs 10 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.50 \
  --kv-cache-dtype fp8 \
  --speculative-config "{\"method\":\"dflash\",\"model\":\"$HOME/models/Qwen3.8-27B-DFlash2\",\"num_speculative_tokens\":7}" \
  --default-chat-template-kwargs '{"reasoning_effort": "medium"}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3
