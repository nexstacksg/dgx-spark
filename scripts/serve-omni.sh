#!/usr/bin/env bash
# Voice model server — Qwen3-Omni-30B-A3B on :8091
# No official FP8 repo exists, so vLLM quantizes the BF16 weights to FP8 at load
# time (--quantization fp8) — ~30GB in memory instead of ~66GB.
# NOTE: vLLM serves the Omni *thinker*: audio/image/video IN -> text out.
# Speech synthesis (talker) is handled downstream — see scripts/voice_demo.py.
set -euo pipefail

VENV="$HOME/Documents/GitHub/dgx-spark/.venv"
MODEL="$HOME/models/Qwen3-Omni-30B-A3B-Instruct"

# torchcodec needs FFmpeg shared libs; no system ffmpeg (no sudo), so we ship
# BtbN's FFmpeg 7.1 shared build in ~/.local/ffmpeg-shared
export LD_LIBRARY_PATH="$HOME/.local/ffmpeg-shared/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# vLLM's compile step subprocesses `ninja`; under systemd the venv is not
# activated, so its bin dir must be on PATH explicitly.
#
# /usr/local/cuda/bin must ALSO be explicit: it is only added by
# /etc/profile.d/nv_paths.sh, which login shells read but the boot-time
# `systemd --user` manager (started by linger, before GNOME login) does not.
# Without `nvcc` on PATH vLLM's has_flashinfer() is False, the
# vllm::flashinfer_mm_fp4 custom op is never registered, and the cached AOT
# graph that calls it dies in profile_run:
#   AttributeError: '_OpNamespace' 'vllm' object has no attribute 'flashinfer_mm_fp4'
# That killed the first start attempt on every reboot (2026-09-02, 2026-09-09);
# only systemd's retry, which ran after login had imported the shell PATH, came up.
export PATH="$VENV/bin:/usr/local/cuda/bin:$PATH"

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
