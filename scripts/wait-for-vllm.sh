#!/usr/bin/env bash
# Block until a vLLM server answers /v1/models, or give up after TIMEOUT seconds.
#
# WHY THIS EXISTS: systemd's After= only orders when the start command is
# *dispatched*, not when the model is loaded. With Type=simple, a vLLM unit
# counts as "started" the instant the process forks, so two units listed
# After= one another still load weights concurrently. vLLM sizes its KV cache
# by measuring free memory during a profiling window; if a neighbour allocates
# 30 GB of weights inside that window, the profiler charges it to *this*
# server's budget and KV cache memory comes out negative:
#
#   ValueError: No available memory for the cache blocks
#
# Both servers died this way on 2026-08-22. Gating on real readiness serialises
# the loads so each profiler sees a stable baseline.
set -euo pipefail

URL="${1:?usage: wait-for-vllm.sh URL [TIMEOUT_SECONDS]}"
TIMEOUT="${2:-1800}"

deadline=$(( SECONDS + TIMEOUT ))
until curl -sf -m 5 "$URL" >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then
    echo "wait-for-vllm: $URL not ready after ${TIMEOUT}s" >&2
    exit 1
  fi
  sleep 5
done
echo "wait-for-vllm: $URL is ready"
