#!/usr/bin/env bash
# Switch the agent model from vLLM (:8000) to the SGLang pilot (:30000).
#
# Stops BOTH vLLM user services (SGLang's 0.50 mem-fraction + load transients
# need the machine to itself — see ../README.md "SGLang pilot"), then runs the
# pilot in the FOREGROUND: Ctrl+C stops SGLang and nothing else changes.
# Switch back with: scripts/sglang-pilot-stop.sh
#
# The vLLM services stay enabled — a reboot brings vLLM back, not SGLang.
set -euo pipefail

PILOT_REPO="$HOME/Documents/GitHub/dgx-spark-qwen38"
[ -x "$PILOT_REPO/run.sh" ] || { echo "pilot repo missing: $PILOT_REPO (clone hasso5703/dgx-spark-qwen38 and ./install.sh --no-service first)" >&2; exit 1; }

echo "── stopping vLLM services (they stay enabled; reboot restores them)"
systemctl --user stop qwen38.service qwen-omni.service

# vLLM frees its allocation on exit, but give the kernel a moment to reclaim
for i in $(seq 1 12); do
  AVAIL_GB=$(awk '/^MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  [ "$AVAIL_GB" -ge 90 ] && break
  sleep 5
done
echo "── ${AVAIL_GB} GB available, launching SGLang on :30000 (first boot ≈ 9 min)"
exec "$PILOT_REPO/run.sh"
