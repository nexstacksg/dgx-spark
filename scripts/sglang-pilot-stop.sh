#!/usr/bin/env bash
# Switch back from the SGLang pilot to the vLLM stack.
# Stops the pilot container (if still running), restarts both vLLM services
# (the omni unit's ExecStartPre gate serialises the two loads), and waits for
# :8000 to actually answer.
set -euo pipefail

echo "── stopping SGLang pilot container (ok if not running)"
docker stop qwen38-sglang-run 2>/dev/null || sg docker -c "docker stop qwen38-sglang-run" 2>/dev/null || true

echo "── starting vLLM services"
systemctl --user start qwen38.service qwen-omni.service

echo "── waiting for :8000 (warm start ≈ 4 min)"
for i in $(seq 1 90); do
  if curl -s -m 3 http://localhost:8000/v1/models 2>/dev/null | grep -q qwen; then
    echo "✅ vLLM agent server back on :8000"
    exit 0
  fi
  systemctl --user is-active --quiet qwen38.service || { echo "qwen38.service failed — check: journalctl --user -u qwen38.service -n 50" >&2; exit 1; }
  sleep 10
done
echo "still not answering after 15 min — check: journalctl --user -u qwen38.service -f" >&2
exit 1
