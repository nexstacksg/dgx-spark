#!/usr/bin/env bash
# Restart a vLLM unit if its HTTP endpoint stops answering.
#
# WHY THIS EXISTS: when the engine subprocess dies at startup, the vLLM
# *frontend* process can stay alive. systemd then reports the unit as
# "active (running)" with ExecMainStatus=0 and NRestarts=0, so Restart=on-failure
# never fires and nothing listens on the port. The failure is invisible until a
# client gets a connection error.
#
# GRACE PERIOD (do not remove): a cold start on this box takes ~9 minutes —
# weight load, then torch.compile of the main graph, warmup, then a second
# torch.compile for the MTP draft head. During all of that the unit is already
# "active" but the HTTP port is not open yet. Without the grace period this
# script restarts the server mid-load, forever. That happened on 2026-08-22:
# the kill landed during FastAPI route registration and surfaced as a bogus
# pydantic SchemaError, which looks nothing like the actual cause.
set -euo pipefail

UNIT="${1:?usage: healthcheck-vllm.sh UNIT URL [GRACE_SECONDS]}"
URL="${2:?usage: healthcheck-vllm.sh UNIT URL [GRACE_SECONDS]}"
GRACE="${3:-1200}"

# Only police units that are supposed to be up.
systemctl --user is-active --quiet "$UNIT" || exit 0

# Skip units still inside their startup window.
enter=$(systemctl --user show "$UNIT" -p ActiveEnterTimestamp --value)
if [[ -n "$enter" ]]; then
  enter_epoch=$(date -d "$enter" +%s 2>/dev/null || echo 0)
  if (( enter_epoch > 0 )); then
    uptime=$(( $(date +%s) - enter_epoch ))
    if (( uptime < GRACE )); then
      echo "healthcheck: $UNIT up ${uptime}s (< ${GRACE}s grace) — still starting, skipping"
      exit 0
    fi
  fi
fi

if curl -sf -m 10 "$URL" >/dev/null 2>&1; then
  exit 0
fi

echo "healthcheck: $UNIT active but $URL not answering after ${GRACE}s — restarting" >&2
systemctl --user restart "$UNIT"
