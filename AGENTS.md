# Repository Guidelines

## Project Structure & Module Organization

This repository operates a local LLM stack on NVIDIA DGX Spark (ARM64, 128 GB unified memory). `README.md` documents architecture, deployment, and incident history; read it before changing serving flags.

- `scripts/serve-qwen38.sh` and `serve-omni.sh` launch the agent/vision and audio servers on ports 8000 and 8091.
- `scripts/wait-for-vllm.sh` and `healthcheck-vllm.sh` handle readiness and recovery.
- `scripts/sglang-pilot-{start,stop}.sh` switch to an experimental backend; `serve-qwen36.sh` is the legacy launcher.
- `scripts/voice_demo.py` demonstrates audio input and agent delegation.

There are no dedicated test or asset directories. Model weights live outside Git in `~/models/`; user service units live in `~/.config/systemd/user/`.

## Build, Test, and Development Commands

There is no build pipeline. Use the existing uv-managed Python environments: `.venv-0.29` for the agent server and `.venv` for Omni; do not upgrade Omni's environment.

```bash
# Check shell syntax without launching services
for script in scripts/*.sh; do bash -n "$script" || break; done
# Inspect deployed services and agent logs
systemctl --user status qwen38.service qwen-omni.service
journalctl --user -u qwen38.service -f
# Apply agent launcher changes
systemctl --user restart qwen38.service
# Check HTTP readiness
curl -f http://localhost:8000/v1/models
# Exercise voice input
.venv/bin/python scripts/voice_demo.py question.wav
```

## Coding Style & Naming Conventions

Match existing style: Bash uses two-space indentation, `set -euo pipefail`, quoted expansions, and uppercase configuration variables. Python uses four-space indentation and snake_case functions. Use descriptive, hyphenated shell filenames. No formatter or linter is configured. Explain new flags in script comments and `README.md`.

## Testing Guidelines

No automated framework or coverage threshold exists. After runtime changes, verify HTTP readiness, inspect journals, and run the README chat-completions smoke test. Check port 8091 for Omni changes. For voice delegation, first align the demo's legacy `qwen3.6-35b-a3b` model name with the active server.

## Commit & Pull Request Guidelines

History uses imperative subjects such as “Add SGLang pilot switch scripts.” Keep commits focused. PRs should describe the operational change, rationale, affected services, validation results, and any related issue or external configuration changes.

## Operational Constraints

Preserve serialized model loading, readiness gates, health-check grace periods, explicit CUDA/venv PATH exports, and the `MAX_JOBS` cap. Budget roughly nine minutes for cold starts. Coordinate shared-memory budgets and keep served model names synchronized with Hermes configuration and clients.
