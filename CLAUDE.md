# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Operations repo for a self-hosted LLM stack on a single NVIDIA DGX Spark (GB10, ARM64,
128 GB unified memory, no sudo, no Docker for the main stack). It contains the vLLM serve
scripts and helper scripts that the user's systemd units execute, plus `README.md`, which is
the detailed as-built record (rationale for every flag, incident history, memory budget).
There is no build, lint, or test suite. Changes are verified by restarting a service and
checking its logs and HTTP endpoint. Read `README.md` before changing any serve flag; most
flags were set after a specific outage and the README says which one.

## Layout

- `scripts/serve-qwen38.sh` — agent/vision server, Qwen3.8-27B NVFP4 on :8000
  (`qwen38.service`). Runs from `.venv-0.29` (vLLM 0.29.0).
- `scripts/serve-omni.sh` — voice (audio-in) server, Qwen3-Omni-30B on :8091
  (`qwen-omni.service`). Runs from `.venv` (vLLM 0.25.0). Do not upgrade this venv.
- `scripts/serve-qwen36.sh` — previous agent model (MoE, faster single-stream). Unit is on
  disk but disabled; it also binds :8000, so never enable it alongside `qwen38.service`.
- `scripts/wait-for-vllm.sh` — `ExecStartPre` readiness gate in `qwen-omni.service`.
- `scripts/healthcheck-vllm.sh` — run by `vllm-healthcheck.timer` every 5 min.
- `scripts/sglang-pilot-{start,stop}.sh` — swap the agent server for an SGLang pilot in a
  sibling repo (`~/Documents/GitHub/dgx-spark-qwen38`). Foreground only; reboot restores vLLM.
- `scripts/voice_demo.py` — reference client for the voice → `ask_agent` delegation flow.
- `.venv`, `.venv-0.29`, `.venv-nightly` — uv-managed CPython 3.12 venvs, gitignored.
  `.venv-nightly` (0.26.1rc) is superseded by `.venv-0.29`.

Things outside the repo that the scripts depend on: systemd units in
`~/.config/systemd/user/`, weights in `~/models/`, FFmpeg shared libs in
`~/.local/ffmpeg-shared/lib`, Hermes config in `~/.hermes/config.yaml`, compile caches in
`~/.cache/vllm/torch_compile_cache/`.

## Common commands

```bash
systemctl --user status qwen38.service qwen-omni.service
journalctl --user -u qwen38.service -f            # live logs
systemctl --user restart qwen38.service           # after editing a serve script
journalctl --user -u vllm-healthcheck.service     # what the health check restarted
journalctl --user -u qwen38.service -b | grep -c 'Scheduled restart job'   # want 0

curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"hello"}]}'

hermes -z "Say READY if you can hear me."         # end-to-end via local vLLM
.venv/bin/python scripts/voice_demo.py question.wav
```

Run vLLM or Python from a venv's `bin/`, never the system Python (no dev headers).

## Rules that are easy to break

- **The two servers must not load weights concurrently.** vLLM sizes its KV cache from
  free memory during load, so a neighbour loading at the same time makes the budget go
  negative. `After=` does not serialise this; only the `ExecStartPre` gate does. When
  restarting both by hand, start `qwen38`, wait for :8000 to answer, then start `qwen-omni`.
- **`--gpu-memory-utilization` is a global budget** (0.50 agent + 0.30 omni) on memory shared
  with the OS. Raising one means lowering another. High idle memory usage is pre-allocation,
  not a leak.
- **Cold start is ~9 minutes** (~12 with a cold FlashInfer JIT cache) and the port is closed the whole time. The health check grace
  periods (1800 s for both units) must stay longer than a cold start, or the checker
  restarts the server mid-load forever. A vLLM upgrade or a flag change invalidates the
  compile cache and costs a full cold start again.
- **Serve scripts must export PATH themselves** (`$VENV/bin` for `ninja`, `/usr/local/cuda/bin`
  for `nvcc`). The boot-time user manager never sees login-shell PATH. Missing `nvcc` produces
  `AttributeError: ... no attribute 'flashinfer_mm_fp4'` on the first start after every reboot.
  The same error on every attempt means a stale AOT cache; delete the hash dir under
  `torch_compile_cache/torch_aot_compile/`.
- **`MAX_JOBS` stays capped** in `serve-qwen38.sh`. Unbounded FlashInfer JIT builds
  OOM-killed the whole box on 2026-09-09.
- **`--served-model-name` must match `~/.hermes/config.yaml`** in both `model.default` and
  `custom_providers[].model`. A mismatch surfaces as a 404 from vLLM, not a config error.
  Never set `HERMES_INFERENCE_MODEL` in `~/.hermes/.env` (it causes a 401 via provider
  auto-detection).
- **Tool-call and reasoning parsers are model-specific**: `qwen3_xml` + `qwen3` for Qwen3.8,
  `hermes` for Omni. Wrong parser means tool calls arrive as plain text with no error.
- **The NVFP4 checkpoint is vLLM-only** (FP8 `lm_head`); SGLang cannot load it. Fallback is
  `Qwen/Qwen3.8-27B-FP8` with `--kv-cache-dtype fp8` removed.
- A pydantic `SchemaError` from the API server is almost always the health check killing a
  server mid-startup. Check the healthcheck journal before touching package versions.

## When you change something

- Editing a serve script: restart the unit, tail the journal until `/v1/models` answers, then
  run the chat-completions smoke test. Budget for the cold start.
- Changing the model or its served name: update the serve script, `~/.hermes/config.yaml`,
  and any hardcoded model names in `scripts/voice_demo.py` together.
- Record the reason for any new flag as a comment in the script and a note in `README.md`;
  that convention is how this repo avoids re-learning outages.
