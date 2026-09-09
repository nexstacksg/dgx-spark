# Agentic AI Stack on DGX Spark

Self-hosted, fully local AI stack for **~10 concurrent users** on a single **NVIDIA DGX Spark**
(GB10 Grace Blackwell, 128 GB unified memory, ARM64). It provides:

- **Text chat, tool calling & autonomous agents** — [Hermes Agent](https://github.com/nousresearch/hermes-agent)
  (Nous Research) running against **Qwen3.8-27B** (NVFP4)
- **Vision** — Qwen3.8 is natively multimodal (image + video input), so agents can read
  screenshots, invoices, shipping docs directly
- **Voice (listen)** — **Qwen3-Omni-30B-A3B**: audio/video in → text out (speech synthesis
  is a separate leg, see *Voice status* below)

Everything runs on-prem, without root, as user services. No data leaves the machine.

> **Status: deployed and verified** on this machine (host `nexstack`) — chat ✓,
> reasoning separation ✓, tool calling ✓, Hermes agent loop ✓ (terminal tool test),
> Omni audio-in ✓, voice→agent delegation loop ✓ (ask_agent round trip),
> services enabled at boot with lingering ✓.
>
> **2026-08-22:** agent model migrated from Qwen3.6-35B-A3B (FP8) to **Qwen3.8-27B
> (NVFP4)**, and the startup race that was silently killing both servers was fixed —
> see *Startup ordering* below.
>
> **2026-09-09:** agent server moved from the vLLM nightly to **vLLM 0.29.0 stable**
> (`.venv-0.29`, flashinfer 0.6.18). Re-verified the same evening: chat ✓, reasoning
> separation ✓, tool calling ✓, Hermes chat + terminal-tool loop ✓, warm restart to
> `/v1/models` in 250 s with zero crash-retries ✓. Two new traps were found and fixed on
> the way — unbounded FlashInfer JIT builds OOM-killing the box (`MAX_JOBS=2`) and a
> cold start that now takes ~12 min (health-check grace raised to 1800 s). The
> boot-time CUDA PATH fix (below) has not yet been exercised across a reboot; check
> the restart-job count after the next one.

---

## Architecture

```
                                  ┌─────────────────────────────────────────┐
                                  │              DGX Spark (128 GB)         │
                                  │                                         │
  Slack / WhatsApp / Telegram ─┐  │  ┌───────────────────────────────────┐  │
  Discord / Signal / CLI       │  │  │  Hermes Agent (gateway + agents)  │  │
                               ├──┼──▶  skills, memory, MCP tools,       │  │
  Web UI / API clients ────────┘  │  │  databases, internal APIs         │  │
                                  │  └────────────────┬──────────────────┘  │
                                  │                   │ OpenAI-compatible   │
                                  │                   ▼                     │
                                  │  ┌───────────────────────────────────┐  │
                                  │  │  vLLM :8000  (qwen38.service)     │  │
                                  │  │  Qwen3.8-27B (Unsloth NVFP4)      │  │
                                  │  │  text + image + video, tools      │  │
                                  │  └────────────────▲──────────────────┘  │
                                  │                   │ agent API as a      │
                                  │                   │ tool (deep tasks)   │
  Mic / speaker clients ──────────┼──▶ ┌──────────────┴──────────────────┐  │
                                  │    │  vLLM :8091 (qwen-omni.service) │  │
                                  │    │  Qwen3-Omni-30B-A3B (FP8 @load) │  │
                                  │    │  audio/video in → text out      │  │
                                  │    └─────────────────────────────────┘  │
                                  └─────────────────────────────────────────┘
```

**Division of labour:** the Omni server handles voice sessions and simple actions itself
(native function calling). For heavy agent work during a voice session ("check the delivery
status of PO 4521"), the voice handler forwards an `ask_agent` tool call to the port-8000
side — see `scripts/voice_demo.py`.

---

## Models

| Role | Model | Quant | Port | Service |
|---|---|---|---|---|
| Agents, chat, vision | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | [NVFP4 (Unsloth Dynamic)](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) | 8000 | `qwen38.service` |
| Voice (audio understanding) | [Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | FP8 at load (`--quantization fp8`; no official FP8 repo exists) | 8091 | `qwen-omni.service` |

Weights live in `~/models/`. Both are Apache 2.0. Qwen3.8-27B is a **dense** 27B with a
hybrid attention stack — 64 layers laid out as 16 x (3 x Gated DeltaNet -> 1 x Gated
Attention) — natively multimodal, 262k native context, and it ships a trained MTP head.

**Why NVFP4 and not the official FP8 build?** On the Spark's ~273 GB/s unified memory,
decode is bandwidth-bound: tokens/sec tracks how many bytes of weights get read per token.
NVFP4 is 23.4 GB against FP8's 30.9 GB, and 30.9/23.4 = 1.32 — which is almost exactly the
+29-34% throughput measured on this hardware ([DGX Spark benchmark thread][bench]). The
speedup is arithmetic, not benchmark noise.

The usual objection to 4-bit is that tool calling degrades first — malformed JSON arguments
are the canary. Unsloth's Dynamic V3.0 quant sidesteps most of that by placing the 4-bit
only where it is cheapest:

| Precision | Layers |
|---|---|
| NVFP4 (4-bit, group size 16) | `mlp.gate/up/down_proj` for layers 0-55 |
| FP8 | all attention (`self_attn.*`, `linear_attn.*`), `lm_head`, layers 56-63 MLP |
| BF16 (untouched) | entire vision tower (27 blocks + merger), MTP head |

Token selection (`lm_head`), every attention layer, and the final 8 layers stay at 8 bit,
which is where tool-call formatting actually lives. The quant also carries a calibrated
static FP8 KV cache scheme, which is why KV usage drops from ~15% to ~5.6%.

Two caveats worth holding: it is a community quant rather than an official Qwen build, and
it is **vLLM-only** — the model card notes SGLang cannot load it because `lm_head` is FP8.
If tool-call reliability ever regresses, `Qwen/Qwen3.8-27B-FP8` is a drop-in fallback: change
`MODEL=` in `scripts/serve-qwen38.sh` and drop `--kv-cache-dtype fp8`.

**What changed versus Qwen3.6-35B-A3B.** The old agent model was an MoE with ~3B active
parameters; this one activates all 27B every token. That is roughly 8x more weight traffic
per token, and single-stream decode drops to ~10-12 tok/s. Aggregate throughput across
concurrent users is unaffected (~134 tok/s at 16-way) because batching amortises the weight
reads — this costs one user waiting, not the fleet. The trade is worth it because Qwen3.8
targets long-horizon agentic work directly: a model that finishes in 4 turns at 11 tok/s
beats one that flails for 15 turns at 50. MTP speculative decoding (below) claws back part
of the gap.

[bench]: https://forums.developer.nvidia.com/t/qwen3-8-27b-on-dgx-spark-using-vllm-nvfp4-vs-fp8-performance/380258

---

## How it runs (as built)

Everything is **native, rootless, user-space** — no Docker (the login user is not in the
docker group), no sudo:

- **Python**: uv-managed CPython 3.12 in `.venv/` — NOT the system Python. The system
  Python has no dev headers (`python3-dev` needs sudo) and Triton compiles a CUDA stub
  at startup that requires `Python.h`. The uv-managed runtime bundles its headers.
- **vLLM** with torch cu130 — official aarch64 CUDA wheels from PyPI. Two venvs: the
  agent server runs `.venv-0.29` (vLLM 0.29.0, flashinfer 0.6.18, since 2026-09-09 —
  DFlash2 speculative decoding landed in stable 0.28.0, which retired the 2026-08-22
  nightly `.venv-nightly`); omni stays on `.venv` (0.25.0). Do not upgrade `.venv`.
- **FFmpeg shared libs** in `~/.local/ffmpeg-shared/lib` (BtbN 7.1 build): vLLM's
  torchcodec dlopens `libavutil.so.*` at import; there is no system FFmpeg. The serve
  scripts put this dir on `LD_LIBRARY_PATH`. A static `ffmpeg` binary (for Hermes voice
  messages) is in `~/.local/bin`.
- **systemd user services**, enabled at boot, with `loginctl enable-linger` set so they
  run without an active login session. Because they start *before* GNOME login they do not
  see `/etc/profile.d` PATH additions — the serve scripts export `/usr/local/cuda/bin`
  themselves (see "First start after a reboot crashes" below):

```bash
systemctl --user status qwen38.service qwen-omni.service   # health
journalctl --user -u qwen38.service -f                     # logs
systemctl --user restart qwen38.service                    # after config edits
```

Serve configs live in `scripts/serve-qwen38.sh` and `scripts/serve-omni.sh`. Flags that
were discovered the hard way and should not be changed casually:

- `--tool-call-parser qwen3_xml` — Qwen3.8 emits XML-style tool calls
  (`<tool_call><function=...><parameter=...>`), **not** the classic Hermes JSON format.
  With the wrong parser, tool calls silently arrive as plain text and agents break.
  Community recipes for this model often say `qwen3_coder`; in vLLM 0.25.0 the two are
  aliases for the same `Qwen3EngineToolParser`, so either works.
- `--kv-cache-dtype fp8` — the NVFP4 checkpoint carries calibrated static KV scales
  (`kv_cache_scheme` in `config.json`). This flag activates them; without it the KV cache
  falls back to 16-bit and the ~5.6% KV footprint becomes ~15%.
- `--speculative-config '{"method":"dflash",...}'` — switched 2026-08-22 from the model's
  built-in MTP head to a DFlash2 draft model (`~/models/Qwen3.8-27B-DFlash2`,
  `num_speculative_tokens: 7`; needs vLLM >= 0.28). This recovers part of the
  decode speed lost by moving from a 3B-active MoE to a dense 27B, but the speedup is
  **content-dependent** — measured on this box: ~55 tok/s short answers, ~38 tok/s code,
  ~20 tok/s free prose (raw non-speculative decode is ~10-12). Check acceptance in the
  logs before assuming it helps on a given workload; drop the flag to disable.
- `--default-chat-template-kwargs '{"reasoning_effort": "medium"}'` — Qwen3.8's chat
  template defaults `reasoning_effort` to **`xhigh`** when the client doesn't specify it,
  so even a "hi" burns hundreds of thinking tokens before the visible reply (~20 s at
  single-stream speeds). This resets the server default to `medium`; clients can still
  send `reasoning_effort` (or `enable_thinking: false`) per request to override either way.
- `CUTE_DSL_ARCH=sm_121a` — GB10 reports `sm_121`, but CUTLASS DSL kernels want the `a`
  suffix. Harmless if unused.
- `MAX_JOBS=2` (agent script) — caps FlashInfer's JIT `ninja -j`. Unbounded, a fresh
  flashinfer version compiles the NVFP4 CUTLASS GEMM with ~64 parallel `cicc` processes at
  ~4 GB each, which OOM-killed the whole box on 2026-09-09 (first start of vLLM 0.29.0 /
  flashinfer 0.6.18 with omni and Chrome resident). Only slows first-start compiles;
  cached kernels under `~/.cache/flashinfer/<ver>/121a/cached_ops/` are unaffected.
- `--reasoning-parser qwen3` — Qwen3.8 is a thinking model; without this, chain-of-thought
  leaks into `content` instead of `reasoning_content`.
- `export PATH="$VENV/bin:$PATH"` in the scripts — vLLM's startup compile subprocesses
  `ninja`; systemd services don't activate the venv.
- `--max-model-len` / `--max-num-seqs` / `--gpu-memory-utilization` (0.50 agent / 0.30
  omni) are the shared-memory budget. Raising one means lowering another; 128 GB is
  unified with the OS (~115 GB usable). vLLM pre-allocates its pools at startup, so
  high memory usage at idle is normal and constant — it is not a leak. RAM and GPU
  memory are the SAME pool on the Spark (no separate VRAM), so `free` showing ~110 GiB
  used with both servers idle is expected. If the desktop ever feels starved, drop the
  agent server to 0.45 first (~6 GiB back to the OS). NVFP4 freed ~11 GiB versus the old
  Qwen3.6 FP8 weights, which is what paid for raising `--max-model-len` from 65536 to
  131072. Only the 16 full-attention layers hold a per-token KV cache (4 KV heads x 256
  head dim, FP8 = ~32 KB/token), so a 131072-token sequence costs ~4.2 GB; the 48 linear
  attention layers keep a fixed-size recurrent state instead.
- Omni only: `vllm[audio]` extras (av/soundfile/soxr) are required for audio input —
  without them requests 400/500 at runtime, not at startup.
- Omni only: `--chat-template $MODEL/chat_template.jinja` — the tool-capable template
  ships in `chat_template.json` (processor level), which vLLM's tool path doesn't read;
  it was extracted to a `.jinja` file the server can load. Without it, any request with
  `tools` fails with a "default chat template no longer allowed" 400.

### Startup ordering (do not remove the readiness gate)

The two servers **must not load their weights at the same time**. vLLM sizes its KV cache by
profiling free memory while the model loads; if a neighbour allocates ~30 GB inside that
window, the profiler charges it against *this* server's budget and the KV cache comes out
negative:

```
INFO  [gpu_worker.py:538] Available KV cache memory: -5.49 GiB
ERROR ValueError: No available memory for the cache blocks
```

Both servers died exactly this way on 2026-08-22. `After=qwen38.service` alone does **not**
prevent it: with `Type=simple`, systemd considers a unit started the moment the process
forks, not when the model finishes loading, so `After=` orders the launch and the two
5-minute weight loads still overlap completely.

The fix is `ExecStartPre` in `qwen-omni.service`, which blocks on real HTTP readiness:

```ini
ExecStartPre=%h/Documents/GitHub/dgx-spark/scripts/wait-for-vllm.sh http://127.0.0.1:8000/v1/models 1800
```

The `0.50 / 0.30` split is a **global** budget, not a per-process reservation — it only
holds if the loads are serialised.

### Cold start takes ~9 minutes (~12 with a cold FlashInfer cache), warm restart ~4

Budget for it. A cold `qwen38.service` start is roughly:

| Phase | Time |
|---|---|
| Weight load (21.81 GiB from EXT4) | ~26 s |
| `torch.compile` main graph | ~66 s |
| Initial profiling / warmup | ~134 s |
| `torch.compile` draft head + warmup | ~120 s |
| KV cache sizing + CUDA graph capture | remainder |

The two compile passes are the price of `--speculative-config` (MTP or DFlash2); without that flag
startup is roughly half. Results are cached under `~/.cache/vllm/torch_compile_cache/`, so
**restarts are much faster than the first run** — but a vLLM upgrade or a flag change
invalidates the cache and you pay full price again. A flashinfer upgrade adds a third
cost on top: ~25 attention/GEMM/sampling kernels are JIT-compiled with `nvcc` into
`~/.cache/flashinfer/<ver>/121a/cached_ops/`. Measured 2026-09-09 (vLLM 0.29.0, all
caches cold, `MAX_JOBS=2`, omni resident): **~12 min** from start to `/v1/models`.
With every cache warm, a `systemctl --user restart qwen38.service` measured **250 s**
to `/v1/models` on 2026-09-09 (weights ~30 s, the rest is compile-cache load, profiling
and CUDA-graph capture). Clients — including Hermes — get `APIConnectionError` for that
whole window; that is the restart, not a broken server.

Nothing listens on :8000 for that entire window. Clients get connection refused, not a
"still loading" response.

### First start after a reboot crashes, then the retry succeeds

On every reboot the **first** `qwen38.service` attempt dies ~3.5 min in, during memory
profiling, and only systemd's `Restart=on-failure` retry (15 s later) comes up. Net effect:
nothing listens on :8000 for ~8 minutes after boot instead of ~4, and `qwen-omni` queues
behind it. The error is:

```
AttributeError: '_OpNamespace' 'vllm' object has no attribute 'flashinfer_mm_fp4'
RuntimeError: Engine core initialization failed.
```

Root cause (diagnosed 2026-09-09; same crash-then-retry pattern in the 2026-09-02 boot log):
`/usr/local/cuda/bin` is only put on PATH by `/etc/profile.d/nv_paths.sh`, which login
shells read but the boot-time `systemd --user` manager (started by linger, before GNOME
login) does not — `/etc/environment` has no CUDA dir. vLLM's `has_flashinfer()` requires
`nvcc` on PATH (no `flashinfer-cubin` package is installed), so on the first attempt it
returns False and the `vllm::flashinfer_mm_fp4` custom op is never registered — but the
cached AOT-compiled graph still calls it. Once GNOME logs the user in it imports the shell
environment into the user manager, so the retry inherits the CUDA path and works.

Fix (applied 2026-09-09): both serve scripts now export `/usr/local/cuda/bin` explicitly.
Do not rely on the user manager's imported environment. Verify with:

```bash
journalctl --user -u qwen38.service -b | grep -c 'Scheduled restart job'   # want 0
```

Without `nvcc` vLLM also silently picks the slower `CutlassNvFp4LinearKernel` GEMM instead
of `FlashInferCutlassNvFp4LinearKernel`, so a server that *does* come up this way is slower.

**Stale torch-compile cache** is the other thing that produces the same AttributeError: a
vLLM/flashinfer reinstall can leave a cached graph pointing at an op that no longer
registers, and then it fails on **every** attempt, not just the first. If the retry also
fails after touching package versions, delete the AOT cache and let it rebuild (~4 min):

```bash
rm -rf ~/.cache/vllm/torch_compile_cache/torch_aot_compile/<hash-for-this-config>
```

### Health check (failures are otherwise invisible)

When the engine subprocess dies at startup, the vLLM *frontend* process can survive it.
systemd then reports `active (running)` with `ExecMainStatus=0` and `NRestarts=0`, so
`Restart=on-failure` never fires and nothing listens on the port — the stack looks healthy
while every client gets a connection error. `vllm-healthcheck.timer` runs every 5 minutes,
curls each `/v1/models`, and restarts any unit that is active but not answering:

```bash
systemctl --user list-timers vllm-healthcheck.timer
journalctl --user -u vllm-healthcheck.service    # what it has restarted
```

⚠️ The third argument to `healthcheck-vllm.sh` is a **startup grace period** (1800s for
both units since 2026-09-09; was 1200s agent) and must stay longer than a cold start. Without it the health check restarts the
server *during* its 9-minute load, forever, and it never comes up. This happened on
2026-08-22: the kill landed inside FastAPI route registration and surfaced as

```
SchemaError: Error building "literal" validator:
KeyboardInterrupt: terminated
```

which reads like a pydantic/fastapi dependency conflict and is nothing of the sort. If you
ever see a pydantic schema error from the API server, check
`journalctl --user -u vllm-healthcheck.service` before touching any package versions.

Smoke tests:

```bash
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"hello"}]}'
```

---

### Running it by hand

Nothing needs to be started manually after a reboot — both units are enabled and
lingering is on. Day to day:

```bash
systemctl --user status qwen38.service qwen-omni.service   # is it up?
journalctl --user -u qwen38.service -f                     # watch the agent server load
systemctl --user restart qwen38.service                    # after editing its serve script
systemctl --user stop qwen38.service qwen-omni.service     # free the memory
systemctl --user start qwen38.service qwen-omni.service    # bring both back (omni waits for :8000)
```

A unit stopped while it was still loading shows `failed` (result `timeout` or `signal`)
instead of `inactive`, because the stop had to be force-killed. That is cosmetic; systemd
does not auto-restart after a manual stop, and `start` works normally.

To run a server in the foreground instead — for example to see startup output directly —
stop its unit first so the port and memory are free, then run its serve script. The
scripts set up PATH, `LD_LIBRARY_PATH` and the venv themselves, so no activation is
needed. Ctrl+C stops it; the two servers must still not load at the same time.

```bash
systemctl --user stop qwen38.service
~/Documents/GitHub/dgx-spark/scripts/serve-qwen38.sh
```

Budget ~4 min for a warm start and up to ~12 for a cold one; the port refuses connections
until the log prints `Application startup complete`.

---

## Hermes Agent (as built)

Installed from PyPI as a uv tool (`uv tool install hermes-agent`, v0.18.2) — binaries
`hermes`, `hermes-acp`, `hermes-agent` in `~/.local/bin`.

Configuration that actually routes to the local server (learned by reading the source —
the docs' env-var-only description is incomplete):

**`~/.hermes/config.yaml`** (single source of truth for model routing):

```yaml
model:
  default: qwen3.8-27b
  provider: custom
  base_url: http://localhost:8000/v1
  api_key: local-not-used
custom_providers:
  - name: spark
    base_url: http://localhost:8000/v1
    key_env: OPENAI_API_KEY
    model: qwen3.8-27b
```

The model name must match `--served-model-name` in `scripts/serve-qwen38.sh` in **both**
places. When swapping models, change the serve script and these two keys together — a
mismatch fails as a confusing 404 from the vLLM side, not as a config error.

**`~/.hermes/.env`** (chmod 600):

```bash
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=local-not-used
```

⚠️ Do **NOT** set `HERMES_INFERENCE_MODEL` in `.env` — an env-provided model counts as an
"explicit model request" and triggers provider auto-detection, which routes qwen-named
models to hosted catalogs and fails with `HTTP 401: Missing Authentication header`.

Verified working:

```bash
hermes -z "Say READY if you can hear me."          # → READY (via local vLLM)
hermes -z "Run 'uname -m' and tell me the output." # → aarch64 (agent loop + terminal tool)
hermes                                             # interactive chat
```

### Connecting the 10 users (gateway)

The gateway serves Telegram, Discord, Slack, WhatsApp, Signal and CLI from one process:

```bash
hermes gateway setup    # interactive — needs YOUR platform bot tokens
hermes gateway start
```

This is the one step that couldn't be automated: it requires bot tokens/credentials for
whichever platforms the users are on (e.g. a Slack app token, Telegram bot token).

---

## Voice status

- **Working:** the Omni server accepts audio (and image/video) input over the standard
  chat-completions API and reasons over it — this is vLLM serving the Omni *thinker*.
- **Not served by vLLM:** the *talker* (streaming speech output). For spoken replies,
  pipe the text reply into a TTS of your choice, or run Qwen's own serving recipe from
  [QwenLM/Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni) (transformers-based, heavier).
  Benchmark before promising simultaneous voice sessions.
- `scripts/voice_demo.py` demonstrates the full session flow: WAV in → Omni → optional
  `ask_agent` delegation to port 8000 → final speakable text.

---

## Known constraints

1. **One GPU, shared everything.** Both servers share GB10 compute and memory bandwidth;
   no MIG. Keep `--max-num-seqs` capped on the agent server; expect some voice latency
   wobble under peak agent load. They also must not *start* together — see
   *Startup ordering*.
2. **Dense agent model.** Qwen3.8-27B activates all 27B parameters per token, unlike the
   MoE it replaced. Single-stream decode is ~10-12 tok/s and that is a hardware bandwidth
   limit, not a tuning problem. Throughput scales with concurrency; per-user latency does
   not improve. If interactive latency matters more than answer quality for some workload,
   the old `qwen36.service` unit and its FP8 weights are still on disk — re-enable with
   `systemctl --user enable --now qwen36.service` after disabling `qwen38.service`.
3. **ARM64.** Anything added later (rerankers, TTS, embedders) needs aarch64 builds.
   Check the [DGX Spark playbooks](https://build.nvidia.com/spark).
4. **No sudo assumed.** The whole stack lives in `$HOME`. If you get sudo one day:
   `sudo usermod -aG docker kenling` re-opens the NVIDIA container path, and
   `sudo apt install ffmpeg python3-dev` removes two workarounds.

## Upgrade path

- **More load:** a second DGX Spark links via ConnectX-7 — voice on one box, agents on
  the other gives real isolation with zero architecture changes.
- **Better models:** each service swaps independently — edit the `MODEL=` line in its
  serve script, download new weights to `~/models/`, restart the service.
