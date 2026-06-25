# Gemma 4 26B-A4B (NVFP4 + DFlash) — Self-Hosted Inference via vLLM on DGX Spark

[![vLLM](https://img.shields.io/badge/vLLM-0.23.0%20sm__121a-blue)](https://github.com/vllm-project/vllm)
[![Model](https://img.shields.io/badge/model-Gemma_4_26B--A4B-informational)](https://huggingface.co/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-lightgrey)](LICENSE)
[![arch](https://img.shields.io/badge/arch-arm64%20(GB10)-lightgrey)](#)

A production-ready vLLM deployment wrapper for **[Gemma 4 26B-A4B](https://huggingface.co/AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4)** — an NVFP4-quantized MoE (≈3.8B active) with **DFlash speculative decoding**, tuned for a single **NVIDIA DGX Spark (GB10, sm_121)**.

Bundles a ready-to-run Docker launch, the chat template, start/stop scripts, and a reproducible concurrency-scaling benchmark — spin up a fully OpenAI-compatible server in minutes.

> **TL;DR — 65.6 tok/s single stream (p50 of 16, mean 70.3), 406 tok/s at 8 concurrent sessions, on one desk-side box.**

---

## ✨ Key Features

| Feature | Details |
|---|---|
| **Model** | `AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4` — NVFP4 MoE (26B total / ~3.8B active, 128 experts top-8 + 1 shared) |
| **Inference Engine** | vLLM `0.23.0` sm_121a build (`ghcr.io/aeon-7/aeon-vllm-ultimate`) |
| **Speculative Decoding** | **DFlash**, 5 speculative tokens, `flash_attn` drafter backend |
| **Quantization** | NVFP4 — MLP + MoE experts 4-bit; attention + vision tower BF16 |
| **Context Window** | **262,144 tokens (full native 256K)** |
| **OpenAI-Compatible API** | `/v1/chat/completions`, `/v1/completions`, `/v1/models` |
| **Vision Support** | Multi-modal image input |
| **Tool Use** | `gemma4` tool-call parser, auto tool choice enabled |
| **Thinking/Reasoning** | Configurable via `enable_thinking`; `gemma4` reasoning parser |
| **Prefix Caching** | `--enable-prefix-caching` |
| **Chunked Prefill** | `--enable-chunked-prefill` |
| **Self-warming** | `start.sh` fires a warm-up request to absorb the one-time compile cliff |

---

## 📊 Performance

> **v1 (2026-06-24): +10% decode speedup over the pre-v1 baseline.** Validated on a 16-iteration,
> 3-warmup bench (mixed prompts, 256 max-tok, T=0, direct to `localhost:8000`). The win came
> from tuning the DFlash drafter's `num_speculative_tokens` from 10 to 5; the full sweep (3, 5,
> 10 + several `--max-num-batched-tokens` values + `--async-scheduling`) is in the lab
> (`lab/2026-06-24/`) and the rejected experiments are in [Troubleshooting](#-troubleshooting).

**v1 headline metrics (c=1, 256 max-tok, T=0, mixed prompts, 3 warmup × 16 iter):**

| Metric | Pre-v1 | v1 | Delta |
|---|---:|---:|---:|
| decode tok/s **p50** | 59.7 | **65.6** | **+9.9%** |
| decode tok/s mean | 68.2 | 70.3 | +3.1% |
| decode tok/s min (floor) | 46.9 | 54.0 | **+15.1%** |
| decode tok/s stddev | 25.1 | 18.6 | **−26%** (tighter) |
| TTFT p50 (ms) | 115 | 107 | −7% |
| TPOT p50 (ms) | 16.33 | 14.90 | **−8.8%** |

### Real-world task latency

What actual extraction/parsing work costs — moderate input, **short structured output** (the common
case). Reproduce with [`bench/realtask.py`](bench/realtask.py).

| Task | Input | Output | Latency |
|---|---:|---:|---:|
| Parse 6k-token Excel/CSV → JSON | 5,998 tok | 23 tok | **1.4 s** |
| Excel/CSV → single value | 5,988 tok | 4 tok | **1.3 s** |
| Image → label (vision) | image | 2 tok | **1.0 s** |

Real tasks finish in **~1 s** because they return a few tokens. The throughput numbers below force a
512-token answer — that's a stress test, *not* what extraction/parsing feels like.

### Concurrency scaling

Each request generates **512 output tokens** (deterministic, `ignore_eos`, T=0) — so a batch of N
concurrent requests = N × 512 tokens. Measured on the DGX against `localhost:8000` at the live
**`--max-model-len 262144` (256K)** config. Reproduce with
[`bench/scaling.py`](bench/scaling.py); raw data in [`bench/results.json`](bench/results.json);
visual in [`assets/report-card.html`](assets/report-card.html).

| Sessions | Cumulative gen tok/s | Per-session avg | Scaling factor | Efficiency |
|---:|---:|---:|---:|---:|
| 1 | 65.6 | 65.6 | — | baseline |
| 2 | 129.3 | 64.7 | 1.90× | 95.0% |
| 4 | 246.0 | 61.5 | 3.61× | 90.2% |
| 8 | 403.0 | 50.5 | 5.92× | 74.0% |

*c=8 number re-bench pending — last measured 2026-06-21 with pre-v1 settings.*

Near-linear to 4 sessions (90% efficiency); DFlash's speculative gain tapers as the GPU fills and
the box trends toward the GB10 bandwidth bound (273 GB/s) — so 8 concurrent users still get ~50
tok/s each, 403 tok/s aggregate. (These are 512-token generations to measure steady throughput;
real short-output tasks finish in ~1 s — see above.) c=2/4/8 are pre-v1 (last measured
2026-06-21); a c=2/4/8 re-bench with the v1 settings is a known TODO.

**What changed (pre-v1 → v1):**

| Setting | Pre-v1 | v1 | Why |
|---|---|---|---|
| `num_speculative_tokens` | 10 | **5** | drafter forward-pass waste on positions 5–9; n=5 lifts the floor by 15% and tightens variance without losing the high-acceptance ceiling (n=3 was tested and rejected — kills ceiling by 12.5%) |
| `--max-num-batched-tokens` | 32768 | **16384** | equivalent throughput, eliminates vLLM's informational 7936 chunked-prefill warning (8192 was tested and rejected — chunked-prefill clamp regresses p50 by ~3.4%) |
| `--async-scheduling` | (implicit default) | **explicit** | no-op on vLLM 0.23+ (already default), made explicit for clarity |

### Long context (full 256K)

Verified end-to-end across depths — needle retrieved correctly at ~39k **and** ~230k tokens.
Reproduce with [`bench/longctx.py`](bench/longctx.py).

Prefill cost (TTFT) by how the context is actually used:

| Scenario | Prefill (TTFT) |
|---|---|
| **Typical agent first request (~39k tokens, cold)** | **~10 s** |
| Cold one-shot paste, 200k+ tokens, nothing cached | ~3–5 min (worst case) |
| Same context queried again (prefix-cache hit) | ~0.5 s |
| Each new turn as context grows (incremental) | only the *new* tokens |

The realistic number is the **~10 s** first request (a real harness already carries ~30–40k tokens
of system prompt + tools + history). After that, `--enable-prefix-caching` means every turn only
prefills its new chunk and re-reads are near-instant — so you pay the cold cost once, shallow. The
multi-minute figure is the rare worst case (a fresh 200k-token blob in one shot). Gemma 4's
sliding-window attention also keeps long-context KV *memory* cheap — the cost is prefill compute.

---

## 📋 Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│              NVIDIA DGX Spark (GB10, 128 GB)          │
│                                                      │
│  start.sh / stop.sh                                  │
│  chat_template.jinja   ← Gemma 4 Jinja template      │
│  .cache/huggingface/   ← model + drafter weights     │
│  .cache/vllm/          ← compile cache (fast restart)│
│  .vllm.log / .vllm.pid                               │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │ Docker: aeon-vllm-ultimate (vLLM 0.23 sm_121)│    │
│  │   Gemma 4 26B-A4B NVFP4 + DFlash drafter     │    │
│  │   OpenAI API on :8000                        │    │
│  └──────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
```

---

## 🛠️ Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| **Device** | NVIDIA DGX Spark (GB10, sm_121) | arm64 / aarch64 |
| **Memory** | 128 GB unified | ~17 GB weights + KV cache |
| **OS / driver** | Ubuntu 24.04, driver ≥ 580 | CUDA 13 |
| **Docker** | 24.0+ | with NVIDIA Container Toolkit |
| **Disk** | ~30 GB free | weights + caches |
| **curl** | any | readiness probe |

> The `aeon-vllm-ultimate` image is built for sm_121 (GB10) + RTX 50-series. It will not run on other GPUs without a rebuild.

---

## 🚀 Quick Start

### 1. Clone
```bash
git clone https://github.com/<you>/gemma-4-26b-a4b-nvfp4-dgx-spark
cd gemma-4-26b-a4b-nvfp4-dgx-spark
```

### 2. (Optional) HuggingFace token (faster downloads)
```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxx"
```

### 3. Start
```bash
./start.sh
```
This will: check Docker/curl → create cache dirs → remove any stale container → launch the
container (`--gpus all`, `--restart always`) → stream logs to `.vllm.log` → poll `/v1/models`
until ready → **fire a warm-up request** → print the OpenAI base URL.

### 4. Test
```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4",
       "messages":[{"role":"user","content":"Explain NVFP4 in one sentence."}],
       "max_tokens":256}' | jq
```

### 5. Stop
```bash
./stop.sh
```

---

## ⚙️ Configuration

All options live in [`start.sh`](start.sh). Key variables: `MODEL_ID`, `DRAFTER_ID`, `IMAGE`,
`CONTAINER_NAME`, `PORT`, `HF_TOKEN`.

### Model inference parameters (GB10-specific)

| Flag / env | Value | Why |
|---|---|---|
| `--quantization` | `compressed-tensors` | NVFP4 weights |
| `--attention-backend` | `triton_attn` | Gemma 4's heterogeneous head dims need Triton |
| `--speculative-config` | DFlash, 5 tokens, `flash_attn` (drafter: `z-lab/gemma-4-26B-A4B-it-DFlash`) | drafter **must** use `flash_attn` (`flex_attention` crashes); `num_speculative_tokens=5` (was 10) — +9.9% p50, +3.1% mean, +15.1% min (16-iter clean bench, 2026-06-24) |
| `VLLM_NVFP4_GEMM_BACKEND` | `flashinfer-cutlass` | valid output on GB10 (no native FP4 cores) |
| `VLLM_USE_FLASHINFER_MOE_FP4` | `0` | required with the cutlass GEMM path |
| `--gpu-memory-utilization` | `0.80` | dedicated box → generous KV cache |
| `--max-model-len` | `262144` | full native 256K context |
| `--max-num-seqs` | `64` | concurrent sequence cap |
| `--max-num-batched-tokens` | `16384` (was `32768`) | equivalent throughput; eliminates the vLLM 7936 chunked-prefill warning |
| `--async-scheduling` | (explicit flag) | no-op in vLLM 0.23+ (already default); documented for clarity |
| `--enable-prefix-caching` | — | KV reuse across shared prefixes |
| `--tool-call-parser` / `--reasoning-parser` | `gemma4` | native tool + thinking support |

---

## 🧩 Chat template

[`chat_template.jinja`](chat_template.jinja) is the Gemma 4 Jinja template with full
tool-calling serialization and a configurable `enable_thinking` toggle for chain-of-thought.
Mounted into the container and passed via the engine's chat-template handling.

---

## 📁 Project Structure

```
gemma-4-26b-a4b-nvfp4-dgx-spark/
├── README.md            ← this file
├── AGENTS.md            ← one-shot install guide for AI coding agents
├── start.sh             ← launch (download + serve + warm-up)
├── stop.sh              ← stop & cleanup
├── chat_template.jinja  ← Gemma 4 chat template
├── bench/
│   ├── scaling.py       ← concurrency-scaling benchmark
│   └── results.json     ← raw results (c=1/2/4/8)
├── assets/report-card.html  ← shareable results card
├── LICENSE              ← MIT (scripts)
└── .gitignore

# Runtime artifacts (auto-created, gitignored):
#   .vllm.log  .vllm.pid  .cache/
```

---

## 🐳 Docker details

| Property | Value |
|---|---|
| **Image** | `ghcr.io/aeon-7/aeon-vllm-ultimate` (pinned by digest) |
| **Container** | `gemma4-vllm` |
| **Network / IPC** | `host` / `host` |
| **GPUs** | all (`--gpus all`) |
| **Restart** | `always` |
| **Volumes** | HF cache, vLLM compile cache, chat template |

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| `docker is not on PATH` | install Docker / NVIDIA Container Toolkit |
| Exited before ready | check `.vllm.log`; verify GB10 drivers + sm_121 image |
| NaN / garbage output | confirm `VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass` + `VLLM_USE_FLASHINFER_MOE_FP4=0` |
| Drafter crash on first request | drafter `attention_backend` must be `flash_attn`, not `flex_attention` |
| OOM | lower `--gpu-memory-utilization` or `--max-num-seqs` |
| First request slow (~50 s) | one-time compile; `start.sh` already warms it — add a keep-warm cron for reboots |

### Rejected speculative / batch configurations (2026-06-24 sweep)

The following configurations were tested against the v1 settings and rejected:

- `num_speculative_tokens = 3` — no p50 win, but kills the high-acceptance ceiling by 12.5% (max 113.2 → 99.1 tok/s). The drafter can't produce more than 3 accepted tokens per forward pass, which trims the upside on prompt classes that DFlash already accepts well.
- `--max-num-batched-tokens 8192` — chunked-prefill clamp at 7936 (= 8192 − speculative budget), regresses p50 by ~3.4%. Use 16384 or higher to disengage the clamp.
- `--async-scheduling` as an explicit override — no-op on vLLM 0.23+ (already default). Documented for clarity, no perf effect.

---

## 📝 License

- **Model weights:** [Gemma 4 license — Apache 2.0](https://ai.google.dev/gemma/apache_2). The
  uncensored weights are a community derivative; review their model card. Performance is identical
  to the official `google/gemma-4-26B-A4B-it` weights — swap the `MODEL_ID` if you prefer those.
- **This codebase:** MIT.

## 📚 Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [Gemma 4 on HuggingFace](https://huggingface.co/google/gemma-4-26B-A4B-it)
- [DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
