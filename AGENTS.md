# AGENTS.md — install & run this on a DGX Spark

Guidance for an AI coding agent setting up **Gemma 4 26B-A4B (NVFP4 + DFlash)** on an
**NVIDIA DGX Spark (GB10, sm_121)**. Work top to bottom; every command is copy-paste runnable.
This is a single-model deployment wrapper — there is no build step.

## 1. Preconditions (check, then proceed)
- `nvidia-smi` shows a **GB10 / DGX Spark**. The container image is **sm_121 only** — if the GPU
  is anything else, stop and tell the user (it won't run without a rebuild).
- `docker --version` ≥ 24, with the NVIDIA Container Toolkit. `curl` on PATH.
- ~30 GB free disk for weights + caches.
- Optional, recommended: `export HF_TOKEN=hf_...` (faster HuggingFace downloads).

## 2. Install + run — one command
```bash
./start.sh
```
What it does: downloads the model + the DFlash drafter from HuggingFace, launches the vLLM
container (`--restart always`), polls `/v1/models` until ready, fires a warm-up request, and
prints the OpenAI base URL (`http://0.0.0.0:8000/v1`). **First boot** also pays a one-time ~50 s
compile — `start.sh` absorbs it with the warm-up so the first real request is fast.

## 3. Verify it works
```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4",
       "messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
```
Expect HTTP 200 with a coherent reply. Vision (image input) and tool-calling are enabled.

## 4. Stop
```bash
./stop.sh
```

## 5. Reproduce the benchmarks (optional)
```bash
python3 bench/scaling.py     # concurrency scaling c=1/2/4/8  -> bench/results.json
python3 bench/longctx.py     # long-context needle test
python3 bench/realtask.py    # real-world extraction + vision latency
```

## 6. Do NOT "fix" these — they are load-bearing for GB10
- `--attention-backend triton_attn` — Gemma 4's heterogeneous head dims require it.
- Drafter must use `flash_attn` in `--speculative-config`; `flex_attention` crashes this image.
- `VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass` + `VLLM_USE_FLASHINFER_MOE_FP4=0` — GB10 has no
  native FP4 tensor cores; this combo produces valid output (no NaN).
- The image is **pinned by digest** in `start.sh`. Do not switch it to `:latest`.

## 7. Tuning knobs (only when asked)
- `PORT=8000` env to change the port.
- `--gpu-memory-utilization` (default `0.80`) — raise for more KV cache headroom.
- `--max-model-len 262144` is full 256K; lower it to trade context window for more concurrency.

## 8. Auto-start on reboot (production)
The container runs with `--restart always`, so Docker restarts it on boot. For stronger
self-healing, install a systemd unit whose `ExecStart` runs `./start.sh` (re-applies the stack on
boot) and `enable` it.

## 9. What to expect (perf)
- Real extraction/parsing (moderate input, short output): **~1 s**.
- Throughput: ~68 tok/s single stream, ~403 tok/s aggregate at 8 concurrent.
- Full 256K context works. A cold 200k+ one-shot prefill takes minutes (bandwidth-bound); with
  prefix caching, incrementally-grown context is cheap and re-reads are near-instant.
