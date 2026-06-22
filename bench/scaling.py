#!/usr/bin/env python3
"""Concurrency-scaling benchmark for Gemma 4 26B-A4B NVFP4 + DFlash on DGX Spark.
Runs on the DGX against localhost:8000 (raw model, no network/proxy).
Fixed-length deterministic generations (ignore_eos) for clean apples-to-apples scaling.
Outputs Mia-style table: cumulative gen tok/s, per-session avg, scaling factor, efficiency."""
import json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4"
PROMPT = ("Write a detailed technical explanation of how paged attention and "
          "continuous batching work in a modern LLM inference server, with examples "
          "and trade-offs.")
MAXTOK = 512

def one_request():
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAXTOK, "temperature": 0, "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=600))
    dt = time.time() - t0
    return r["usage"]["completion_tokens"], dt

def run_concurrency(c):
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=c) as ex:
        res = list(ex.map(lambda _: one_request(), range(c)))
    wall = time.time() - t0
    total = sum(ct for ct, _ in res)
    per_session = [ct / dt for ct, dt in res]
    return {
        "concurrency": c,
        "wall_s": round(wall, 2),
        "total_tokens": total,
        "cumulative_tok_s": round(total / wall, 2),
        "per_session_avg_tok_s": round(sum(per_session) / len(per_session), 2),
    }

print("warmup..."); one_request()
results, base = [], None
for c in [1, 2, 4, 8]:
    r = run_concurrency(c)
    if base is None:
        base = r["cumulative_tok_s"]
    r["scaling_factor"] = round(r["cumulative_tok_s"] / base, 2)
    r["efficiency_pct"] = round(100 * r["scaling_factor"] / c, 1)
    results.append(r)
    print(json.dumps(r))
json.dump(results, open("/tmp/gemma4-scaling.json", "w"), indent=2)
print("DONE")
