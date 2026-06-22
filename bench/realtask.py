#!/usr/bin/env python3
"""Real-world task latency: extraction-style (moderate input, SHORT structured output)
and a vision request. Mirrors 'parse an Excel / image and return a few values'."""
import json, time, urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4"

def call(messages, max_tokens, label):
    body = json.dumps({"model": MODEL, "messages": messages,
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=120))
    dt = time.time() - t0
    out = r["choices"][0]["message"]["content"]
    ct = r["usage"]["completion_tokens"]
    pt = r["usage"]["prompt_tokens"]
    print(f"{label}: {dt:.2f}s  (in={pt} tok, out={ct} tok)  -> {out[:70]!r}")

# 1) "big-ish Excel" — ~150-row CSV table, ask for a short JSON extraction
rows = "\n".join(f"{i},Widget-{i},{i*7%200},{(i*13)%2}" for i in range(1, 400))
table = "id,name,qty,flag\n" + rows
call([{"role": "user", "content": f"Here is a CSV:\n{table}\n\nReturn ONLY JSON: "
       '{"total_rows": N, "sum_qty": N}. No prose.'}], 60, "excel-extract (short out)")

# 2) smaller table, single value
call([{"role": "user", "content": f"CSV:\n{table}\n\nHow many rows have flag=1? Answer with just the number."}],
     12, "excel-single-value")

# 3) vision — image in, short answer out
call([{"role": "user", "content": [
        {"type": "text", "text": "What animal is in this image? One word."},
        {"type": "image_url", "image_url": {"url": "http://images.cocodataset.org/val2017/000000039769.jpg"}}]}],
     10, "vision-extract (short out)")

print("DONE")
