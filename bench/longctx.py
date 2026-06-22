#!/usr/bin/env python3
"""Long-context test for Gemma 4 at 256k: ~180k-token prompt with a needle.
Proves the model ingests deep context; measures TTFT (prefill at depth) + retrieval."""
import json, time, urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4"

FILLER = "The quick brown fox jumps over the lazy dog. "
NEEDLE = " IMPORTANT FACT: the secret pass-phrase is crimson-otter-42. Remember it. "
N = 18000
parts = []
for i in range(N):
    if i == N // 2:
        parts.append(NEEDLE)
    parts.append(FILLER)
context = "".join(parts)
prompt = context + "\n\nQuestion: what is the secret pass-phrase stated above? Answer with only the phrase."

body = json.dumps({
    "model": MODEL,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 32, "temperature": 0, "stream": True,
}).encode()

print(f"prompt_chars={len(prompt)}  approx_tokens={len(prompt)//4}")
req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
t0 = time.time(); ttft = None; out = ""
resp = urllib.request.urlopen(req, timeout=900)
for raw in resp:
    line = raw.decode("utf-8", "ignore").strip()
    if not line.startswith("data:"):
        continue
    d = line[5:].strip()
    if d == "[DONE]":
        break
    try:
        j = json.loads(d)
        delta = j["choices"][0]["delta"].get("content", "")
        if delta and ttft is None:
            ttft = time.time() - t0
        out += delta
    except Exception:
        pass
total = time.time() - t0
print(f"TTFT(prefill ~180k)={ttft:.1f}s  total={total:.1f}s")
print(f"answer={out.strip()!r}")
print(f"needle_found={'crimson-otter-42' in out}")
print("DONE")
