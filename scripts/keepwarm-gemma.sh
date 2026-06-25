#!/usr/bin/env bash
# keepwarm-gemma.sh — periodic 1-token ping against the vLLM server.
#
# Purpose: vLLM pays a one-time ~50 s compile / CUDA-graph / drafter warmup on
# the very first request. After that it stays hot as long as the GPU is in
# use. Long idle periods (overnight, weekend, rebooted box) drop the warm
# caches and the next real user pays the cliff again.
#
# This script fires one cheap, 1-token chat-completions request so the
# drafter and CUDA graphs stay compiled. Run it from cron (default: every
# minute — see KEEPWARM_SCHEDULE).
#
# Env overrides:
#   KEEPWARM_URL     endpoint to ping          (default: http://127.0.0.1:8000/v1/chat/completions)
#   KEEPWARM_MODEL   model id                  (default: AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4)
#   KEEPWARM_LOG     log file path             (default: /var/log/keepwarm-gemma.log)
#   KEEPWARM_TIMEOUT per-request timeout, sec  (default: 30)
set -euo pipefail

URL="${KEEPWARM_URL:-http://127.0.0.1:8000/v1/chat/completions}"
MODEL="${KEEPWARM_MODEL:-AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4}"
LOG_FILE="${KEEPWARM_LOG:-/var/log/keepwarm-gemma.log}"
TIMEOUT="${KEEPWARM_TIMEOUT:-30}"

ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
t0=$(date +%s)

# Capture HTTP code; default to 000 if curl itself fails.
code=$(curl -sS -o /dev/null -w '%{http_code}' -m "${TIMEOUT}" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1}" \
  "${URL}" 2>/dev/null || echo "000")

t1=$(date +%s)
dt=$((t1 - t0))

# Always log one line — success or failure. Never exit non-zero: a flaky
# network blip should not spam cron with mail.
printf '%s http=%s t=%ss\n' "${ts}" "${code}" "${dt}" >> "${LOG_FILE}"
exit 0
