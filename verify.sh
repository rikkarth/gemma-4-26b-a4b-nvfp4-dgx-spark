#!/usr/bin/env bash
# verify.sh — post-install health check for the Gemma 4 vLLM deployment.
#
# Re-runnable. Each check prints OK / FAIL with a one-line fix-it. Exits 0
# only when every check passes. Use it after ./setup.sh, or any time you
# suspect something is off (reboots, image updates, drafter crashes).
#
# Env overrides:
#   RUNTIME_DIR    where the compose stack + .env live   (default: /home/system/gemma4-runtime)
#   VLLM_URL       base URL of the running vLLM          (default: http://127.0.0.1:8000)
#   VLLM_MODEL     model id expected in /v1/models        (default: AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4)
#   CONTAINER_NAME docker container name                  (default: gemma4-vllm)
set -euo pipefail

RUNTIME_DIR="${RUNTIME_DIR:-/home/system/gemma4-runtime}"
VLLM_URL="${VLLM_URL:-http://127.0.0.1:8000}"
VLLM_MODEL="${VLLM_MODEL:-AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4}"
CONTAINER_NAME="${CONTAINER_NAME:-gemma4-vllm}"
HF_DIR_SLUG="models--${VLLM_MODEL//\//--}"

ok=0
fail=0

# ── check NAME CONDITION FIX ─────────────────────────────────────────────
# CONDITION is a shell command; "ok" means exit-0, anything else is a fail.
check() {
  local name="$1" cond="$2" fix="$3"
  if eval "${cond}" >/dev/null 2>&1; then
    printf '  [OK]   %s\n' "${name}"
    ok=$((ok + 1))
  else
    printf '  [FAIL] %s\n' "${name}"
    printf '         fix: %s\n' "${fix}"
    fail=$((fail + 1))
  fi
}

echo "Verifying Gemma 4 vLLM deployment"
echo "  runtime:   ${RUNTIME_DIR}"
echo "  endpoint:  ${VLLM_URL}"
echo "  container: ${CONTAINER_NAME}"
echo "  model:     ${VLLM_MODEL}"
echo

# 1. Docker daemon
check "Docker daemon reachable" \
  "docker info" \
  "start Docker (sudo systemctl start docker) and re-run"

# 2. vLLM container is running
check "container ${CONTAINER_NAME} is running" \
  "[[ \$(docker inspect -f '{{.State.Status}}' '${CONTAINER_NAME}' 2>/dev/null) == running ]]" \
  "cd ${RUNTIME_DIR} && docker compose up -d   (or: ./start.sh)"

# 3. /v1/models lists the model
check "/v1/models lists ${VLLM_MODEL}" \
  "curl -fsS -m 10 '${VLLM_URL}/v1/models' | grep -q '\"id\":\"${VLLM_MODEL}\"'" \
  "wait for vLLM to finish loading (tail ${RUNTIME_DIR}/.vllm.log)"

# 4. /v1/chat/completions roundtrip
check "chat/completions roundtrip (1 token)" \
  "curl -fsS -m 60 '${VLLM_URL}/v1/chat/completions' \
     -H 'Content-Type: application/json' \
     -d '{\"model\":\"${VLLM_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}' \
     | grep -q '\"choices\"'" \
  "docker logs ${CONTAINER_NAME} --tail 200  (look for OOM, NaN, drafter errors)"

# 5. keepwarm cron installed
check "keepwarm cron entry installed" \
  "crontab -l 2>/dev/null | grep -q 'keepwarm-gemma'" \
  "see scripts/keepwarm-gemma.sh — re-run setup.sh to install the cron"

# 6. model weights present
check "model weights present under ${RUNTIME_DIR}/.cache/huggingface" \
  "find '${RUNTIME_DIR}/.cache/huggingface' -path '*${HF_DIR_SLUG}*' -name '*.safetensors' -print -quit | grep -q ." \
  "run setup.sh without --no-pull (or: huggingface-cli download ${VLLM_MODEL} --local-dir ${RUNTIME_DIR}/.cache/huggingface/hub)"

# Summary
echo
total=$((ok + fail))
if [[ ${fail} -eq 0 ]]; then
  printf 'PASS  %d/%d checks  --  deployment is healthy\n' "${ok}" "${total}"
  printf '  OpenAI base URL: %s/v1\n' "${VLLM_URL}"
  exit 0
else
  printf 'FAIL  %d/%d checks failed  (%d passed)\n' "${fail}" "${total}" "${ok}"
  exit 1
fi
