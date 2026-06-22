#!/usr/bin/env bash
set -euo pipefail

# ── Gemma 4 26B-A4B (NVFP4 + DFlash) on DGX Spark (GB10, sm_121) via vLLM ──
MODEL_ID="AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4"
DRAFTER_ID="z-lab/gemma-4-26B-A4B-it-DFlash"
IMAGE="ghcr.io/aeon-7/aeon-vllm-ultimate@sha256:be9e05a11da6e72607ab6f3e960993b253b673af0727005122a3266129a518e3"
CONTAINER_NAME="gemma4-vllm"
HOST="0.0.0.0"
PORT="${PORT:-8000}"
PID_FILE=".vllm.pid"
LOG_FILE=".vllm.log"
WORK_DIR="$(pwd)"
HF_HOME="${WORK_DIR}/.cache/huggingface"
VLLM_CACHE="${WORK_DIR}/.cache/vllm"           # persists the compile cache → faster restarts
READY_URL="http://127.0.0.1:${PORT}/v1/models"
CHAT_URL="http://127.0.0.1:${PORT}/v1/chat/completions"

command -v docker >/dev/null 2>&1 || { echo "docker is not on PATH"; exit 1; }
command -v curl   >/dev/null 2>&1 || { echo "curl is not on PATH"; exit 1; }

mkdir -p "${HF_HOME}" "${VLLM_CACHE}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is already running"; echo "Log: ${LOG_FILE}"; exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

echo "Starting vLLM container for ${MODEL_ID}"
echo "Image: ${IMAGE}"
echo "Listening on ${HOST}:${PORT}"
echo "Writing progress to ${LOG_FILE}"
echo "[$(date -Is)] launching vLLM container" > "${LOG_FILE}"

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host --ipc host --gpus all --restart always \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e TORCH_MATMUL_PRECISION=high \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e VLLM_TEST_FORCE_FP8_MARLIN=0 \
  -e VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "${VLLM_CACHE}:/root/.cache/vllm" \
  -v "${WORK_DIR}/chat_template.jinja:/workspace/chat_template.jinja" \
  --entrypoint vllm \
  "${IMAGE}" \
  serve "${MODEL_ID}" \
    --host "${HOST}" --port "${PORT}" \
    --tensor-parallel-size 1 --dtype auto \
    --quantization compressed-tensors \
    --attention-backend triton_attn \
    --max-model-len 262144 \
    --max-num-seqs 64 --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.80 \
    --enable-chunked-prefill --enable-prefix-caching --trust-remote-code \
    --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
    --speculative-config "{\"method\":\"dflash\",\"model\":\"${DRAFTER_ID}\",\"num_speculative_tokens\":10,\"attention_backend\":\"flash_attn\"}" \
  >/dev/null

container_id="$(docker inspect -f '{{.Id}}' "${CONTAINER_NAME}")"
echo "${container_id}" > "${PID_FILE}"
echo "Spawned container ${CONTAINER_NAME} (${container_id})"

log_follow_pid=""
trap '[[ -n "${log_follow_pid}" ]] && kill "${log_follow_pid}" 2>/dev/null || true' EXIT
(docker logs -f "${CONTAINER_NAME}" >> "${LOG_FILE}" 2>&1) & log_follow_pid=$!

echo "Waiting for HTTP readiness at ${READY_URL}"
echo "  (first boot downloads ~17 GB of weights, then compiles — be patient)"
until curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "vLLM container exited before becoming ready"; tail -n 200 "${LOG_FILE}" || true; exit 1
  fi
  echo "  still starting..."; sleep 5
done

# Fire one synthetic request to absorb the one-time ~50s compile/CUDA-graph/drafter
# cliff, so the first real user never waits (vLLM's recommended production pattern).
echo "Warming up..."
curl -sS -o /dev/null -m 120 "${CHAT_URL}" -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"warmup\"}],\"max_tokens\":1}" || true

echo "vLLM is ready and responding; shell is now free."
echo "OpenAI base URL: http://${HOST}:${PORT}/v1"
