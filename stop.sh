#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="gemma4-vllm"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  docker rm -f "${CONTAINER_NAME}" >/dev/null && echo "Stopped and removed ${CONTAINER_NAME}"
else
  echo "${CONTAINER_NAME} is not running"
fi

rm -f .vllm.pid
