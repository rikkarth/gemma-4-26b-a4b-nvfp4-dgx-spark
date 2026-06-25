#!/usr/bin/env bash
# setup.sh — single-command fresh-install for Gemma 4 26B-A4B (NVFP4 + DFlash)
# on a NVIDIA DGX Spark (GB10, sm_121).
#
# What it does (11 steps):
#   0.  Pre-flight (Linux + CUDA / GB10).
#   1.  Install Docker if missing.
#   2.  Install the NVIDIA Container Toolkit if missing.
#   3.  Create the runtime dir (default /home/system/gemma4-runtime).
#   4.  Write .env with HF_TOKEN + host paths.
#   5.  Pull the model + drafter weights (idempotent; honors --no-pull).
#   6.  Stage docker-compose.gemma4.yml + chat_template.jinja into the runtime dir.
#   7.  Install the keepwarm script + cron.
#   8.  Start the vLLM container via `docker compose up -d`.
#   9.  Wait for the OpenAI endpoint to become healthy (up to 15 min).
#   10. Smoke test (/v1/models lists the model; 1-token chat roundtrip works).
#   11. Print a green/red summary.
#
# Idempotent. Re-runnable. Safe to interrupt and restart.
#
# Usage:
#   ./setup.sh [--hf-token TOKEN] [--runtime-dir /path] [--no-pull] [--dry-run] [-h]
#
#   --hf-token TOKEN   HuggingFace token (else $HF_TOKEN; else prompts silently)
#   --runtime-dir DIR  compose stack + .env + .cache live here
#                      (default: /home/system/gemma4-runtime)
#   --no-pull          skip model weight download (assume weights are present)
#   --dry-run          print what would happen; do not execute system changes
#   -h, --help         show this help and exit
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────
RUNTIME_DIR="/home/system/gemma4-runtime"
HF_TOKEN_ARG=""
NO_PULL=0
DRY_RUN=0

# SUDO: empty if already root, "sudo" otherwise. Used for every system call
# until the operator re-logs in and the docker-group usermod takes effect.
SUDO=""
if [[ "$(id -u)" -ne 0 ]]; then SUDO="sudo"; fi

# ── Tiny logger ──────────────────────────────────────────────────────────
log()  { printf '%s\n' "$*"; }
warn() { printf 'WARN  %s\n' "$*" >&2; }
err()  { printf 'ERR   %s\n' "$*" >&2; }

# run_or_print CMD...  — print in dry-run, else execute (set -e applies).
run_or_print() {
  if [[ ${DRY_RUN} -eq 1 ]]; then
    printf '  $ %s\n' "$*"
  else
    "$@"
  fi
}

# ── Usage ────────────────────────────────────────────────────────────────
usage() {
  cat <<'USAGE'
setup.sh — fresh-install for Gemma 4 26B-A4B (NVFP4 + DFlash) on a DGX Spark (GB10, sm_121).

Usage:
  ./setup.sh [--hf-token TOKEN] [--runtime-dir /path] [--no-pull] [--dry-run] [-h]

  --hf-token TOKEN   HuggingFace token (else $HF_TOKEN; else prompts silently)
  --runtime-dir DIR  compose stack + .env + .cache live here
                     (default: /home/system/gemma4-runtime)
  --no-pull          skip model weight download (assume weights are present)
  --dry-run          print what would happen; do not execute system changes
  -h, --help         show this help and exit

Idempotent. Re-runnable. Safe to interrupt and restart.
USAGE
  exit 0
}

# ── Arg parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-token)        HF_TOKEN_ARG="$2"; shift 2;;
    --hf-token=*)      HF_TOKEN_ARG="${1#*=}"; shift;;
    --runtime-dir)     RUNTIME_DIR="$2"; shift 2;;
    --runtime-dir=*)   RUNTIME_DIR="${1#*=}"; shift;;
    --no-pull)         NO_PULL=1; shift;;
    --dry-run)         DRY_RUN=1; shift;;
    -h|--help)         usage;;
    *) err "unknown arg: $1"; usage; exit 2;;
  esac
done

# Resolve HF_TOKEN: --hf-token > $HF_TOKEN > silent prompt.
if [[ -z "${HF_TOKEN_ARG}" ]]; then
  if [[ -n "${HF_TOKEN:-}" ]]; then
    HF_TOKEN_ARG="${HF_TOKEN}"
  elif [[ -t 0 ]]; then
    log "HuggingFace token not set. Get one at https://huggingface.co/settings/tokens"
    log "(Both the model and drafter are gated — accept the licenses on their"
    log " model pages first, then paste the token here.)"
    read -r -s -p "HF_TOKEN: " HF_TOKEN_ARG; printf '\n'
  else
    err "no HF_TOKEN supplied and no TTY to prompt. Pass --hf-token or set \$HF_TOKEN."
    exit 2
  fi
fi

# ── Banner ───────────────────────────────────────────────────────────────
log "==================================================================="
log "Gemma 4 26B-A4B (NVFP4 + DFlash) on DGX Spark — fresh install"
log "==================================================================="
log "  runtime dir: ${RUNTIME_DIR}"
log "  HF token:    ${HF_TOKEN_ARG:0:4}*** (${#HF_TOKEN_ARG} chars)"
log "  model pull:  $([[ ${NO_PULL} -eq 1 ]] && echo 'skip (--no-pull)' || echo 'yes')"
log "  mode:        $([[ ${DRY_RUN} -eq 1 ]] && echo 'dry-run' || echo 'live')"
log "==================================================================="

step_start() { printf '\n[step %s] %s\n' "$1" "$2"; }

# ── Step 0: pre-flight ───────────────────────────────────────────────────
step_0() {
  step_start 0/11 "Pre-flight (Linux + CUDA)"
  if [[ "$(uname -s)" != "Linux" ]]; then
    warn "this script targets Linux (DGX Spark). Detected: $(uname -s)"
  else
    log "  Linux: ok ($(uname -r))"
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local gpu
    gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo unknown)"
    log "  GPU:   ${gpu}"
    if ! echo "${gpu}" | grep -qiE 'gb10|dgx spark'; then
      warn "GPU is not a GB10 / DGX Spark. The aeon-vllm-ultimate image is sm_121 only."
      warn "It will not run on other GPUs without a rebuild."
    fi
  else
    warn "nvidia-smi not on PATH. Install NVIDIA drivers (>= 580) before proceeding."
    warn "The script will continue, but the vLLM container will fail to start without drivers."
  fi
  log "OK  pre-flight"
}

# ── Step 1: install Docker ───────────────────────────────────────────────
step_1() {
  step_start 1/11 "Install Docker"
  if command -v docker >/dev/null 2>&1; then
    log "  Docker already installed: $(docker --version)"
    log "OK  Docker"
    return
  fi
  log "  installing via apt (docker.io) ..."
  run_or_print ${SUDO} apt-get update -y
  run_or_print ${SUDO} apt-get install -y --no-install-recommends \
    ca-certificates curl python3-pip
  run_or_print ${SUDO} apt-get install -y --no-install-recommends docker.io
  run_or_print ${SUDO} systemctl enable --now docker
  # Add the operator to the docker group so future `docker` calls don't need sudo.
  local op_user="${USER:-$(id -un)}"
  if [[ -n "${SUDO}" ]] && id -u "${op_user}" >/dev/null 2>&1; then
    run_or_print ${SUDO} usermod -aG docker "${op_user}" || true
    log "  added ${op_user} to the docker group (log out + back in to take effect)."
    log "  until then, this script uses '${SUDO} docker' for every docker call."
  fi
  if ! command -v docker >/dev/null 2>&1; then
    err "Docker install failed. Check apt output above."
    err "fix: install Docker manually — https://docs.docker.com/engine/install/ubuntu/"
    return 1
  fi
  log "OK  Docker ($(docker --version))"
}

# ── Step 2: install NVIDIA Container Toolkit ────────────────────────────
step_2() {
  step_start 2/11 "Install NVIDIA Container Toolkit"
  if ${SUDO} docker info 2>/dev/null | grep -qi 'Default Runtime: nvidia'; then
    log "  nvidia runtime already configured."
    log "OK  NVIDIA Container Toolkit"
    return
  fi
  log "  adding nvidia-container-toolkit repo + installing ..."
  run_or_print ${SUDO} install -m 0755 -d /usr/share/keyrings
  run_or_print bash -c "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | ${SUDO} tee /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg >/dev/null"
  run_or_print bash -c "curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | ${SUDO} tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null"
  run_or_print ${SUDO} apt-get update -y
  run_or_print ${SUDO} apt-get install -y --no-install-recommends nvidia-container-toolkit
  run_or_print ${SUDO} nvidia-ctk runtime configure --runtime=docker
  run_or_print ${SUDO} systemctl restart docker
  if ! ${SUDO} docker info 2>/dev/null | grep -qi 'Default Runtime: nvidia'; then
    err "nvidia runtime not active after install."
    err "fix: ${SUDO} nvidia-ctk runtime configure --runtime=docker && ${SUDO} systemctl restart docker"
    return 1
  fi
  log "OK  NVIDIA Container Toolkit"
}

# ── Step 3: create runtime dir ──────────────────────────────────────────
step_3() {
  step_start 3/11 "Create runtime dir"
  run_or_print ${SUDO} mkdir -p "${RUNTIME_DIR}"
  run_or_print ${SUDO} chown "$(id -un):$(id -gn)" "${RUNTIME_DIR}" 2>/dev/null || true
  run_or_print mkdir -p "${RUNTIME_DIR}/.cache/huggingface" "${RUNTIME_DIR}/.cache/vllm"
  log "  runtime dir: ${RUNTIME_DIR}"
  log "OK  runtime dir"
}

# ── Step 4: write .env ──────────────────────────────────────────────────
step_4() {
  step_start 4/11 "Write .env"
  local env_file="${RUNTIME_DIR}/.env"
  if [[ -f "${env_file}" ]] && grep -q '^HF_TOKEN=.' "${env_file}"; then
    log "  .env already has HF_TOKEN — keeping existing (edit ${env_file} to change)."
  else
    if [[ ${DRY_RUN} -eq 1 ]]; then
      log "  would write ${env_file} (HF_TOKEN=${HF_TOKEN_ARG:0:4}***)"
    else
      cat > "${env_file}" <<EOF
# Generated by setup.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ)
HF_TOKEN=${HF_TOKEN_ARG}
HF_HOME=${RUNTIME_DIR}/.cache/huggingface
VLLM_CACHE=${RUNTIME_DIR}/.cache/vllm
CHAT_TEMPLATE=${RUNTIME_DIR}/chat_template.jinja
PORT=8000
CONTAINER_NAME=gemma4-vllm
EOF
      chmod 600 "${env_file}"
      log "  wrote ${env_file} (mode 600)"
    fi
  fi
  log "OK  .env"
}

# ── Step 5: pull model + drafter ────────────────────────────────────────
step_5() {
  step_start 5/11 "Pull model + drafter weights"
  if [[ ${NO_PULL} -eq 1 ]]; then
    log "  --no-pull set; skipping download."
    log "OK  model weights (skipped)"
    return
  fi

  # Install huggingface_hub on first run.
  if ! command -v huggingface-cli >/dev/null 2>&1; then
    log "  installing huggingface_hub (pip --break-system-packages) ..."
    if ! command -v pip3 >/dev/null 2>&1; then
      run_or_print ${SUDO} apt-get install -y --no-install-recommends python3-pip
    fi
    run_or_print ${SUDO} pip3 install --break-system-packages --quiet --upgrade huggingface_hub
  fi

  local models=(
    "AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4"
    "z-lab/gemma-4-26B-A4B-it-DFlash"
  )
  for m in "${models[@]}"; do
    local slug="models--${m//\//--}"
    local dest="${RUNTIME_DIR}/.cache/huggingface/hub/${slug}"
    # Heuristic: any non-empty snapshot dir => skip.
    if [[ -d "${dest}/snapshots" ]] \
       && find "${dest}/snapshots" -mindepth 1 -maxdepth 2 -print -quit 2>/dev/null | grep -q .; then
      log "  ${m}: present, skipping"
      continue
    fi
    log "  downloading ${m} -> ${dest}"
    if [[ ${DRY_RUN} -eq 1 ]]; then
      log "  $ HF_TOKEN=*** huggingface-cli download ${m} --local-dir ${dest} --token ***"
    else
      if ! HF_TOKEN="${HF_TOKEN_ARG}" huggingface-cli download "${m}" \
            --local-dir "${dest}" --token "${HF_TOKEN_ARG}" 2>&1 | tail -n 50; then
        err "download failed for ${m}."
        err "fix: confirm your HF token has access to ${m} (accept the model license on"
        err "     the model page), then re-run ./setup.sh."
        return 1
      fi
    fi
  done
  log "OK  model weights"
}

# ── Step 6: stage compose + chat template ───────────────────────────────
step_6() {
  step_start 6/11 "Stage compose + chat_template.jinja into runtime dir"
  local repo_root
  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  local compose_src="${repo_root}/docker-compose.gemma4.yml"
  local template_src="${repo_root}/chat_template.jinja"
  if [[ ! -f "${compose_src}" ]]; then
    err "compose source not found: ${compose_src}"
    err "fix: re-clone the repo (or copy docker-compose.gemma4.yml next to setup.sh)."
    return 1
  fi
  if [[ ! -f "${template_src}" ]]; then
    err "chat template not found: ${template_src}"
    return 1
  fi
  run_or_print cp -f "${compose_src}" "${RUNTIME_DIR}/docker-compose.gemma4.yml"
  run_or_print cp -f "${template_src}" "${RUNTIME_DIR}/chat_template.jinja"
  log "  compose:   ${RUNTIME_DIR}/docker-compose.gemma4.yml"
  log "  template:  ${RUNTIME_DIR}/chat_template.jinja"
  log "OK  compose + chat template"
}

# ── Step 7: install keepwarm + cron ─────────────────────────────────────
step_7() {
  step_start 7/11 "Install keepwarm script + cron"
  local repo_root keepwarm_src keepwarm_dst schedule log_file
  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  keepwarm_src="${repo_root}/scripts/keepwarm-gemma.sh"
  keepwarm_dst="/usr/local/bin/keepwarm-gemma"
  schedule="${KEEPWARM_SCHEDULE:-* * * * *}"
  log_file="${KEEPWARM_LOG:-/var/log/keepwarm-gemma.log}"

  if [[ ! -f "${keepwarm_src}" ]]; then
    err "keepwarm script not found: ${keepwarm_src}"
    return 1
  fi
  # Install cron daemon if missing (rare on Ubuntu server, common on minimal/cloud images).
  if ! command -v crontab >/dev/null 2>&1; then
    log "  installing cron (needed for keepwarm schedule) ..."
    run_or_print ${SUDO} apt-get install -y --no-install-recommends cron
    run_or_print ${SUDO} systemctl enable --now cron
  fi

  run_or_print ${SUDO} install -m 0755 "${keepwarm_src}" "${keepwarm_dst}"

  # Ensure the log file is writable. /var/log may be unwritable without sudo on
  # some cloud images; fall back to ~/.local/share/keepwarm-gemma.log.
  if [[ ! -e "${log_file}" ]]; then
    if ${SUDO} install -m 0644 /dev/null "${log_file}" 2>/dev/null; then
      log "  log: ${log_file}"
    else
      log_file="${HOME:-/root}/.local/share/keepwarm-gemma.log"
      run_or_print mkdir -p "$(dirname "${log_file}")"
      : > "${log_file}"
      log "  /var/log not writable; using ${log_file}"
    fi
  fi

  # Install (or replace) the cron entry, keyed on the absolute path so it stays single-instance.
  if [[ ${DRY_RUN} -eq 0 ]]; then
    ( crontab -l 2>/dev/null | grep -v -F "${keepwarm_dst}" || true
      printf '%s %s >> %s 2>&1\n' "${schedule}" "${keepwarm_dst}" "${log_file}"
    ) | crontab -
  else
    log "  $ would add cron line: ${schedule} ${keepwarm_dst} >> ${log_file} 2>&1"
  fi
  log "  schedule: ${schedule}"
  log "OK  keepwarm + cron"
}

# ── Step 8: start container ─────────────────────────────────────────────
step_8() {
  step_start 8/11 "Start vLLM container"
  if [[ ${DRY_RUN} -eq 1 ]]; then
    log "  (dry-run) would run: cd ${RUNTIME_DIR} && ${SUDO} docker compose up -d"
    log "OK  vLLM container (dry-run)"
    return
  fi
  if ${SUDO} docker ps --format '{{.Names}}' 2>/dev/null | grep -qx gemma4-vllm; then
    log "  container gemma4-vllm already running; skipping start."
    log "OK  vLLM container (already running)"
    return
  fi
  log "  ${SUDO} docker compose up -d ..."
  ( cd "${RUNTIME_DIR}" && ${SUDO} docker compose up -d ) 2>&1 | tail -n 30
  log "OK  vLLM container"
}

# ── Step 9: wait for healthy ────────────────────────────────────────────
step_9() {
  step_start 9/11 "Wait for /v1/models (up to 15 min)"
  local ready_url="http://127.0.0.1:8000/v1/models"
  log "  ${ready_url}"
  if [[ ${DRY_RUN} -eq 1 ]]; then
    log "  (dry-run) would poll up to 15 min"
    log "OK  /v1/models (dry-run)"
    return
  fi
  local deadline=$((SECONDS + 900)) last_print=0
  while (( SECONDS < deadline )); do
    if curl -fsS -m 5 "${ready_url}" >/dev/null 2>&1; then
      log "  ready after ${SECONDS}s"
      log "OK  /v1/models"
      return
    fi
    if ! ${SUDO} docker ps --format '{{.Names}}' 2>/dev/null | grep -qx gemma4-vllm; then
      err "container exited before becoming healthy."
      err "fix: ${SUDO} docker logs gemma4-vllm --tail 200"
      return 1
    fi
    if (( SECONDS - last_print >= 30 )); then
      log "  still starting... (${SECONDS}s elapsed)"
      last_print=${SECONDS}
    fi
    sleep 5
  done
  err "timed out after 15 min."
  err "fix: ${SUDO} docker logs gemma4-vllm --tail 200  (compile / drafter / OOM)"
  return 1
}

# ── Step 10: smoke test ────────────────────────────────────────────────
step_10() {
  step_start 10/11 "Smoke test (model listed + 1-token chat roundtrip)"
  local model_id="AEON-7/Gemma-4-26B-A4B-it-Uncensored-NVFP4"
  local models_json
  models_json="$(curl -fsS -m 10 http://127.0.0.1:8000/v1/models 2>/dev/null || true)"
  if [[ -z "${models_json}" ]]; then
    err "/v1/models is empty (vLLM not responding?)"
    return 1
  fi
  if ! grep -q "\"id\":\"${model_id}\"" <<<"${models_json}"; then
    err "/v1/models does not list ${model_id}"
    err "fix: ${SUDO} docker logs gemma4-vllm --tail 200"
    return 1
  fi
  log "  /v1/models lists ${model_id}"

  local body
  body="$(curl -fsS -m 60 http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${model_id}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" 2>/dev/null || true)"
  if ! grep -q '"choices"' <<<"${body}"; then
    err "1-token chat roundtrip failed"
    err "fix: ${SUDO} docker logs gemma4-vllm --tail 200"
    return 1
  fi
  log "  chat/completions roundtrip: ok"
  log "OK  smoke test"
}

# ── Step 11: summary ────────────────────────────────────────────────────
step_11() {
  local ok_count="$1" total_count="$2"
  printf '\n===================================================================\n'
  if [[ ${ok_count} -eq ${total_count} ]]; then
    printf '  PASS  %d/%d steps — Gemma 4 26B-A4B is up on this box\n' "${ok_count}" "${total_count}"
    printf '  OpenAI base URL:    http://127.0.0.1:8000/v1\n'
    printf '  Re-verify any time: RUNTIME_DIR=%s %s/verify.sh\n' "${RUNTIME_DIR}" "${RUNTIME_DIR}"
    printf '  View logs:          %s/.vllm.log\n' "${RUNTIME_DIR}"
    printf '  Tail container:     %s docker logs -f gemma4-vllm\n' "${SUDO}"
    exit 0
  fi
  printf '  FAIL  %d/%d steps passed. Re-run ./setup.sh to retry (each step is idempotent).\n' "${ok_count}" "${total_count}"
  exit 1
}

# ── Main ─────────────────────────────────────────────────────────────────
main() {
  local ok=0 total=0
  for n in 0 1 2 3 4 5 6 7 8 9 10; do
    total=$((total + 1))
    if step_${n}; then
      ok=$((ok + 1))
    else
      err "step ${n} failed — see fix: line above. Re-run ./setup.sh to retry."
      step_11 "${ok}" "${total}"
    fi
  done
  step_11 "${ok}" "${total}"
}

main "$@"
