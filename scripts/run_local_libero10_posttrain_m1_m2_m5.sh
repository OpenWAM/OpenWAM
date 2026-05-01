#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
source scripts/env_local_data_openwam.sh

MODE="${1:-preflight}"
DEFAULT_CASES="${DEFAULT_CASES:-m2,m5}"
RUN_ID="${RUN_ID:-libero10_posttrain_m1_m2_m5_local_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${OPEN_WAM_RUNS_ROOT}/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${OPEN_WAM_LOG_DIR}/${RUN_ID}}"
WAIT_SECONDS="${WAIT_SECONDS:-5400}"
POST_SLEEP_WAIT_SECONDS="${POST_SLEEP_WAIT_SECONDS:-3600}"
ASSET_POLL_SECONDS="${ASSET_POLL_SECONDS:-300}"
NUM_STEPS="${NUM_STEPS:-5000}"
RUN_SMOKE="${RUN_SMOKE:-1}"
GPU_MODE="${GPU_MODE:-auto}"
MIN_FREE_GPU_MB="${MIN_FREE_GPU_MB:-70000}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-openwam-libero10-posttrain}"
WANDB_MODE_REQUESTED="${WANDB_MODE:-online}"
ALLOW_OFFLINE_WANDB_FALLBACK="${ALLOW_OFFLINE_WANDB_FALLBACK:-1}"
ENABLE_CHECKPOINT_PRUNER="${ENABLE_CHECKPOINT_PRUNER:-1}"
CHECKPOINT_KEEP_LAST="${CHECKPOINT_KEEP_LAST:-1}"
CHECKPOINT_PRUNE_SECONDS="${CHECKPOINT_PRUNE_SECONDS:-60}"
CHECKPOINT_PRUNE_MIN_AGE_SECONDS="${CHECKPOINT_PRUNE_MIN_AGE_SECONDS:-900}"

DATA_ROOT="${DATA_ROOT:-/data/lingbot_data_exp/libero_heng/libero_10}"
EMPTY_TEXT_EMB="${EMPTY_TEXT_EMB:-/data/lingbot_data_exp/robotwin-clean-and-aug-lerobot/empty_emb.pt}"
LINGBOT_BASE="${LINGBOT_BASE:-/data/lingbot_data_exp/pretrain_models/lingbot-va-base}"
M1_CKPT="${M1_CKPT:-/data/openwam_exp/runs/parallel_stream_libero_lingbot_exact_heng_compatible/checkpoints/checkpoint_step_400}"
M2_CKPT="${M2_CKPT:-/data/openwam_exp/runs/parallel_stream_libero_lingbot_joint_denoise_heng_compatible/checkpoints/checkpoint_step_600}"
M5_CKPT="${M5_CKPT:-/data/openwam_exp/runs/method5_mot_full_segment_non_joint_action_only_libero/checkpoints/checkpoint_step_2800}"
M1_RESUME_FROM_EXPLICIT="${M1_RESUME_FROM+x}"
M1_TRANSFORMER_SUBDIR_EXPLICIT="${M1_TRANSFORMER_SUBDIR+x}"
M2_RESUME_FROM_EXPLICIT="${M2_RESUME_FROM+x}"
M2_TRANSFORMER_SUBDIR_EXPLICIT="${M2_TRANSFORMER_SUBDIR+x}"
M5_RESUME_FROM_EXPLICIT="${M5_RESUME_FROM+x}"
M5_TRANSFORMER_SUBDIR_EXPLICIT="${M5_TRANSFORMER_SUBDIR+x}"
M1_RESUME_FROM="${M1_RESUME_FROM:-${M1_CKPT}/model_state.pt}"
M1_TRANSFORMER_SUBDIR="${M1_TRANSFORMER_SUBDIR:-${M1_CKPT}/transformer}"
M2_RESUME_FROM="${M2_RESUME_FROM:-${M2_CKPT}/model_state.pt}"
M2_TRANSFORMER_SUBDIR="${M2_TRANSFORMER_SUBDIR:-${M2_CKPT}/transformer}"
M5_RESUME_FROM="${M5_RESUME_FROM:-${M5_CKPT}/model_state.pt}"
M5_TRANSFORMER_SUBDIR="${M5_TRANSFORMER_SUBDIR:-${M5_CKPT}/transformer}"

M1_SAVE_ROOT="${M1_SAVE_ROOT:-${RUN_ROOT}/m1_exact_libero10_from_all_subsets_step400}"
M2_SAVE_ROOT="${M2_SAVE_ROOT:-${RUN_ROOT}/m2_joint_libero10_from_all_subsets_step600}"
M5_SAVE_ROOT="${M5_SAVE_ROOT:-${RUN_ROOT}/m5_action_only_libero10_from_all_subsets_step2800}"
M1_GPUS="${M1_GPUS:-0,1}"
M2_GPUS="${M2_GPUS:-0,1}"
M5_GPUS="${M5_GPUS:-2,3}"
SEQ_GPUS="${SEQ_GPUS:-0,1}"

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*" >&2
}

count_csv_items() {
  local csv="$1"
  awk -F',' '{print NF}' <<<"${csv}"
}

has_wandb_auth() {
  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    return 0
  fi
  uv run python - <<'PY' >/dev/null 2>&1
from netrc import netrc
try:
    raise SystemExit(0 if netrc().authenticators("api.wandb.ai") else 1)
except Exception:
    raise SystemExit(1)
PY
}

effective_wandb_mode() {
  if [[ "${WANDB_MODE_REQUESTED}" != "online" ]]; then
    printf '%s\n' "${WANDB_MODE_REQUESTED}"
    return 0
  fi
  if has_wandb_auth; then
    printf 'online\n'
    return 0
  fi
  if [[ "${ALLOW_OFFLINE_WANDB_FALLBACK}" == "1" ]]; then
    log "WandB online requested but no auth is present; using offline mode."
    printf 'offline\n'
    return 0
  fi
  log "Missing WandB auth and offline fallback is disabled."
  return 1
}

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "${path}" ]]; then
    log "missing directory: ${label}: ${path}"
    return 1
  fi
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    log "missing file: ${label}: ${path}"
    return 1
  fi
}

count_files() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    printf '0\n'
    return 0
  fi
  find "${path}" -type f | wc -l
}

count_transformer_weights() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    printf '0\n'
    return 0
  fi
  find "${path}" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' \) | wc -l
}

checkpoint_step_number() {
  local path="$1"
  local name
  name="$(basename "${path}")"
  printf '%s\n' "${name#checkpoint_step_}"
}

checkpoint_is_complete() {
  local path="$1"
  [[ -s "${path}/full_training_state.pt" ]] || return 1
  [[ -s "${path}/model_state.pt" ]] || return 1
  [[ -s "${path}/train_state.json" ]] || return 1
  [[ -s "${path}/resolved_config.yaml" ]] || return 1
  [[ -s "${path}/transformer/config.json" ]] || return 1
  [[ "$(count_transformer_weights "${path}/transformer")" -gt 0 ]] || return 1
}

checkpoint_last_modified() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    date +%s
    return 0
  fi
  find "${path}" -type f -printf '%T@\n' 2>/dev/null | sort -nr | head -n 1 | cut -d. -f1
}

latest_complete_checkpoint_dir() {
  local save_root="$1"
  local checkpoint_root="${save_root}/checkpoints"
  local latest=""
  [[ -d "${checkpoint_root}" ]] || return 1
  while IFS= read -r checkpoint_dir; do
    if checkpoint_is_complete "${checkpoint_dir}"; then
      latest="${checkpoint_dir}"
    fi
  done < <(find "${checkpoint_root}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint_step_*' | sort -V)
  [[ -n "${latest}" ]] || return 1
  printf '%s\n' "${latest}"
}

checkpoint_dir_from_resume_path() {
  local resume_from="$1"
  if [[ -d "${resume_from}" ]]; then
    printf '%s\n' "${resume_from}"
    return 0
  fi
  case "$(basename "${resume_from}")" in
    full_training_state.pt|model_state.pt)
      dirname "${resume_from}"
      ;;
    *)
      return 1
      ;;
  esac
}

checkpoint_has_runtime_transformer() {
  local checkpoint_dir="$1"
  [[ -d "${checkpoint_dir}/transformer" ]] || return 1
  [[ -s "${checkpoint_dir}/transformer/config.json" ]] || return 1
  [[ "$(count_transformer_weights "${checkpoint_dir}/transformer")" -gt 0 ]] || return 1
}

case_list() {
  local cases_csv="${1:-${DEFAULT_CASES}}"
  tr ',' '\n' <<<"${cases_csv}" | awk '{gsub(/^[[:space:]]+|[[:space:]]+$/, ""); if (length($0) > 0) print $0}'
}

case_save_root() {
  case "$1" in
    m1) printf '%s\n' "${M1_SAVE_ROOT}" ;;
    m2) printf '%s\n' "${M2_SAVE_ROOT}" ;;
    m5) printf '%s\n' "${M5_SAVE_ROOT}" ;;
    *) log "unknown case: $1"; return 2 ;;
  esac
}

case_gpus() {
  case "$1" in
    m1) printf '%s\n' "${M1_GPUS}" ;;
    m2) printf '%s\n' "${M2_GPUS}" ;;
    m5) printf '%s\n' "${M5_GPUS}" ;;
    *) log "unknown case: $1"; return 2 ;;
  esac
}

case_port() {
  case "$1" in
    m1) printf '29611\n' ;;
    m2) printf '29621\n' ;;
    m5) printf '29631\n' ;;
    *) log "unknown case: $1"; return 2 ;;
  esac
}

case_smoke_port() {
  case "$1" in
    m1) printf '29661\n' ;;
    m2) printf '29641\n' ;;
    m5) printf '29651\n' ;;
    *) log "unknown case: $1"; return 2 ;;
  esac
}

case_checkpoint_root() {
  case "$1" in
    m1) printf '%s\n' "${M1_CKPT}" ;;
    m2) printf '%s\n' "${M2_CKPT}" ;;
    m5) printf '%s\n' "${M5_CKPT}" ;;
    *) log "unknown case: $1"; return 2 ;;
  esac
}

case_resume_from_explicit() {
  case "$1" in
    m1) [[ -n "${M1_RESUME_FROM_EXPLICIT}" ]] ;;
    m2) [[ -n "${M2_RESUME_FROM_EXPLICIT}" ]] ;;
    m5) [[ -n "${M5_RESUME_FROM_EXPLICIT}" ]] ;;
    *) log "unknown case: $1"; return 2 ;;
  esac
}

case_transformer_subdir_explicit() {
  case "$1" in
    m1) [[ -n "${M1_TRANSFORMER_SUBDIR_EXPLICIT}" ]] ;;
    m2) [[ -n "${M2_TRANSFORMER_SUBDIR_EXPLICIT}" ]] ;;
    m5) [[ -n "${M5_TRANSFORMER_SUBDIR_EXPLICIT}" ]] ;;
    *) log "unknown case: $1"; return 2 ;;
  esac
}

case_base_resume_from() {
  case "$1" in
    m1) printf '%s\n' "${M1_RESUME_FROM}" ;;
    m2) printf '%s\n' "${M2_RESUME_FROM}" ;;
    m5) printf '%s\n' "${M5_RESUME_FROM}" ;;
    *) log "unknown case: $1"; return 2 ;;
  esac
}

case_base_transformer_subdir() {
  case "$1" in
    m1) printf '%s\n' "${M1_TRANSFORMER_SUBDIR}" ;;
    m2) printf '%s\n' "${M2_TRANSFORMER_SUBDIR}" ;;
    m5) printf '%s\n' "${M5_TRANSFORMER_SUBDIR}" ;;
    *) log "unknown case: $1"; return 2 ;;
  esac
}

resolve_case_resume_from() {
  local case_key="$1"
  local save_root="$2"
  local latest_checkpoint
  if ! case_resume_from_explicit "${case_key}" && latest_checkpoint="$(latest_complete_checkpoint_dir "${save_root}")"; then
    printf '%s\n' "${latest_checkpoint}/full_training_state.pt"
    return 0
  fi
  case_base_resume_from "${case_key}"
}

resolve_case_transformer_subdir() {
  local case_key="$1"
  local resume_from="$2"
  local checkpoint_dir
  if ! case_transformer_subdir_explicit "${case_key}"; then
    if checkpoint_dir="$(checkpoint_dir_from_resume_path "${resume_from}")" \
      && checkpoint_has_runtime_transformer "${checkpoint_dir}"; then
      printf '%s\n' "${checkpoint_dir}/transformer"
      return 0
    fi
  fi
  case_base_transformer_subdir "${case_key}"
}

prune_checkpoints_once() {
  local case_key="$1"
  local save_root="$2"
  local keep_last="${3:-${CHECKPOINT_KEEP_LAST}}"
  local min_age_seconds="${4:-${CHECKPOINT_PRUNE_MIN_AGE_SECONDS}}"
  local checkpoint_root="${save_root}/checkpoints"
  [[ "${keep_last}" -ge 1 ]] || keep_last=1
  [[ -d "${checkpoint_root}" ]] || return 0

  local complete_list
  complete_list="$(mktemp)"
  while IFS= read -r checkpoint_dir; do
    if checkpoint_is_complete "${checkpoint_dir}"; then
      printf '%s\t%s\n' "$(checkpoint_step_number "${checkpoint_dir}")" "${checkpoint_dir}" >>"${complete_list}"
    fi
  done < <(find "${checkpoint_root}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint_step_*' | sort)

  local complete_count delete_count index checkpoint_dir
  complete_count="$(wc -l <"${complete_list}" | tr -d ' ')"
  delete_count=$((complete_count - keep_last))
  if (( delete_count > 0 )); then
    index=0
    while IFS=$'\t' read -r _ checkpoint_dir; do
      if (( index >= delete_count )); then
        break
      fi
      log "checkpoint_prune: ${case_key}: deleting older complete checkpoint ${checkpoint_dir}"
      rm -rf --one-file-system "${checkpoint_dir}"
      index=$((index + 1))
    done < <(sort -n "${complete_list}")
  fi
  rm -f "${complete_list}"

  local now mtime age newest_checkpoint_dir
  now="$(date +%s)"
  newest_checkpoint_dir="$(
    find "${checkpoint_root}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint_step_*' \
      | sort -V \
      | tail -n 1
  )"
  while IFS= read -r checkpoint_dir; do
    if checkpoint_is_complete "${checkpoint_dir}"; then
      continue
    fi
    if [[ -n "${newest_checkpoint_dir}" && "${checkpoint_dir}" == "${newest_checkpoint_dir}" ]]; then
      continue
    fi
    mtime="$(checkpoint_last_modified "${checkpoint_dir}")"
    if [[ -z "${mtime}" ]]; then
      mtime="$(stat -c %Y "${checkpoint_dir}" 2>/dev/null || printf '%s\n' "${now}")"
    fi
    age=$((now - mtime))
    if (( age >= min_age_seconds )); then
      log "checkpoint_prune: ${case_key}: deleting stale incomplete checkpoint ${checkpoint_dir}"
      rm -rf --one-file-system "${checkpoint_dir}"
    fi
  done < <(find "${checkpoint_root}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint_step_*' | sort)
}

preflight_case() {
  local case_key="$1"
  local ok=0
  case "${case_key}" in
    m1)
      require_dir "${M1_CKPT}" "M1 checkpoint root" || ok=1
      require_file "${M1_CKPT}/model_state.pt" "M1 model_state.pt" || ok=1
      require_dir "${M1_CKPT}/transformer" "M1 transformer export" || ok=1
      require_file "${M1_CKPT}/transformer/config.json" "M1 transformer config" || ok=1
      require_file "${M1_RESUME_FROM}" "M1 resume checkpoint" || ok=1
      require_dir "${M1_TRANSFORMER_SUBDIR}" "M1 runtime transformer subdir" || ok=1
      require_file "${M1_TRANSFORMER_SUBDIR}/config.json" "M1 runtime transformer config" || ok=1
      if [[ "$(count_transformer_weights "${M1_CKPT}/transformer")" -le 0 ]]; then
        log "missing M1 transformer weight file under ${M1_CKPT}/transformer"
        ok=1
      fi
      if [[ "$(count_transformer_weights "${M1_TRANSFORMER_SUBDIR}")" -le 0 ]]; then
        log "missing M1 runtime transformer weight file under ${M1_TRANSFORMER_SUBDIR}"
        ok=1
      fi
      ;;
    m2)
      require_dir "${M2_CKPT}" "M2 checkpoint root" || ok=1
      require_file "${M2_CKPT}/model_state.pt" "M2 model_state.pt" || ok=1
      require_dir "${M2_CKPT}/transformer" "M2 transformer export" || ok=1
      require_file "${M2_CKPT}/transformer/config.json" "M2 transformer config" || ok=1
      require_file "${M2_RESUME_FROM}" "M2 resume checkpoint" || ok=1
      require_dir "${M2_TRANSFORMER_SUBDIR}" "M2 runtime transformer subdir" || ok=1
      require_file "${M2_TRANSFORMER_SUBDIR}/config.json" "M2 runtime transformer config" || ok=1
      if [[ "$(count_transformer_weights "${M2_CKPT}/transformer")" -le 0 ]]; then
        log "missing M2 transformer weight file under ${M2_CKPT}/transformer"
        ok=1
      fi
      if [[ "$(count_transformer_weights "${M2_TRANSFORMER_SUBDIR}")" -le 0 ]]; then
        log "missing M2 runtime transformer weight file under ${M2_TRANSFORMER_SUBDIR}"
        ok=1
      fi
      ;;
    m5)
      require_dir "${M5_CKPT}" "M5 checkpoint root" || ok=1
      require_file "${M5_CKPT}/model_state.pt" "M5 model_state.pt" || ok=1
      require_dir "${M5_CKPT}/transformer" "M5 transformer export" || ok=1
      require_file "${M5_CKPT}/transformer/config.json" "M5 transformer config" || ok=1
      require_file "${M5_RESUME_FROM}" "M5 resume checkpoint" || ok=1
      require_dir "${M5_TRANSFORMER_SUBDIR}" "M5 runtime transformer subdir" || ok=1
      require_file "${M5_TRANSFORMER_SUBDIR}/config.json" "M5 runtime transformer config" || ok=1
      if [[ "$(count_transformer_weights "${M5_CKPT}/transformer")" -le 0 ]]; then
        log "missing M5 transformer weight file under ${M5_CKPT}/transformer"
        ok=1
      fi
      if [[ "$(count_transformer_weights "${M5_TRANSFORMER_SUBDIR}")" -le 0 ]]; then
        log "missing M5 runtime transformer weight file under ${M5_TRANSFORMER_SUBDIR}"
        ok=1
      fi
      ;;
    *)
      log "unknown preflight case: ${case_key}"
      ok=1
      ;;
  esac
  return "${ok}"
}

preflight() {
  local cases_csv="${1:-${DEFAULT_CASES}}"
  local ok=0 case_key
  log "preflight: cases=${cases_csv} data=${DATA_ROOT}"
  require_dir "${DATA_ROOT}" "LIBERO-10 data root" || ok=1
  require_dir "${DATA_ROOT}/data" "LIBERO-10 parquet data" || ok=1
  require_dir "${DATA_ROOT}/latents" "LIBERO-10 latent data" || ok=1
  require_dir "${DATA_ROOT}/meta" "LIBERO-10 metadata" || ok=1
  require_file "${DATA_ROOT}/meta/info.json" "LIBERO metadata info" || ok=1
  require_file "${DATA_ROOT}/meta/episodes.jsonl" "LIBERO episodes metadata" || ok=1
  require_file "${DATA_ROOT}/meta/tasks.jsonl" "LIBERO task metadata" || ok=1
  require_file "${EMPTY_TEXT_EMB}" "empty text embedding" || ok=1
  require_dir "${LINGBOT_BASE}" "LingBot base model" || ok=1

  local parquet_count latent_count
  parquet_count="$(count_files "${DATA_ROOT}/data")"
  latent_count="$(count_files "${DATA_ROOT}/latents")"
  log "preflight: parquet_files=${parquet_count} latent_files=${latent_count}"
  if [[ "${parquet_count}" -le 0 ]]; then
    log "LIBERO parquet data is empty."
    ok=1
  fi
  if [[ "${latent_count}" -le 0 ]]; then
    log "LIBERO latent data is empty."
    ok=1
  fi

  while IFS= read -r case_key; do
    preflight_case "${case_key}" || ok=1
  done < <(case_list "${cases_csv}")

  if ! effective_wandb_mode >/dev/null; then
    ok=1
  fi

  if [[ "${ok}" -eq 0 ]]; then
    log "preflight: ok"
    return 0
  fi
  log "preflight: blocked"
  return 1
}

write_manifest() {
  local launch_mode="$1"
  local wandb_mode="$2"
  cat >"${LOG_ROOT}/manifest.env" <<EOF
RUN_ID=${RUN_ID}
RUN_ROOT=${RUN_ROOT}
LOG_ROOT=${LOG_ROOT}
LAUNCH_MODE=${launch_mode}
DEFAULT_CASES=${DEFAULT_CASES}
NUM_STEPS=${NUM_STEPS}
RUN_SMOKE=${RUN_SMOKE}
WANDB_MODE=${wandb_mode}
WANDB_PROJECT=${WANDB_PROJECT_NAME}
ENABLE_CHECKPOINT_PRUNER=${ENABLE_CHECKPOINT_PRUNER}
CHECKPOINT_KEEP_LAST=${CHECKPOINT_KEEP_LAST}
CHECKPOINT_PRUNE_SECONDS=${CHECKPOINT_PRUNE_SECONDS}
CHECKPOINT_PRUNE_MIN_AGE_SECONDS=${CHECKPOINT_PRUNE_MIN_AGE_SECONDS}
DATA_ROOT=${DATA_ROOT}
EMPTY_TEXT_EMB=${EMPTY_TEXT_EMB}
LINGBOT_BASE=${LINGBOT_BASE}
M1_CKPT=${M1_CKPT}
M1_RESUME_FROM=${M1_RESUME_FROM}
M1_TRANSFORMER_SUBDIR=${M1_TRANSFORMER_SUBDIR}
M1_SAVE_ROOT=${M1_SAVE_ROOT}
M1_GPUS=${M1_GPUS}
M2_CKPT=${M2_CKPT}
M2_RESUME_FROM=${M2_RESUME_FROM}
M2_TRANSFORMER_SUBDIR=${M2_TRANSFORMER_SUBDIR}
M2_SAVE_ROOT=${M2_SAVE_ROOT}
M2_GPUS=${M2_GPUS}
M5_CKPT=${M5_CKPT}
M5_RESUME_FROM=${M5_RESUME_FROM}
M5_TRANSFORMER_SUBDIR=${M5_TRANSFORMER_SUBDIR}
M5_SAVE_ROOT=${M5_SAVE_ROOT}
M5_GPUS=${M5_GPUS}
EOF
}

gpu_count() {
  nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | wc -l
}

free_gpu_count() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
    | awk -v min="${MIN_FREE_GPU_MB}" '$1 >= min {count += 1} END {print count + 0}'
}

choose_launch_mode() {
  if [[ "${GPU_MODE}" == "concurrent" || "${GPU_MODE}" == "sequential" ]]; then
    printf '%s\n' "${GPU_MODE}"
    return 0
  fi
  local total free
  total="$(gpu_count)"
  free="$(free_gpu_count)"
  log "gpu availability: total=${total} free_at_or_above_${MIN_FREE_GPU_MB}MB=${free}"
  if [[ "${total}" -ge 4 && "${free}" -ge 4 ]]; then
    printf 'concurrent\n'
  else
    printf 'sequential\n'
  fi
}

case_command() {
  local case_key="$1"
  local save_root="$2"
  local ckpt_root="$3"
  local num_steps="$4"
  local wandb_mode="$5"
  local smoke="$6"
  shift 6
  local launcher config_name wandb_flag resume_from transformer_subdir
  if [[ "${case_key}" == "m1" ]]; then
    launcher="scripts/run_parallel_stream_posttrain_libero.sh"
    config_name="parallel_stream_libero_lingbot_exact_heng_compatible"
  elif [[ "${case_key}" == "m2" ]]; then
    launcher="scripts/run_parallel_stream_posttrain_libero.sh"
    config_name="parallel_stream_libero_lingbot_joint_denoise_heng_compatible"
  elif [[ "${case_key}" == "m5" ]]; then
    launcher="scripts/run_mot_full_segment_nonjoint_libero.sh"
    config_name="mot_libero_latent_local_full_segment_non_joint_action_only"
  else
    log "unknown case: ${case_key}"
    return 2
  fi
  resume_from="$(resolve_case_resume_from "${case_key}" "${save_root}")"
  transformer_subdir="$(resolve_case_transformer_subdir "${case_key}" "${resume_from}")"
  if [[ "${smoke}" == "1" ]]; then
    wandb_flag="--disable-wandb"
  else
    wandb_flag="--enable-wandb"
  fi

  log "case_config: ${case_key}: resume_from=${resume_from} transformer_subdir=${transformer_subdir}"
  CONFIG_NAME="${config_name}" \
  WANDB_MODE="${wandb_mode}" \
  WANDB_PROJECT="${WANDB_PROJECT_NAME}" \
  bash "${launcher}" \
    --save-root "${save_root}" \
    --resume-from "${resume_from}" \
    --transformer-subdir "${transformer_subdir}" \
    "${wandb_flag}" \
    --wandb-mode "${wandb_mode}" \
    --wandb-project "${WANDB_PROJECT_NAME}" \
    --set "data.local_root=${DATA_ROOT}" \
    --set "data.empty_text_embedding_path=${EMPTY_TEXT_EMB}" \
    --set "backbone.pretrained_model_name_or_path=${LINGBOT_BASE}" \
    --set "training.num_steps=${num_steps}" \
    "$@"
}

run_case_foreground() {
  local case_key="$1"
  local gpus="$2"
  local port="$3"
  local save_root="$4"
  local ckpt_root="$5"
  local log_path="$6"
  local num_steps="$7"
  local wandb_mode="$8"
  local smoke="$9"
  shift 9
  local ngpu
  ngpu="$(count_csv_items "${gpus}")"
  mkdir -p "$(dirname "${log_path}")" "${save_root}"
  log "starting ${case_key}: gpus=${gpus} ngpu=${ngpu} save_root=${save_root} log=${log_path}"
  (
    export CUDA_VISIBLE_DEVICES="${gpus}"
    export NGPU="${ngpu}"
    export MASTER_PORT="${port}"
    export LOG_RANK=0
    case_command "${case_key}" "${save_root}" "${ckpt_root}" "${num_steps}" "${wandb_mode}" "${smoke}" "$@"
  ) >"${log_path}" 2>&1
}

start_case_background() {
  local case_key="$1"
  local gpus="$2"
  local port="$3"
  local save_root="$4"
  local ckpt_root="$5"
  local log_path="$6"
  local num_steps="$7"
  local wandb_mode="$8"
  local ngpu
  ngpu="$(count_csv_items "${gpus}")"
  mkdir -p "$(dirname "${log_path}")" "${save_root}"
  log "starting ${case_key} in background: gpus=${gpus} ngpu=${ngpu} save_root=${save_root} log=${log_path}"
  (
    export CUDA_VISIBLE_DEVICES="${gpus}"
    export NGPU="${ngpu}"
    export MASTER_PORT="${port}"
    export LOG_RANK=0
    case_command "${case_key}" "${save_root}" "${ckpt_root}" "${num_steps}" "${wandb_mode}" "0"
  ) >"${log_path}" 2>&1 &
  local pid=$!
  printf '%s\n' "${pid}" >"${LOG_ROOT}/${case_key}.pid"
  log "${case_key} pid=${pid}"
}

start_checkpoint_pruner() {
  local case_key="$1"
  local save_root="$2"
  local log_path="$3"
  PRUNER_PID=""
  if [[ "${ENABLE_CHECKPOINT_PRUNER}" != "1" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "${log_path}")"
  (
    while true; do
      prune_checkpoints_once "${case_key}" "${save_root}" "${CHECKPOINT_KEEP_LAST}" "${CHECKPOINT_PRUNE_MIN_AGE_SECONDS}" || true
      sleep "${CHECKPOINT_PRUNE_SECONDS}"
    done
  ) >"${log_path}" 2>&1 &
  local pid=$!
  log "${case_key} checkpoint pruner pid=${pid} keep_last=${CHECKPOINT_KEEP_LAST} interval=${CHECKPOINT_PRUNE_SECONDS}s"
  PRUNER_PID="${pid}"
}

stop_checkpoint_pruner() {
  local pid="${1:-}"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
}

run_smoke_checks() {
  local cases_csv="${1:-${DEFAULT_CASES}}"
  local case_key gpus port ckpt_root
  while IFS= read -r case_key; do
    gpus="$(case_gpus "${case_key}")"
    port="$(case_smoke_port "${case_key}")"
    ckpt_root="$(case_checkpoint_root "${case_key}")"
    log "smoke: ${case_key} one optimizer step"
    run_case_foreground "${case_key}" "${gpus}" "${port}" "${RUN_ROOT}/_smoke_${case_key}" "${ckpt_root}" \
      "${LOG_ROOT}/${case_key}_smoke.log" 1 disabled 1 \
      --set "training.gradient_accumulation_steps=1" \
      --set "trainer.save_interval=null" \
      --set "trainer.enable_checkpointing=false" \
      --set "trainer.limit_val_batches=0" \
      --set "data.num_workers=2"
  done < <(case_list "${cases_csv}")
  log "smoke: ok"
}

launch_training() {
  preflight "${DEFAULT_CASES}"
  local wandb_mode launch_mode
  wandb_mode="$(effective_wandb_mode)"
  launch_mode="$(choose_launch_mode)"
  write_manifest "${launch_mode}" "${wandb_mode}"
  if [[ "${RUN_SMOKE}" == "1" ]]; then
    run_smoke_checks "${DEFAULT_CASES}"
  else
    log "smoke: skipped"
  fi
  local -a launch_cases=()
  local case_key
  while IFS= read -r case_key; do
    launch_cases+=("${case_key}")
  done < <(case_list "${DEFAULT_CASES}")
  if [[ "${#launch_cases[@]}" -eq 0 ]]; then
    log "no launch cases selected"
    return 2
  fi
  if [[ "${launch_mode}" == "concurrent" ]]; then
    local -a pruner_pids=()
    local -a statuses=()
    local save_root ckpt_root gpus port pruner_pid pid status
    for case_key in "${launch_cases[@]}"; do
      save_root="$(case_save_root "${case_key}")"
      ckpt_root="$(case_checkpoint_root "${case_key}")"
      gpus="$(case_gpus "${case_key}")"
      port="$(case_port "${case_key}")"
      start_case_background "${case_key}" "${gpus}" "${port}" "${save_root}" "${ckpt_root}" \
        "${LOG_ROOT}/${case_key}_train.log" "${NUM_STEPS}" "${wandb_mode}"
      start_checkpoint_pruner "${case_key}" "${save_root}" "${LOG_ROOT}/${case_key}_checkpoint_pruner.log" || true
      pruner_pids+=("${PRUNER_PID}")
    done
    local failed=0
    set +e
    for case_key in "${launch_cases[@]}"; do
      pid="$(cat "${LOG_ROOT}/${case_key}.pid")"
      wait "${pid}"
      status=$?
      statuses+=("${case_key}:${status}")
      if [[ "${status}" -ne 0 ]]; then
        failed=1
      fi
    done
    set -e
    for pruner_pid in "${pruner_pids[@]}"; do
      stop_checkpoint_pruner "${pruner_pid}"
    done
    if [[ "${failed}" -ne 0 ]]; then
      log "training failed: ${statuses[*]}"
      return 1
    fi
  else
    local pruner_pid status save_root ckpt_root port
    for case_key in "${launch_cases[@]}"; do
      save_root="$(case_save_root "${case_key}")"
      ckpt_root="$(case_checkpoint_root "${case_key}")"
      port="$(case_port "${case_key}")"
      start_checkpoint_pruner "${case_key}" "${save_root}" "${LOG_ROOT}/${case_key}_checkpoint_pruner.log" || true
      pruner_pid="${PRUNER_PID}"
      set +e
      run_case_foreground "${case_key}" "${SEQ_GPUS}" "${port}" "${save_root}" "${ckpt_root}" \
        "${LOG_ROOT}/${case_key}_train.log" "${NUM_STEPS}" "${wandb_mode}" 0
      status=$?
      set -e
      stop_checkpoint_pruner "${pruner_pid}"
      [[ "${status}" -eq 0 ]] || return "${status}"
    done
  fi
}

launch_single_case() {
  local case_key="$1"
  preflight "${case_key}"
  local wandb_mode pruner_pid status save_root ckpt_root gpus port
  wandb_mode="$(effective_wandb_mode)"
  write_manifest "single-${case_key}" "${wandb_mode}"
  if [[ "${RUN_SMOKE}" == "1" ]]; then
    log "smoke: skipped for single-${case_key}"
  else
    log "smoke: skipped"
  fi

  save_root="$(case_save_root "${case_key}")"
  ckpt_root="$(case_checkpoint_root "${case_key}")"
  gpus="$(case_gpus "${case_key}")"
  port="$(case_port "${case_key}")"
  start_checkpoint_pruner "${case_key}" "${save_root}" "${LOG_ROOT}/${case_key}_checkpoint_pruner.log" || true
  pruner_pid="${PRUNER_PID}"
  set +e
  run_case_foreground "${case_key}" "${gpus}" "${port}" "${save_root}" "${ckpt_root}" "${LOG_ROOT}/${case_key}_train.log" "${NUM_STEPS}" "${wandb_mode}" 0
  status=$?
  set -e
  stop_checkpoint_pruner "${pruner_pid}"
  return "${status}"
}

prune_cases_once() {
  local cases_csv="${1:-m1,m2,m5}"
  local case_key save_root
  while IFS= read -r case_key; do
    save_root="$(case_save_root "${case_key}")"
    prune_checkpoints_once "${case_key}" "${save_root}" "${CHECKPOINT_KEEP_LAST}" "${CHECKPOINT_PRUNE_MIN_AGE_SECONDS}"
  done < <(case_list "${cases_csv}")
}

wait_for_assets_after_sleep() {
  log "sleeping for ${WAIT_SECONDS}s before asset preflight"
  sleep "${WAIT_SECONDS}"
  local deadline=$((SECONDS + POST_SLEEP_WAIT_SECONDS))
  until preflight; do
    if (( SECONDS >= deadline )); then
      log "assets did not pass preflight within ${POST_SLEEP_WAIT_SECONDS}s after sleep"
      return 1
    fi
    log "assets not ready; polling again in ${ASSET_POLL_SECONDS}s"
    sleep "${ASSET_POLL_SECONDS}"
  done
}

case "${MODE}" in
  preflight)
    preflight
    ;;
  preflight-m1)
    preflight m1
    ;;
  preflight-m2)
    preflight m2
    ;;
  preflight-m5)
    preflight m5
    ;;
  smoke)
    preflight "${DEFAULT_CASES}"
    run_smoke_checks "${DEFAULT_CASES}"
    ;;
  prune-once)
    prune_cases_once
    ;;
  launch)
    launch_training
    ;;
  launch-m1)
    launch_single_case m1
    ;;
  launch-m2)
    launch_single_case m2
    ;;
  launch-m5)
    launch_single_case m5
    ;;
  wait-launch)
    {
      log "supervisor start: run_id=${RUN_ID}"
      wait_for_assets_after_sleep
      launch_training
      log "supervisor finished"
    } 2>&1 | tee -a "${LOG_ROOT}/supervisor.log"
    ;;
  *)
    echo "Usage: $0 {preflight|preflight-m1|preflight-m2|preflight-m5|smoke|prune-once|launch|launch-m1|launch-m2|launch-m5|wait-launch}" >&2
    exit 2
    ;;
esac
