#!/usr/bin/env bash
# Shared process launcher for config-owned training experiments.

open_wam_reject_cli_config_override_args() {
  local arg
  for arg in "$@"; do
    case "${arg}" in
      --config-name|--config-name=*|--cfg|--cfg=*|--config|--config=*)
        echo "Do not pass ${arg} to this launcher; set CONFIG_NAME=... instead." >&2
        return 2
        ;;
    esac
  done
}

open_wam_print_train_argv_json() {
  local argv_python="${OPEN_WAM_TRAIN_ARGV_PYTHON:-python3}"
  "${argv_python}" - "$@" <<'PY'
import json
import sys

print(json.dumps(sys.argv[1:]))
PY
}

open_wam_maybe_print_train_argv() {
  if [[ "${OPEN_WAM_PRINT_TRAIN_ARGV:-0}" == "1" ]]; then
    open_wam_print_train_argv_json "$@"
    exit 0
  fi
}

open_wam_launch_training() {
  local config_name="${1:?Pass the experiment config name as the first argument.}"
  shift
  local ngpu="${NGPU:-1}"
  local master_port="${MASTER_PORT:-29501}"
  local log_rank="${LOG_RANK:-0}"
  local -a train_args=()

  export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
  export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}
  export WANDB_MODE=${WANDB_MODE:-"online"}

  open_wam_reject_cli_config_override_args "$@"
  train_args=(
    --config-name "${config_name}"
    --devices "${ngpu}"
    "$@"
  )
  open_wam_maybe_print_train_argv "${train_args[@]}"

  if [ "${ngpu}" -gt 1 ]; then
    uv run python -m torch.distributed.run \
      --nproc_per_node="${ngpu}" \
      --local-ranks-filter="${log_rank}" \
      --master_port "${master_port}" \
      --tee 3 \
      -m open_wam.cli.train \
      "${train_args[@]}"
  else
    uv run python -m open_wam.cli.train \
      "${train_args[@]}"
  fi
}
