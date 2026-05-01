#!/usr/bin/env bash
# Source this file before running Open-WAM jobs on this local /data volume.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPEN_WAM_DATA_ROOT="${OPEN_WAM_DATA_ROOT:-/data}"

export OPEN_WAM_RUNS_ROOT="${OPEN_WAM_RUNS_ROOT:-${OPEN_WAM_DATA_ROOT}/openwam_exp/runs}"
export OPEN_WAM_LOG_DIR="${OPEN_WAM_LOG_DIR:-${OPEN_WAM_DATA_ROOT}/openwam_logs}"
export OPEN_WAM_LOCAL_PATHS="${OPEN_WAM_LOCAL_PATHS:-${REPO_ROOT}/configs/local_paths.yaml}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${OPEN_WAM_DATA_ROOT}/openwam_cache/uv}"
export HF_HOME="${HF_HOME:-${OPEN_WAM_DATA_ROOT}/openwam_cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export WANDB_DIR="${WANDB_DIR:-${OPEN_WAM_DATA_ROOT}/openwam_cache/wandb}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_NET="${OPEN_WAM_NCCL_NET:-Socket}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

mkdir -p "${OPEN_WAM_RUNS_ROOT}" "${OPEN_WAM_LOG_DIR}" "${UV_CACHE_DIR}" \
  "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${WANDB_DIR}"
