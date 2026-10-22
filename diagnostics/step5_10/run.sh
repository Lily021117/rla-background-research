#!/usr/bin/env bash
set -euo pipefail
: "${RLA_PROJECT_ROOT:?Set the accepted project root}"
: "${RUN_DIR:?Set a new empty diagnostic output directory}"
: "${RLA_CACHE_ROOT:?Set an isolated cache directory on the data disk}"
: "${DIAGNOSTIC_GPU_UUID:?Set the device approved after resource/ECC checks}"
: "${DIAGNOSTIC_PYTHON:?Set the existing accepted Python interpreter}"
TASK_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$RUN_DIR" "$RLA_CACHE_ROOT"
export TMPDIR="$RLA_CACHE_ROOT" HF_HOME="$RLA_CACHE_ROOT/hf" TORCH_HOME="$RLA_CACHE_ROOT/torch" XDG_CACHE_HOME="$RLA_CACHE_ROOT/xdg"
export CUDA_CACHE_PATH="$RLA_CACHE_ROOT/cuda" TRITON_CACHE_DIR="$RLA_CACHE_ROOT/triton" NUMBA_CACHE_DIR="$RLA_CACHE_ROOT/numba" MPLCONFIGDIR="$RLA_CACHE_ROOT/mpl"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=''
"$DIAGNOSTIC_PYTHON" "$TASK_SCRIPT_DIR/prepare.py" > "$RUN_DIR/prepare.log" 2>&1
export CUDA_VISIBLE_DEVICES="$DIAGNOSTIC_GPU_UUID"
"$DIAGNOSTIC_PYTHON" "$TASK_SCRIPT_DIR/swap.py" > "$RUN_DIR/swap.log" 2>&1
export CUDA_VISIBLE_DEVICES=''
"$DIAGNOSTIC_PYTHON" "$TASK_SCRIPT_DIR/analyze.py" > "$RUN_DIR/analysis.log" 2>&1
