#!/usr/bin/env bash
set -euo pipefail
: "${RLA_PROJECT_ROOT:?}" "${RUN_DIR:?}"
export HF_HOME="$RUN_DIR/cache/hf" TORCH_HOME="$RUN_DIR/cache/torch"
export XDG_CACHE_HOME="$RUN_DIR/cache/xdg" TMPDIR="$RUN_DIR/tmp"
export MPLCONFIGDIR="$RUN_DIR/cache/mpl" CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMPDIR" "$MPLCONFIGDIR"
export PYTHONPATH="$RLA_PROJECT_ROOT/worktrees/rla-official-0fc6e97:$RLA_PROJECT_ROOT/worktrees/rla-official-0fc6e97/third_party/diffusion_policy"
"$RLA_PROJECT_ROOT/.venv/bin/python" "$(dirname "$0")/audit.py"
