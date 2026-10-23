#!/usr/bin/env bash
set -euo pipefail
: "${TASK_DIR:?}" "${RLA_GPU_UUID:?}" "${AE_STAGE:?}"
export CUDA_VISIBLE_DEVICES="$RLA_GPU_UUID" RLA_MIN_FREE_GIB=15
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
export HF_HOME=/data/lyc/tmp/rla_background_invariance/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/data/lyc/hf_cache/hub
export TRANSFORMERS_CACHE=/data/lyc/hf_cache/transformers
export TORCH_HOME=/data/lyc/tmp/rla_background_invariance/cache/torch
export XDG_CACHE_HOME=/data/lyc/tmp/rla_background_invariance/cache/xdg
export TMPDIR=/data/lyc/tmp/rla_background_invariance
export NUMBA_CACHE_DIR="$TMPDIR/cache/numba"
project=/data/lyc/projects/rla-wm
python="$project/.venv-sidecar/bin/python"
src=experiments/official_libero_rla_consistency/ae_consistency
case "$AE_STAGE" in
cache) "$python" "$src/cache_features.py" --project "$project" --dataset "$TASK_DIR" --run "$TASK_DIR/ae_run" ;;
C|D) "$python" "$src/train_ae.py" --project "$project" --run "$TASK_DIR/ae_run" --arm "$AE_STAGE" ;;
evaluate|sidecar) "$python" "$src/evaluate_and_extract.py" --project "$project" --run "$TASK_DIR/ae_run" --mode "$AE_STAGE" ;;
*) exit 2 ;;
esac
