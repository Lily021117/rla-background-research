#!/usr/bin/env bash
set -euo pipefail
: "${TASK_DIR:?}" "${RLA_GPU_UUID:?}" "${AE_STAGE:?}"
export CUDA_VISIBLE_DEVICES="$RLA_GPU_UUID"
if [[ "$RLA_GPU_UUID" == GPU-205a3f5a-d91b-6385-c588-4e4cf3505f7e ]]; then
  export RLA_MIN_FREE_GIB="${RLA_MIN_FREE_GIB:-24.6}"
else export RLA_MIN_FREE_GIB="${RLA_MIN_FREE_GIB:-15}"; fi
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
export HF_HOME=/data/lyc/tmp/rla_background_invariance/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/data/lyc/hf_cache/hub
export TRANSFORMERS_CACHE=/data/lyc/hf_cache/transformers
export TORCH_HOME=/data/lyc/tmp/rla_background_invariance/cache/torch
export XDG_CACHE_HOME=/data/lyc/tmp/rla_background_invariance/cache/xdg
export TMPDIR=/data/lyc/tmp/rla_background_invariance
export NUMBA_CACHE_DIR="$TMPDIR/cache/numba"
export CUDA_CACHE_PATH="$TMPDIR/cache/cuda" TRITON_CACHE_DIR="$TMPDIR/cache/triton"
mkdir -p "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR"
project=/data/lyc/projects/rla-wm
python="$project/.venv-sidecar/bin/python"
src=experiments/official_libero_rla_consistency/ae_consistency
ae_run="${AE_RUN:-$TASK_DIR/ae_run}"
resume_args=()
if [[ -n "${AE_RESUME:-}" ]]; then resume_args=(--resume-from "$AE_RESUME" --output-name "${AE_OUTPUT_NAME:?}"); fi
case "$AE_STAGE" in
cache) exec "$python" "$src/cache_features.py" --project "$project" --dataset "$TASK_DIR" --run "$ae_run" ;;
C|D) exec "$python" "$src/train_ae.py" --project "$project" --run "$ae_run" --arm "$AE_STAGE" "${resume_args[@]}" ;;
evaluate|sidecar) exec "$python" "$src/evaluate_and_extract.py" --project "$project" --run "$ae_run" --mode "$AE_STAGE" ;;
rawcheck) exec "$python" "$src/raw_source_check.py" --project "$project" --run "$ae_run" ;;
*) exit 2 ;;
esac
