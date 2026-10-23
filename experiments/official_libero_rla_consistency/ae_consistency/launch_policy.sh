#!/usr/bin/env bash
set -euo pipefail
: "${RLA_RUN:?}" "${RLA_GPU_UUID:?}" "${POLICY_ARM:?}" "${RLA_REFERENCE:?}" "${RLA_PYTHON:?}" "${RLA_CACHE_ROOT:?}" "${RLA_SIDECAR:?}"
case "$RLA_CACHE_ROOT" in /data/*) ;; *) echo 'Cache must be on the data volume' >&2; exit 2;; esac
export CUDA_VISIBLE_DEVICES="$RLA_GPU_UUID"
export HF_HOME="${HF_HOME:-$RLA_CACHE_ROOT/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$RLA_CACHE_ROOT/transformers}"
export TORCH_HOME="$RLA_CACHE_ROOT/torch"
export XDG_CACHE_HOME="$RLA_CACHE_ROOT/xdg"
export TMPDIR="$RLA_CACHE_ROOT/tmp"
export NUMBA_CACHE_DIR="$RLA_CACHE_ROOT/numba"
mkdir -p "$TMPDIR" "$TORCH_HOME" "$XDG_CACHE_HOME" "$NUMBA_CACHE_DIR"
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export RLA_STEPS=8 RLA_QUERIES=16 RLA_DIM=64 RLA_NORM=center LAMBDA_ACT=1
export LAMBDA_RLA=0.22408536585365854 RLA_LOG_EVERY=1000
export PYTHONPATH="$PWD/experiments/official_libero_rla_consistency:${PYTHONPATH:-}"
exec "$RLA_PYTHON" experiments/official_libero_rla_consistency/baseline_update_smoke.py --reference "$RLA_REFERENCE" --run "$RLA_RUN/${RLA_RUN_SUBDIR:-$POLICY_ARM}" --batch-size 1 --experiment-entry "${POLICY_ENTRY:-ae_consistency.policy_training}"
