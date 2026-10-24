#!/usr/bin/env bash
set -euo pipefail
: "${TASK_DIR:?}" "${RLA_GPU_UUID:?}" "${POLICY_ARM:?}" "${AE_RUN:?}"
if [[ "$POLICY_ARM" =~ ^[BCD]$ && -f "$TASK_DIR/policy_run/$POLICY_ARM/isolated_smoke/execution_configuration.json" ]]; then
  /data/lyc/projects/vla-adapter-ref/.venv-cpu/bin/python experiments/official_libero_rla_consistency/multitask/wait_existing.py --arm-root "$TASK_DIR/policy_run/$POLICY_ARM"
  exit 0
fi
if [[ "$POLICY_ARM" =~ ^[BCD]$ && -f "$TASK_DIR/policy_run/$POLICY_ARM/isolated_smoke/training_acceptance.json" ]]; then
  /data/lyc/projects/vla-adapter-ref/.venv-cpu/bin/python -c 'import json,sys,pathlib; p=pathlib.Path(sys.argv[1]); x=json.loads(p.read_text()); assert x["successful_updates"]==600 and x["strict_reload"] and (p.parent/"step0600.pt").is_file(); print("REUSE COMPLETED FIXED600 WITHOUT UPDATES",p)' "$TASK_DIR/policy_run/$POLICY_ARM/isolated_smoke/training_acceptance.json"
  exit 0
fi
export RLA_RUN="$TASK_DIR/policy_run"
if [[ "$RLA_GPU_UUID" == GPU-205a3f5a-d91b-6385-c588-4e4cf3505f7e ]]; then
  export RLA_MIN_FREE_GIB="${RLA_MIN_FREE_GIB:-24.6}"
else export RLA_MIN_FREE_GIB="${RLA_MIN_FREE_GIB:-15}"; fi
export RLA_MEMORY_FRACTION=.4
export POLICY_CHECKPOINTING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RLA_REFERENCE=/data/lyc/projects/rla-wm/experiments/official_libero_rla_consistency/run_20261023_02
export RLA_PYTHON=/data/lyc/projects/vla-adapter-ref/.venv-cpu/bin/python
export RLA_CACHE_ROOT=/data/lyc/tmp/rla_background_invariance/cache
export CUDA_CACHE_PATH="$RLA_CACHE_ROOT/cuda" TRITON_CACHE_DIR="$RLA_CACHE_ROOT/triton"
mkdir -p "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR"
export RLA_SIDECAR="$AE_RUN/sidecars/C/clean"
export PYTHONPATH="/data/lyc/projects/LIBERO-plus:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH=/data/lyc/projects/LIBERO-plus/.libero
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LP_NUM_THREADS=2
if [[ "$POLICY_ARM" == preflight_B ]]; then export POLICY_MODEL_KIND=baseline; fi
if [[ "$POLICY_ARM" == evaluation ]]; then export POLICY_CHECKPOINTING=0 POLICY_ENTRY=ae_consistency.policy_evaluation; fi
exec bash experiments/official_libero_rla_consistency/ae_consistency/launch_policy.sh
