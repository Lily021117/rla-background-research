#!/usr/bin/env bash
# Fixed-device, fail-fast pipeline. Never retries training or task failures.
set -euo pipefail
: "${TASK_DIR:?}" "${AE_RUN:?}" "${RLA_GPU_UUID:?}"
export TASK_DIR AE_RUN RLA_GPU_UUID PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/data/lyc/tmp/rla_background_invariance
export XDG_CACHE_HOME="$TMPDIR/cache/xdg"
src=experiments/official_libero_rla_consistency
python=/data/lyc/projects/vla-adapter-ref/.venv-cpu/bin/python
for stage in evaluate sidecar rawcheck; do
  AE_STAGE="$stage" bash "$src/multitask/launch_ae.sh" > "$TASK_DIR/$stage.log" 2>&1
done
CUDA_VISIBLE_DEVICES='' "$python" "$src/ae_consistency/audit_sidecar_reader.py" --project /data/lyc/projects/rla-wm --run "$AE_RUN" > "$TASK_DIR/reader.log" 2>&1
"$python" "$src/multitask/prepare_runs.py" --project /data/lyc/projects/rla-wm --task "$TASK_DIR" --ae-run "$AE_RUN" --stage policy
for arm in preflight6 preflight_B B C D evaluation; do
  POLICY_ARM="$arm" bash "$src/multitask/launch_policy_task.sh" > "$TASK_DIR/policy_$arm.log" 2>&1
done
echo 'TASK FIXED BUDGET AND180 DEVELOPMENT TRIALS COMPLETE'
