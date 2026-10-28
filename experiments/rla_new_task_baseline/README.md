# New-task baseline admission

Branch: `experiments/rla-new-task-baseline`, based on
`0d8a04c25ebc07b821f52647e020decbc30ec103`.

Current status: **BLOCKED at source-data admission**, not a training result.
See `delivery/run_20261028_01/report.md`. No policy/AE training or policy rollout
was run. These scripts do not substitute regenerated model XML for a proven
source model.

Use the existing isolated environment with h5py, NumPy, Pillow, MuJoCo and the
already accepted local LIBERO/robosuite bridge. Do not upgrade dependencies.
Set `CUDA_VISIBLE_DEVICES=''`, `PYTHONDONTWRITEBYTECODE=1`,
`MUJOCO_GL=osmesa`, `PYOPENGL_PLATFORM=osmesa`, the local `LIBERO_CONFIG_PATH`
and `PYTHONPATH`. Set cache/TMP paths to the data volume. No GPU is required.

```bash
python experiments/rla_new_task_baseline/audit_source.py \
  --source <SOURCE_TASK_HDF5> --bddl <LOCAL_TASK_BDDL> \
  --out <NEW_RUN>/source_audit
python experiments/rla_new_task_baseline/finalize_admission.py \
  --out <NEW_RUN> --project <PROJECT_ROOT> --data-root <DATA_ROOT>
```

Use fresh output directories; existing audit results must not be overwritten.
Do not run the finalizer if new evidence changes the admission outcome; its
report explicitly describes this run's unresolved source-model/first-frame
evidence. Audit creates only CPU reset/state-setting/render checks, never
physics stepping. Original first-frame inconsistency is supported by the
local regeneration code, not inferred by correlation matching.

The sanitized delivery excludes model XML, source/derived images, raw states,
weights, videos and supervision caches. Public official encoder provenance is
reused and hash checked; task-specific pretraining exclusion is not established.
No new methods, tasks, seeds or hyperparameters were introduced.
