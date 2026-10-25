# Stove: matched low policy auxiliary weight

This is a custom LIBERO-derived development experiment, not an exact paper replication.
No Autoencoder update, physical loss, lateral-action suppression, oracle correction,
or new coefficient search is performed.

| Arm | Frozen target | Policy coefficient |
|---|---|---|
| Historical C | (z_C - mu_C) / s_C | original |
| C-low | (z_C - mu_C) / s_C | 0.1 × original |
| Historical D-center | (z_D - mu_D) / s_C | original |
| D-center-low | (z_D - mu_D) / s_C | 0.1 × original |

Original coefficient: 0.22408536585365854. Scale and centers are reused,
not refitted. Two new policies start from the accepted common ORIGINAL
policy step0, each with 600 updates, Math SDPA, BF16, microbatch8 ×
accumulation2, non-reentrant activation checkpointing, AdamW LR0.0002,
unchanged task data and common sampling order.

## Reproduction

Resolve private local input paths in prepare.py/run_plan.json to your authorized
dataset/model locations. Those files are not redistributed. Use the accepted
isolated official VLA environment and cache directories on the data volume.

1. prepare.py --out DATA_VOLUME/unique_run
2. launch.py --out ... --stage gate
3. launch.py --out ... --stage train-C-low
4. launch.py --out ... --stage train-D-center-low
5. verify_initialization.py --out ...
6. launch.py --out ... --stage evaluate
7. launch.py --out ... --stage diagnostics
8. analyze.py --out ...

Training processes have separate RNG/optimizer state; concurrent execution is
allowed only after GPU UUID/free-memory/ECC checks. Failed attempts are preserved.
No automatic backend/GPU/parameter fallback. Final600 checkpoints only.

Evaluation uses 20 frozen development physical states × three appearances ×
two new models = 120 rollouts. Historical results require matching state/XML/
language hashes and unchanged control/success rules. Cluster bootstrap resamples
physical states jointly across appearances. These are single-seed development
results, not independent final-test evidence.

Stage reconstruction uses saved states and mj_forward. Recomputed contact/goal
timing may differ from cached post-substep predicates. It NEVER replaces live
recorded success labels. Contact loss is not independently validated dropping.

Source lineage: official mlzxy/rla-wm commit
0fc6e97212442ebccffa06f1d42f5481e5f22950, its VLA-Adapter overlay, and
the previously validated stove_centering trainer/evaluator in this repository.
Existing upstream licenses and attribution remain applicable.

## Recorded recovery for this run

Both policies saved a complete step300 checkpoint before a CUDA execution
failure during validation on the originally authorized device. The user then
explicitly authorized serial recovery on GPU 0. `resume_precheck.py` validates
the complete batch without updates; `resume_run.py` restores each policy,
optimizer, scheduler, RNG and next sample position before updates301--600.
No optimizer update was replayed. The original partial validation and failure
logs are preserved; `merge_recovery.py` includes the complete repeated
validation300 instead, and `recovery_validation_comparison.py` checks the
overlapping rows. It does not remove the failed-attempt evidence.

After the recovery runner and fixed inference finish, run `analyze.py` then
`finalize.py`. The interruption delivery and completion delivery have separate
paths. `publish.py` publishes the completed report from `completion_report.md`
and the `*_completed` inventories, excluding raw trajectories and process IDs.
Do not rerun the initial orchestration to recover an interrupted run.
