# B-only single-task pilot

This is a derived-data learnability experiment, not a full official LIBERO
reproduction. It has no RLA auxiliary supervision or future-image target.

Immutable references: RLA `0fc6e97212442ebccffa06f1d42f5481e5f22950`, VLA-Adapter
`23fa0c9c159e2aa04341cdd3e924f44061311060`. Accepted environment: PyTorch 2.2,
Transformers 4.40.1, PEFT 0.11.1, timm 0.9.10, TensorFlow 2.15 and matched
MuJoCo/robosuite/LIBERO. Math SDPA is explicitly selected; no environment upgrade.
Use the prior verified Prismatic initialization inventory, not Spatial-Pro.

## Execution order

All paths below are placeholders for local data-volume paths. Set HF_HOME,
HF_HUB_CACHE, TRANSFORMERS_CACHE, TORCH_HOME, XDG_CACHE_HOME, TMPDIR and
NUMBA_CACHE_DIR explicitly. Set PYTHONDONTWRITEBYTECODE=1. Never open held-out
episode contents to fill a data shortfall.

1. `single_task_preflight.py --project PROJECT_ROOT --out RUN_DIR`
2. `render_admitted.py --project PROJECT_ROOT --out RUN_DIR`
3. `accept_rendered_pool.py --run RUN_DIR`
4. After UUID/ECC/free-memory checks, run `baseline_update_smoke.py --reference
   REFERENCE_RUN --run RUN_DIR`, then separately `--batch-size 8`. These perform
   isolated updates and must never supply formal initialization weights.
5. Review and freeze single_task_training_plan.json as the run's training_plan.json.
   Run the same entry with `--mode formal --batch-size 8` from fresh Prismatic.
6. After acceptance, with the matched LIBERO environment on PYTHONPATH, OSMesa,
   and explicit LIBERO_CONFIG_PATH, run `--mode development --batch-size 1`.
   Exactly two frozen trials, no retries or model selection.
7. CPU accounting: `regional_sequence_audit.py --run RUN_DIR`, then
   `analyze_development.py --run RUN_DIR`.
8. Delivery: `finish_single_task.py --run RUN_DIR --reference REFERENCE_RUN
   --code CODE_DIR`. Existing output directories are not reused.

GPU processes must set CUDA_VISIBLE_DEVICES and RLA_GPU_UUID to the same
authorized UUID. CPU rendering must expose no GPU. The executed formal run used
270 updates, batch8, accumulation2, with only four independent training demos.

## Checkpoints and deviations

formal_B/step*.pt stores trainable deltas, optimizer/scheduler, RNG, global step
and sampling position. It is not a standalone VLA: strictly assemble with the
same immutable Prismatic base and official modules, verifying trainable keys.
No automatic resume/retry/GPU failover/follow-on C/D training is implemented.
Audit persisted updates before any later resume; never silently replay updates.

The adapter uses official normalization, chunking, RLDSBatchTransform, collator
and run_forward_pass. Its outer NPZ episode sampler is custom, not the untouched
TFDS launcher. STOP repair, training-only statistics, paired material augmentation,
small data/budget and Math SDPA are explicit deviations. Upstream random image
augmentation is not additionally enabled.

No checkpoints, raw images, trajectories, videos or credentials are committed.
Existing upstream licenses and provenance remain applicable.
