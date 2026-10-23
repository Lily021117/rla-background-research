# Official double-view RLA appearance consistency pilot

This experiment adapts the author-released LIBERO 16x64 step90000 AE, not the
paper's step40000 and not the earlier single-view ManiSkill pilot. Source RLA
commit: `0fc6e97212442ebccffa06f1d42f5481e5f22950`; official VLA upstream:
`23fa0c9c159e2aa04341cdd3e924f44061311060`.

## Environments and storage

Keep the accepted environments isolated: AE/DINO uses Torch2.8/Transformers4.57;
official VLA uses Torch2.2/Transformers4.40. Set HF_HOME, HUGGINGFACE_HUB_CACHE,
TRANSFORMERS_CACHE, TORCH_HOME, XDG_CACHE_HOME, TMPDIR, NUMBA_CACHE_DIR and
PYTHONDONTWRITEBYTECODE to explicit data-volume locations before Python.
Bind CUDA_VISIBLE_DEVICES and RLA_GPU_UUID to the same explicitly authorized UUID.
Default startup check is26GiB; this run's later user authorization permits15GiB
on GPUs1/3 through RLA_MIN_FREE_GIB=15. This is risk authorization, not evidence of
hardware repair. GPU3 was not used. A CUDA fault must stop the process; no automatic
GPU failover or unreported update replay. FifteenGiB is not a full-policy training
memory guarantee. The full batch8 resource preflight remains required next turn.

## Execution, from the code workspace

Use explicit `PROJECT`, `DATASET_RUN`, and a new `RUN_DIR` on the data volume.

1. `cache_features.py --project PROJECT --dataset DATASET_RUN --run RUN_DIR`
   extracts frozen official224px DINO tokens, FP32, fixed4-frame batches, per
   episode with mmap-able output. No all-dataset in-memory feature array.
2. `train_ae.py --project PROJECT --run RUN_DIR --arm calibrate` uses only four
   fixed training pairs to freeze600 updates, effective physical batch8, shared
   sampling and consistency scale. No development-dependent loss selection.
3. Run the same entry with `--arm C`, then `--arm D`. The official composition,
   Plucker rays, encoder/decoder forwards and FP32-master optimization utilities
   are reused. C/D differ only in the extra latent MSE coefficient.
4. In this run D failed after380 logged updates. After explicit reauthorization,
   resume was `--arm D --resume-from ORIGINAL_D/resume_step0300.pt --output-name
   AE_D_recovery_01`. The recovery path is deliberately specific to this audited
   incident, not a generic auto-resumer. It restores master/optimizer/RNG/sample
   position. Eighty recomputed updates exactly reproduced prior loss/gradient logs.
5. `evaluate_and_extract.py --project PROJECT --run RUN_DIR --mode evaluate`
   evaluates the frozen reference and matching final C/D encoder/decoder on fixed
   development pairs, including copy-current and zero/wrong-z decoder controls.
   `--mode sidecar` emits separate model/appearance stores with official anchored
   h1..8 and tail masks. All stores share C-training-only normalization statistics.
6. `raw_source_check.py` re-extracts fixed source images; `audit_sidecar_reader.py`
   checks official RlaStore joins for all25 episodes in all4 stores.
7. CPU `prepare_development.py --run RUN_DIR --dataset DATASET_RUN`, with matched
   LIBERO on PYTHONPATH and OSMesa, fixes20 task reset seeds and three conditions.
   It performs static geometry/segmentation/depth checks only; no policy rollout.
8. `prepare_policy_inputs.py --project PROJECT --run RUN_DIR` prepares the shared
   original B20 step0 reference. Invoke the existing `baseline_update_smoke.py`
   with `--experiment-entry ae_consistency.policy_readiness` and explicit RLA
   environment (`RLA_STEPS=8`, `RLA_QUERIES=16`, `RLA_DIM=64`, `LAMBDA_ACT=1`,
   `LAMBDA_RLA=0.0`, `RLA_NORM=center`, `RLA_SIDECAR=RUN_DIR/sidecars/C/clean`).
   This dispatches before the original optimizer-update smoke. It performs
   forward/backward checks only and never calls optimizer.step.
9. The first readiness attempt caught unmatched upstream random padding-token
   draws. Per-arm RNG restoration fixes this without changing the transform;
   the successful rerun uses `--output-name policy_readiness_retry1` on input
   preparation and that new directory as the bootstrap's `--run`.
10. `finish_delivery.py --run RUN_DIR` validates final artifacts. The earlier
    `audit_partial_delivery.py` was CPU-only fault-phase evidence; its BLOCKED
    file is retained. Final status is `final_delivery_verification.json`.

Actual measured results and frozen future policy/evaluation configuration are
in `delivery/`. New RLA parameters are initialized in fork_rng without changing
the shared original action parameters or initial RNG. C/D action-only gradients
match exactly. One fixed auxiliary coefficient is calibrated once on C; common
weight does not imply equal auxiliary gradient strength. The policy remains at
step0. No C/D policy training, new closed-loop rollout or held-out scoring was run.

Current D improves normalized appearance consistency but shrinks latent norms,
variance and effective rank. Do not describe this as unqualified representation
improvement, physical-consequence supervision, or action-conditioned prediction.
Do not rerun lambda to hide the tradeoff. Preserve upstream licenses in this repo.
