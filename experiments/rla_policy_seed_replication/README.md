# Fixed-AE policy seed replication

Two preassigned seeds, 20261029 and 20261030. Each runs C and D-center for
600 successful updates from a shared zero-update initialization. No AE changes,
loss search, teacher, keep loss, low LR, or D-inv03 targets.

The selected scope preserves the original common action-path initialization.
Only the auxiliary output readout (`layer_norm2_rla`, `fc2_rla`), training RNG
and seeded permutation order vary with seed. The seed gate verifies that the
readout reset does not alter initial actions. Both methods load the same seed
checkpoint. This is conditional training-randomness replication, not a test
of every possible action-head/LoRA initialization or AE-training seed.

Each method uses the historical original target and its own accepted training
mean, with unchanged common scale and auxiliary coefficient. The training loop
is reused from `stove_centering.training`; the only added routing chooses the
frozen C or D sidecar. The original AdamW, Math SDPA/BF16, activation checkpointing,
batch, STOP/masks and scheduler order are retained. All 600 updates use 2e-4;
the scheduler changes the next LR after the final update.

`prepare.py` freezes plans and sample orders. `target_audit.py` verifies source
identity, masks and actual reader normalization without generating targets.
`pipeline.py` runs the zero-update gates, four training jobs, and two paired
120-trial evaluations, then `analyze.py`. Every stage stops on a fault; recovery
requires an explicit complete-state audit, not automatic parameter changes.

All appearances for a physical state stay together in bootstrap resampling.
The historical seed and two new seeds are reported individually. A three-seed
mean interval conditioned on these runs is not a precise estimate of training
seed uncertainty. The same development states have already been used for
exploration; no independent test or full-method seed-stability claim is made.

Scripts expect the existing project artifact tree and isolated environments.
Adapt local paths when reproducing. Weights, data, videos and raw trajectories
are not distributed in this repository; see sanitized hashes and provenance.
