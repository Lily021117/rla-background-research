# Fixed-compute B4/B20 coverage experiment

This extends the accepted derived LIBERO dataset loader, not the RLA encoder.
No RLA targets or future images enter the baseline. Keep upstream licenses.

## Frozen design

`prepare_paired_training.py --dataset DATASET_RUN --out NEW_RUN` freezes B4 as
the original four training episodes, B20 as the accepted twenty, and the same
five development episodes. It checks subset/split identities and writes the
state-paired 20-trial protocol before training. Both arms share the twenty-train
Q99 statistics; this gives B4 aggregate statistical exposure to that pool.

Each arm has 600 successful updates, batch8/accumulation2, 4800 physical-window
exposures and 9600 appearance observations. Checkpoints/evaluation: 0,100,300,600.
Repeat a fixed-seed permutation of each arm's eligible windows; algorithms and
seeds match, but actual sample sequences cannot match different datasets.
Final600 alone is used for rollout. This is fixed compute, not convergence.

## Commands

Use the accepted dependency environment. Before Python, set all cache and temp
variables to the data volume, set PYTHONDONTWRITEBYTECODE=1, and bind both
CUDA_VISIBLE_DEVICES and RLA_GPU_UUID to the authorized, prechecked UUID.

From the parent code workspace:

```bash
python experiments/official_libero_rla_consistency/baseline_update_smoke.py \
  --reference REFERENCE_RUN --run NEW_RUN/B4 --mode formal --batch-size 8 \
  --experiment-entry coverage_gripper.paired_training
# Repeat for B20, with identical initialization and algorithm.
```

`evaluation_state_preflight.py --run NEW_RUN` first verifies all five saved
initial states under clean/background XML, with no policy actions or dynamics.
For the actual matched LIBERO environment, set LIBERO_CONFIG_PATH to a data-volume
config, PYTHONPATH to the matched LIBERO and official VLA trees, MUJOCO_GL=osmesa,
PYOPENGL_PLATFORM=osmesa, and NUMBA_CACHE_DIR to the data volume. After both
training acceptances exist:

```bash
python experiments/official_libero_rla_consistency/baseline_update_smoke.py \
  --reference REFERENCE_RUN --run NEW_RUN/B4 --mode development --batch-size 1 \
  --experiment-entry coverage_gripper.paired_rollout
python experiments/official_libero_rla_consistency/coverage_gripper/summarize_paired_training.py \
  --run NEW_RUN
```

The rollout runner strictly reloads final600 for every state-paired trial,
restores trainable-key declarations before each frozen reload, executes chunk8
with the unchanged official postprocessing, and stops on an invalid trial.
There are no retries or state substitutions. Development is not independent test.

An OOM, nonfinite quantity, ECC increment, or provenance mismatch stops the arm.
Do not automatically retry, switch devices, replay updates, extend the budget,
or launch downstream C/D. Preserve checkpoints and failure evidence.

Detailed local outputs include sampling identities, trainable deltas plus
optimizer/scheduler/RNG, per-chunk offline predictions and errors, closure timing,
videos and saved trajectories. Publish source/configs/sanitized summaries only.
