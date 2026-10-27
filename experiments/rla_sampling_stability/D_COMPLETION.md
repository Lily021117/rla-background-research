# Paired D-center completion

Retain both balanced C runs. Train ONLY D-center for the same two sampler order
seeds,600 updates each, then60 reused-development rollouts each. Read actual C
step0000 parameters and saved RNG, not its trained model; create fresh optimizer.
Read C's original immutable index files directly. Compare every actual consumed
sample ID row to the corresponding C trace. Frozen historical AE-D, own mu_D,
common s_C and original auxiliary coefficient; no inv03/teacher/keep or AE updates.

The configuration remains the original600 schedule, not the earlier1200-step sweep.
GPU1 is the same explicitly reauthorized device used for the matching C runs;
historical volatile parity1/aggregate2 are retained, not claimed repaired. Stop on
new ECC/CUDA/nonfinite failures, never silently change batch or device.

```
python experiments/rla_sampling_stability/d_completion.py --out <NEW_RUN> --mode prepare
python experiments/rla_sampling_stability/d_completion.py --out <NEW_RUN> --mode run
```

Preparation checks CPU step0 equality, empty optimizer, all existing C sample rows,
fixed configuration, and a few actual D-sidecar reads against source arrays and own
training mean. All original data stays read-only. Only final600 checkpoints are
evaluated. Bootstrap clusters physical states jointly across appearances AND both
orders; it does not estimate population training-seed uncertainty. History and
current comparisons remain development diagnostics, not independent tests.
