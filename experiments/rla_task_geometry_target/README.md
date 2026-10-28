# Task-relative geometry supervision: fixed feasibility experiment

Base: `8cb44ab99194c812837ee862f2dc33fe5c0f1de2`.
This is a simulator-label reference experiment, not DBC or an established RLA improvement.

## Frozen design

- Stove bowl-to-plate task only; existing20 training/5 development demonstrations.
- Existing balanced index orders20261101/20261102; historical C complete step0 and RNG per order.
- Architecture-matched action-only and geometry-target, each600 successful updates.
- C-reference final600 and its200 matched results are reused, not retrained.
- No AE, teacher, keep, explicit invariance loss, gradient surgery or action correction.

The world-frame metre-valued relation is `[bowl-eef, bowl-plate]`.
For each verified cumulative horizon1..8 the target is `[r(t), r(t+h)-r(t)]`.
These horizon indices are confirmed by the actual historical extractor and its
sidecar audit, not inferred from output shape. Only complete eligible windows are
used. Statistics are population mean/std over each unique training physical
window and its8 valid horizons, once per physical example, std floor0.001m.
No appearance duplication, exposure weighting, validation fitting or clipping.

The target is read from the **first12 coordinates of the existing final independent
8x1024 auxiliary output**. This is the first query's first12 channels in the packed
16x64 layout; no token averaging, new head, padding loss or latent-label repetition.
The auxiliary readout is downstream of the shared trunk and is not fed back into
the action path. Architecture and parameters remain unchanged.

Geometry loss is FP32 mean absolute error over valid12-coordinate/horizon entries.
Original action loss and BF16 forward remain unchanged. Lambda is the historical
fixed0.22408536585365854, and exactly0 for action-only. No original RLA loss is
computed. Geometry versus C also changes supervision meaning, effective dimension
and normalization, so it is not a pure single-representation-factor comparison.

## Execution

Use the existing isolated VLA environment; put all caches under the data disk.
Set `RLA_PROJECT_ROOT` to the original artifact project and `RLA_GPU_UUID` to the
explicitly authorized healthy device. UUID/ECC/free-memory checks remain required.
`labels.py` sets saved simulator state and calls only `mj_forward`, never `mj_step`.
It verifies state/proprio/action identity against each saved image record.
`run.py --mode prepare` freezes configuration and references. `--mode run` executes
the two gates, four training groups and the fixed400-trial evaluation, serially on
the authorized GPU UUID, stopping on errors. `analyze.py` requires complete results.

```bash
python labels.py --manifest "$MANIFEST" --out "$OUT/labels"
python run.py --out "$OUT" --mode prepare
python run.py --out "$OUT" --mode run
python analyze.py --out "$OUT"
```

Gates use real effective batch16/microbatch8 with matched RNG, original-C-zero-aux
comparison, target replacement and finite shared geometry gradients. No gate
optimizer update. Training uses AdamW LR2e-4,weight decay0.01, original MultiStepLR
milestone600 after update; checkpoints0/1/100/300/600, validation0/100/300/600.
All600 actual sample-ID rows must match the corresponding historical C.

Evaluation: original20 development states x Clean+four frozen new backgrounds x
four new models=400. Final600 only, no failure retry or state selection. Simulator
truth is used only for ordinary evaluation success/metrics, never policy input.
Statistics resample physical state jointly across appearances and both orders.
Development has been reused; two order seeds do not establish independent test
generalization or full training-seed stability.

Never publish weights, label caches, source states, images or videos.
