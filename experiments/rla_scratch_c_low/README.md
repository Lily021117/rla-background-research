# C-low completion — no changes to the completed eight-arm experiment

Only two new policies: order20261101 and20261102, C-low,600 successful updates each.
The frozen new AE-C targets/statistics and original complete policy step0/RNG are
read directly from the preceding experiment. Lambda is exactly half of
0.22408536585365854. No AE update, coefficient search or historical trial rerun.

`training.py` asserts and records a single target-dispatch source replacement in
the unchanged historical training function, binding only the new constant-half
weight function. A guard verifies parameter and all RNG hashes against C-fixed
before the first update. The underlying model, action/RLA losses, optimizer,
scheduler, batch, activation checkpointing, precision, preprocessing and sample
trace checks are otherwise unchanged. Historical source files are never edited.

`evaluation.py` reuses the exact deployment, changing only the accepted trial
budget to200 (two models x20 states xClean/four backgrounds). No conditional gate
is applied to this explicitly authorized completion. All previous valid trials
are read-only references. State/XML/language matching is asserted in analysis.

Run `prepare.py` with the data disk paths in `common.py`, then `run.py` in the
existing official VLA environment. The supervisor uses only authorized GPU0,
requires26GiB free and no uncorrectableECC, stops on any error, and does not retry
or change devices automatically. Full-state checkpoints preserve recovery state.

Analysis shows each order separately plus equal-order, state-clustered paired
statistics for C-low minus C-fixed and D-low minus C-low. These are repeatedly
used development states, not independent tests or complete training seeds.
Only source/configuration/sanitized small summaries are committed; no weights,
labels, videos or raw trajectories. Original licenses and sources are preserved.
