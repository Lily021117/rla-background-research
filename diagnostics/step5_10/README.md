# Step 5.10 — frozen first-update component substitution

The branch descends from `diagnostics/step5-9`, commit `c3e0dde39d362b76b21c5d6f8619de4a95d63f66`. Source-model code is the accepted VLA worktree at `23fa0c9c159e2aa04341cdd3e924f44061311060`, plus the exact unified source captured in the prior snapshot.

`prepare.py` verifies B step0/step1, freezes the exact14 Step5.9 examples and reads existing optimizer state. `swap.py` runs all8 VLM/head/proprio combinations in memory. It keeps the auxiliary head at step0 and verifies its output cannot affect actions. `analyze.py` writes demonstration-equal metrics and a proposal; it does not train. No script creates an optimizer, calls backward, steps a simulator, or saves a model.

## Inputs and execution

Use the previously accepted Python environment; see Step5.9 `runtime_versions.json`. Supply the external project layout, original checkpoints, sample manifests and cached real policy batches. Models/data are not part of this repository. The needed historical paths are explicit relative to `RLA_PROJECT_ROOT` in `common.py`.

Set `RLA_PROJECT_ROOT`, a new empty `RUN_DIR`, isolated data-disk `RLA_CACHE_ROOT`, `DIAGNOSTIC_PYTHON`, the approved `DIAGNOSTIC_GPU_UUID`, `PYTHONPATH` for the accepted VLA worktree and LIBERO library, and `LIBERO_CONFIG_PATH`. Localize the redacted `${GPU_UUID}` assertion in `swap.py` to the approved device before running. It is an intentional hardware guard, not an automatically expanded Python variable. Run `bash diagnostics/step5_10/run.sh` from this repository. There is no model training command in this workflow.

The actual run used an already-authorized GPU0 after GPU1 failed the26GiB free-memory gate. Math SDPA, eval/inference mode, fixed seed20261021, single-input batch and published action/proprio normalization stayed fixed. Stop token2 was added to the diagnostic training-forward batch; deployment uses the same token. Every combination matched these two paths exactly.

## Metric interpretation

The0.155 drift is mean absolute change over14 appearances ×8 valid chunk positions ×7 normalized action dimensions. These are7 physical windows from7 demonstrations (4 training,3 validation), each rendered in two appearances and observed through two current cameras. It is not a metric in metres or control success.

Report normalized action error and drift separately. Decoded translation/rotation command coordinates are dimensionless. After controller clipping/scaling, translation and rotation-vector command errors use m and rad, separately; these are commanded increments, not measured motion. Gripper command error is separate. The fixed14 targets are allopen, so zero binary-gripper error cannot validate grasp/release timing.

`000/100/010/001/110/101/011/111` means VLM/head/proprio source0 or1. Use fixed final combinations; no checkpoint selection, new states or rollout is involved. The proposed10-update head-LR control is a document only and requires a separate training task.

All shared paths/device IDs are redacted. See local `delivery_verification.json` for historical protection and `source_manifest_step5_10.json` for hashes of executed versus shared scripts. Raw output arrays and all weights/data remain local.
