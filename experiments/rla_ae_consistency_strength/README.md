# D-inv03: stove AE consistency-strength experiment

This is a single-seed development experiment, not an independent test result.
The only intended intervention relative to D-center is `lambda_inv * 0.3`.
The new encoder's targets and training-only center are downstream consequences.
Policy learning rate, target scale, auxiliary coefficient and deployment remain fixed.

## Frozen computation

- AE: common official LIBERO step90000 initialization; paired reconstruction
  plus `0.0027383394816217098 * mean((z_clean-z_bg)^2)`; 600 updates.
- Policy: original common step0; AdamW base LR `2e-4`; fixed auxiliary weight
  `0.22408536585365854`; 600 updates. The scheduler is called **after** the
  optimizer, with milestone 600. Thus all 600 updates use `2e-4`; the saved
  optimizer's next LR is `2e-5`.
- Target: `(z_new - mu_new_train) / s_C`. Internal store name `D` is an alias
  local to this new run, not a reference to the old AE-D checkpoint.
- Evaluation: final policy only, 20 frozen development states and three
  appearances, 60 trials. No teacher, keep loss, sealed tests or automatic sweep.

## Entry points

`prepare.py` freezes plans and checks protected sources. `launch.py --stage ae`
starts the single AE run. `continue_pipeline.py` waits for successful AE
completion and runs representation diagnostics, sidecar extraction, source-image
checks, `target_ready.py`, original `stove_centering.training`, the frozen
evaluation entry, and `analyze.py`. Each stage stops on failure; no automatic
hyperparameter changes or retries are performed.

These scripts use the project's existing local artifact tree and isolated
environments. Configure paths for your installation before execution. Weights,
images, sidecars and simulator states are intentionally not distributed here;
their hashes and provenance are recorded in the sanitized delivery. Required
model/data downloads and licenses remain those of the original project.

`publish.py` copies only allowlisted reports, configuration, small metrics and
plots, replaces local data paths/UUIDs, and rejects oversized or credential-like
payloads. Git commit/push and remote verification are performed separately.

Historical C, D-center, and low-LR results are retained. Device/recovery history
is disclosed rather than claiming bitwise replication across hardware. Missing
historical telemetry is marked unavailable, not interpreted as a negative event.
