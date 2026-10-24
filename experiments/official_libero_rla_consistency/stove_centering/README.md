# Stove target-centering controlled experiment

This experiment changes only the offset applied to the same frozen AE-D
sidecars. D-old uses `(z_D - mu_C) / s_C`; D-center uses `(z_D - mu_D) / s_C`.
No AE updates, extra seed, task, or test-set access. The outcome is not assumed.

`prepare.py` freezes the training/evaluation plans and verifies the original
training-only moments. It uses the extractor's pre-FP16 training moments,
cross-checked against stored sidecars, preserving the original statistical
weighting and precision convention. It does not refit a scale.

`store.py` subclasses the official reader and changes only `offset`.
`training.py` is the accepted bounded trainer with this reader substitution,
strict-loading diagnostics and direct raw auxiliary-loss logging. Official
architecture, BF16, Math SDPA, non-reentrant checkpointing, sampling order,
loss/reduction, auxiliary coefficient, optimizer and scheduler are unchanged.
Each arm starts from the same original checkpoint and RNG, with 600 updates.
The standalone bootstrap avoids the older unrelated reference-batch read.

`gate.py` runs isolated forward/backward checks, without an optimizer update.
`diagnostics.py` separately loads fixed checkpoints 0/1/100/300/600, avoiding
any perturbation of formal training RNG or optimizer. It separates target
squared energy, residual mean/demeaned energy, raw-latent-coordinate error,
and shared action/weighted-auxiliary gradients. The trained objective is L1;
the squared-energy decomposition is only a diagnostic.

`evaluation.py` retains the historical 20-state, three-appearance protocol,
with only the model list replaced by the two current final checkpoints.
Success/failure is retained. Infrastructure failures halt, with no silent
retry. Videos and full trajectories remain local, never in this repository.

## Execution

Use the previously verified isolated VLA environment and data-disk paths.
No model weights, raw sidecars or source episodes are distributed here.
The current source uses the project's documented local path layout; replace
the project/data prefixes consistently for another installation.

```sh
python prepare.py --out "$EXPERIMENT_DIR"
python launch.py --out "$EXPERIMENT_DIR" --stage gate
python launch.py --out "$EXPERIMENT_DIR" --stage train-old
python launch.py --out "$EXPERIMENT_DIR" --stage train-center
python launch.py --out "$EXPERIMENT_DIR" --stage diagnostics
python launch.py --out "$EXPERIMENT_DIR" --stage evaluate
python rollout_timing.py --out "$EXPERIMENT_DIR"
python finalize.py --out "$EXPERIMENT_DIR"
```

Do not reuse an existing output directory or overwrite a failed attempt.
`--attempt` permits a separately named diagnostic attempt with preserved logs.
The two training processes can share an adequately provisioned authorized GPU;
their RNG/optimizer states remain process-local. Current launches require26GiB
free and verify CUDA UUID/ECC. No external process is killed or GPU reset.

Historical dependency/source licenses remain applicable. This is a custom
LIBERO-derived background-material intervention, not official LIBERO-Plus.

After a verified process/infrastructure interruption only, evaluation can be
resumed with `CENTER_RESUME_EVALUATION=1` and a new `--attempt` name. The
completed ledger must be an exact prefix of the frozen order; completed
results and their checkpoint/state/XML identities are checked and skipped.
An interrupted directory remains untouched and its retry uses a new directory.
This mechanism does not retry task failures or change the frozen protocol.
