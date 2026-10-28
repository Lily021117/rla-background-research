# Frozen auxiliary-target / action-gradient diagnostic

Base commit: `b404cde06d6f3f66524e685a68113b089dc9ecd9`.
This bounded diagnostic preserves the decision to archive the present AE-consistency
and shared-Clean-target implementations as negative development results.

## Reproduction

Set an exclusive `OUT` directory on the data disk. Existing artifacts are opened
read-only; outputs use exclusive creation. Paths in the preparation script point to
the documented local experiment layout and must be mapped to your own artifact root.

```bash
python experiments/rla_aux_interference_diagnostic/prepare.py "$OUT"
# Use the accepted official model environment for preparation and launch:
python experiments/rla_aux_interference_diagnostic/launch.py "$OUT" diagnostic
# Use an analysis environment with numpy, pandas and matplotlib:
python experiments/rla_aux_interference_diagnostic/analyze.py "$OUT" diagnostic
```

The launcher checks the authorized GPU UUID, available memory and ECC baseline.
It retains the accepted official Prismatic/bootstrap, BF16, Math SDPA,
non-reentrant activation checkpointing, microbatch8 and accumulation2.
No optimizer or scheduler step is permitted. No policy rollout is present.

The delivered run is **PARTIAL**. An initial implementation included action-only
`layer_norm1/fc1` in its aggregate. The attempted source is preserved locally as
`worker_attempt1.py`; original VLM/proprio statistics remain valid, while mixed
head aggregates are explicitly superseded. `amend_scope.py` records the correction
and the remaining-budget selection. The current worker expects the delivered
`corrected_snapshot_manifest.json`: four final snapshots on the same first seven
batches and common step0 on two batches. Map its artifact paths before reproducing.
Do not silently describe this as a completed13×8run.

## Executed computation path

`rla_lr_budget_sweep/bootstrap.py` loads the original pretrained backbone and
constructs the official RLA head. `stove_centering.training.initialize` restores
the common saved step0; each selected frozen snapshot is loaded strictly using
`control_preserving_rla.common.load_parameters`.

`coverage_gripper.dataset.DerivedEpisodeDataset.sample` creates the existing
dual-camera/language/FK-proprio/action batch. `CenterStore` reads the frozen AE-C
sidecar with the original C offset and shared scale. Appearance-specific and
shared-Clean modes change only `batch['rla']`.

`rla.loss.make_run_forward_pass` wraps the actual official finetune action forward.
The diagnostic captures its returned action loss and its actual `nn.L1Loss`
auxiliary output using a module hook. It does not substitute an approximate loss.
Target lookup takes place outside the policy forward and targets require no gradient.
The official policy forward consumes current images, language, proprio and action
token masks; continuous action labels and RLA targets are supervision only.

The RLA head's action queries can attend to predicted RLA-query features. This is
not a path from the ground-truth target into action inference. The frozen-parameter
target-switch output and action-gradient equivalence gate checks this distinction.

## Budget and numerical interpretation

13 parameter snapshots ×8historical effective batches ×3objectives =312full
effective-batch gradient extractions maximum. Two acceptance units add at most6
extractions. Each extraction uses two microbatch forwards/backwards, preserving
the training accumulation divisor and BF16 gradient accumulation.

Corrected shared-gradient scope: trainable VLM/LoRA/action queries, proprio
projector and shared MLP blocks. Dedicated final action/auxiliary readouts and
the action-only `layer_norm1/fc1` are excluded. Tensor-level dependencies and unused parameters are logged;
zero-valued gradients are not dropped from norms. This does not claim an
elementwise architectural disentanglement of query-specific rows inside a shared
parameter tensor.

All Python/NumPy/CPU/CUDA random states are replayed before each objective,
including token padding, training noise and checkpoint recomputation. Action
equivalence is expected bitwise. Total-gradient decomposition uses a predeclared
2% relative-L2 ceiling because BF16 addition/backpropagation order can differ;
the observed error is reported rather than replaced by the ceiling.

No gradient arrays, model weights, images, sidecars or credentials belong in the
published delivery. Reported gradients are raw local derivatives, not AdamW update
directions or proof of a long-term control mechanism. Batches, checkpoints and
appearances are not independent training seeds.
