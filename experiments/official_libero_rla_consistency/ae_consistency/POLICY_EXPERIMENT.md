# Fixed C/D policy training and development evaluation

This continues the author LIBERO VLA-Adapter overlay at the versions documented
in README.md. It is a custom paired-data adaptation, not the paper's exact
recipe and not official LIBERO-Plus. Do not rerun the AE or select an AE or
policy checkpoint using development outcomes.

## Prerequisites

Use the previously accepted frozen AE-C/AE-D final600 sidecars, common original
RLA policy step0, shared target normalization, B20 sampling list, train-only
action/proprio statistics, and generated20 development states. No sealed data
is needed. All file inputs are explicitly registered by `policy_run_setup.py`.
The common auxiliary coefficient is0.22408536585365854. Do not recalibrate it.
Set `RLA_REFERENCE` to the accepted original initialization run, `RLA_PYTHON`
to the isolated VLA environment Python, `RLA_CACHE_ROOT` to a data-volume cache,
and `RLA_SIDECAR` to the frozen C/clean store for overlay configuration.
`RLA_RUN` is the newly prepared experiment directory, `POLICY_ARM` selects C/D,
and `RLA_RUN_SUBDIR` may name a separate recovery directory without changing
the target identity. The published launcher parameterizes the original local
paths; the frozen local run records retain the exact resolved paths used.

`launch_policy.sh` is a local execution template. Set data-volume cache paths,
the accepted VLA Python environment, GPU UUID and startup threshold explicitly.
The historical user authorization permits GPU1/3 above15GiB; this does not
guarantee enough memory or establish repaired hardware. GPU2 remains excluded.

## Training

`policy_training.execute` dispatches through the existing baseline bootstrap
before any original bootstrap optimizer update. It loads the common official
RLA step0, replaces only the head with the accepted official RLA head, strictly
loads every trainable parameter, and restores all common RNG states.

Use microbatch8, accumulation2, effective16 appearance samples; each8 physical
windows produces Clean and Background observations, in exact B20 order. Use
BF16, Math SDPA, AdamW2e-4/WD.01 and final600. C/D differ only in target source.
The official RLA loss wrapper is used, not a replacement loss implementation.

Memory preflight must cover forward/backward with Adam moment-sized storage
already resident, two complete effective batches, and foreach-sized workspace.
It does not execute optimizer.step. The early post-backward-only reservation
was insufficient: one C update ran before an OOM in update2. Preserve that run;
replay from the common step0 in a distinct directory and count the extra work.

The final memory setup uses non-reentrant activation checkpointing, preserving
RNG for Qwen/vision/head blocks, expandable allocator segments and cache release
between updates. Model, microbatch, effectivebatch, losses, targets, learning
rate and data order are unchanged. B20 did not use activation checkpointing;
floating-point recomputation is a disclosed implementation difference.

Save full optimizer/scheduler/RNG/position checkpoints at0/1/100/300/600.
Validation at0/100/300/600 is diagnostic only, uses eval and Inference phase,
and preserves the training RNG. Do not early stop or extend a model's budget.

## Closed loop and analysis

`policy_evaluation.execute` uses the registry's B20/C/D final600 weights. Run
only after both training acceptance files exist. Use OSMesa with the matched
LIBERO environment, not GPU rendering. Evaluate state -> appearance -> model,
all20 fixed reset seeds, Clean/seen/unseen conditions,220 steps and chunk8.
Reset to the exact state and check geometry at1e-8 for every model/condition.
Keep videos, trajectories and failures locally; stop on infrastructure errors,
retain logs and never silently retry or count them as task failures.

`analyze_policies.py` produces paired flips, exact discordant-binomial tests,
Wilson intervals for absolute SR, and20,000 physical-state-cluster bootstrap
intervals for differences and interactions. Multiple appearances of one state
are not independent samples. Failure stages are fixed simulator-label proxies,
not inferred contact truth. Single-seed/single-task development evidence cannot
establish general method effectiveness; all-success implies poor discrimination.

`policy_delivery.py` verifies saved states and protected input hashes, checks
replayed first-step logs, and exports only sanitized reports and small metrics.
Never publish model weights, input frames, sidecars, state arrays, raw rollouts,
videos, authentication files or credentials.
