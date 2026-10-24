# Fixed two-task extension

This extends the existing official LIBERO VLA-Adapter/RLA recipe. It does not train a dynamics model or add a physical/action-consistency loss. Main comparison: **D−C**, with C−B/D−B secondary. This is a custom LIBERO background-material development protocol, not official LIBERO-Plus.

## Frozen protocol

- Source completeness and manipulation context select an already-open-drawer bowl retrieval task and a stove bowl retrieval task. Neither is chosen using policy outcomes. All three tasks, including the historical cookie-box task, remain bowl pick-and-place tasks.
- Per task:20 training and5 development source episodes; appearance variants follow their episode. Protected identities are excluded before array reads. New derived identities retain source mappings.
- Per task:20 predeclared legal reset development states, Clean/seen/unseen; three fixed unseen palettes assigned by state ordinal modulo3. No unseen palette is used in training. Palette subsets have7/7/6 states, so palette stratification is descriptive, not an independently randomized palette comparison.
- AE-C and AE-D:600 effective updates each, common official LIBERO step90000, identical joint dual-view16×64 structure, sample order and reconstruction objective; D adds latent MSE with fixed coefficient0.009127798272072366.
- Policies B/C/D:600 updates each from original common untrained policy initialization, not trained B20. Microbatch8 appearance examples, accumulation2, Math SDPA/BF16 and activation checkpointing. C/D auxiliary coefficient0.22408536585365854; C/D share AE-C train-only sidecar normalization.
- B uses the original official action-only architecture; C/D use the same official RLA architecture. Do not claim B/C have identical heads.
-180 valid final-checkpoint development rollouts/task;360 new rollouts. No checkpoint/state/palette selection based on results. The historical cookie20-state results are reused as historical evidence, not rescored or promoted to independent test.

## Pipeline and reproducibility

The launcher paths target the authorized data-volume installation; adapt paths for another installation without changing recipe parameters. Keep all caches/output on the data volume. See the delivered frozen JSON files for source identities, version hashes and normalization.

1. `setup.py`: identity/protection audit and pre-score protocol freeze.
2. Existing `coverage_gripper/render_pool.py`: full paired state restoration/rendering; `freeze_dataset.py`: strict all-frame/view geometry and intervention checks plus train-only stats.
3. `launch_ae.sh` (`AE_STAGE=cache,C,D,evaluate,sidecar,rawcheck`): bounded frozen DINO extraction, fixed AE training and diagnostics. Optional `AE_RUN`, `AE_RESUME`, `AE_OUTPUT_NAME` select append-only recovery outputs.
4. `prepare_runs.py`: fixed sampling, budgets and common initialization. `--ae-run` selects an accepted cache version without modifying the original.
5. `finish_task.sh`: fail-fast AE evaluation/extraction/reader checks, full policy preflight, B/C/D training and final development evaluation. It never retries a task failure or automatically switches GPU after a fault.
6. `launch_policy_task.sh`: one specified task/arm/device. Independent predeclared arms may run concurrently if prechecked resource margins allow. `wait_existing.py` joins an active/completed arm and never repeats optimizer updates.
7. `audit_policy_initialization.py`: saved C/D step0 parameters and all RNG streams exact, CPU-only. Preflight also checks actual language tokens, action-only gradients and nonzero shared auxiliary gradients.
8. `analyze.py`: paired state-cluster statistics, independent within-task resampling for the fixed-three-task equal average. Three tasks do not support broad task-population inference. `finalize.py` only runs after all fixed training/trials pass; `resource_release.py` checks worker exit and ECC baselines.

## Preserved hardware interruption

The initial GPU3 run stopped after a new correctable ECC increment. Its partial drawer cache is unaccepted and preserved. Stove AE-C/D each logged44 updates before interruption; only step0 was durable. Recovery restores model/master/optimizer/RNG/sample order and replays44 logged updates per arm. `recovery_audit.py` compares the observed replay prefix, but does not claim bitwise knowledge of unsaved parameters. A possible in-flight old update remains unconfirmed. GPU3 is not reused; GPU2 remains excluded. Cache-wrapper parsing failure after a successfully persisted extraction acceptance is separately recorded, not treated as a model failure or ignored without checking receipts.

## Interpretation

Use absolute Clean and background success counts alongside D−C and interactions. Do not mistake Clean loss for improved robustness. Distance reduction can include scale/rank contraction; use normalized distances, reconstruction/copy-current references and latent ablations. Equal auxiliary coefficients do not imply equal gradient strength. Stage labels use trajectory proxies, not independent grasp/contact truth. Ceiling/floor effects, sparse flips, discrete bootstrap zero bounds, one training seed and fixed development states limit conclusions.

Next-seed work is not automatic. A full three-matched-seed replication changes both AE and policy training seeds together; policy-only reseeding tests conditional policy stability given fixed AEs. No weights, images, sidecars, raw trajectories, videos or credentials belong in this repository.
