# Stove C/D-center learning-rate and budget sweep

## Active revision: short200 and two-stage offline curves

The user paused the1200-update/960-rollout search. `short_run.py` now runs the
remaining seven candidates to200 updates and performs frozen offline diagnostics
for all eight. Cx0.1 had already logged416 successful updates when stopped; full
checkpoints through400 and all logs are preserved. Its200 checkpoint is reused,
and absent50 checkpoint remains missing. No backward/optimizer replay is used to
fill this gap. Sixteen logged updates after400 were not checkpointed; the possible
in-flight417 attempt is not counted as a confirmed success.

The original1200 milestone remains. The active run has no rollout stage and no
teacher/keep logic. Run `short_prepare.py --source LONG_OUTPUT --out NEW_OUTPUT`,
then `short_run.py --out NEW_OUTPUT`. `ae_review.py` only reads historical logs.
`short_diagnostics.py` compares identical fixed train/development observations,
separate action/weighted-aux gradients on the accepted shared parameter scope,
and actual parameter changes. No diagnostic optimizer is instantiated. Existing
state-derived before/after-lift labels do not supply precise four-stage labels.
`short_analyze.py` exports raw/source curves, per-demo/dimension/horizon summaries,
plots and actual computation counts. The original search below is retained for
provenance and is not the active execution authorization.

Run training and diagnostics serially. An attempted overlap encountered an OOM
when another process occupied about35.6GiB; that diagnostic performed no optimizer
update or completed backward. Failed logs remain intact. `diagnostic_registry.json`
can select a separately named retry directory; the Cx1 retry keeps the same frozen
checkpoint, batch and objective. Do not rerun the optional diagnostic pipeline
concurrently with training on a shared GPU.

This is a policy-only development hyperparameter search, not independent testing.
The frozen AE/sidecars and accepted official LIBERO VLA-RLA interface are reused.
C targets `(z_C-mu_C)/s_C`; D-center targets `(z_D-mu_D)/s_C`. All eight candidates
start from the original common Prismatic-derived policy step0, not trained C/D/E.
No teacher, keep loss, AE update, action correction or new supervision is used.

The actual historical optimizer is read from C step1: one AdamW group, LR2e-4,
weight decay0.01, betas(0.9,0.999), eps1e-8. Every group's LR is multiplied by
0.1/0.3/1/3. MultiStepLR keeps gamma0.1 with the milestone moved from600 to1200,
no warmup; the drop occurs after the final update. Thus m1's first600 update LRs
match history. Initialization and input/loss equivalence are separately checked;
do not infer bitwise historical reproduction from LR equality alone.

The accepted4800 physical-window sampling order is repeated twice, identically
for all candidates. Each update sees8 physical windows in clean/background,
effective16 appearances, micro8 with accumulation2, Math SDPA/BF16, activation
checkpointing. Each candidate receives1200 updates,9600 physical-window exposures,
19200 appearance exposures,38400 two-camera image exposures; this is not9600
independent transitions. Source episodes remain20 train/5 development.

All16 predeclared checkpoints (600/1200 for eight runs) are evaluated on the same
20 reused development states and clean/seen/unseen materials,960 trials total.
Rank separately per method by equal appearance-average success, then Clean,
fewer updates, smaller multiplier. These selected scores/CIs are tuning results,
not independent confirmation. Preserve physical-state pairing across appearances.
Future matching seeds20261026/20261027 are proposed, not executed automatically.

Use the accepted isolated VLA environment and data-volume cache paths in launch.py.
Run prepare.py with a new output directory, then run.py --out OUTPUT --gpu1.
The launcher checks authorized UUID, free memory and historical ECC counters.
No fallback or device reset is allowed; faults retain all logs/checkpoints.
Recovery restores model,optimizer,scheduler,RNG,next sample and effective update
count, recording any replay separately. Failed numerical runs are not replaced
with a different LR. Final600/1200 paths are fixed before evaluation.

Core bootstrap/forward/dataset/evaluation lineage is mlzxy/rla-wm official
commit0fc6e97212442ebccffa06f1d42f5481e5f22950 and the accepted repository's
official LIBERO adapters. Reused control_preserving_rla.Interface supplies only
the official batch/forward/checkpointing helper, never the teacher initialization
or keep-loss execution. Existing source attribution and licenses are retained.
