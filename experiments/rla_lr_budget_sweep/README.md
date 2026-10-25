# Stove C/D-center learning-rate and budget sweep

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
