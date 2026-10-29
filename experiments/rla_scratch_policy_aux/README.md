# Scratch AE downstream policy and auxiliary-weight controls

This experiment freezes the **new** official-constructor AE-C/D after 2000 updates,
not the historical released step90000 fine-tunes. Use the inventories in the run
directory to resolve exact paths/hashes. No AE/DINO update, decoder reconstruction
loss, teacher, keep loss or new consistency term is used during policy learning.

Four groups per existing balanced order (20261101, 20261102): C-fixed, D-fixed,
D-low, D-decay. All restore the corresponding historical full policy step0 and
training RNG. The only differences are AE target and specified auxiliary schedule.
Eight groups each run600 successful updates. Lambda0 is0.22408536585365854.
Decay uses the number of *completed* successful updates k, from0 through599;
the final update has zero auxiliary coefficient. Low is lambda0/2 throughout.

Normalization pools all valid training anchors, both appearances and all eight
cumulative horizons, once each: a16x64 mean, and one global **raw** standard
deviation derived from the per-coordinate mean/variance. C and D get their own
statistics. Development never enters estimation. Store shape is(Nvalid,8,16,64),
flattened to(8,1024) in the official policy collator. No invalid tail is materialized.

## Entry points

Set `SCRATCH_POLICY_ROOT` to a fresh data-disk output directory. The project paths
are resolved in `common.py`; adjust them to your local existing artifact layout.
Dependencies are the existing isolated sidecar and official-VLA environments;
no installation or library upgrade is performed. The official commit is
0fc6e97212442ebccffa06f1d42f5481e5f22950. Existing project licenses apply.

1. `prepare.py`: freeze source inventories, split, all600 weights and evaluation.
2. `launch.py extract`: sidecar environment, strict final AE loading and cache checks.
3. `launch.py preflight 20261101 preflight`: full-batch zero-update gate.
4. `orchestrate.py`: VLA environment, serial GPU0 training, Clean evaluation, gate,
   optional all-four-background evaluation, summary and sanitized synchronization.

Activation checkpointing, BF16, Math SDPA, microbatch8 x accumulation2 and the
official action/latent L1 implementations remain fixed. AdamW2e-4, weight decay.01;
MultiStepLR milestone600 called after optimizer.step, so all600 updates use2e-4.
Checkpoints0/1/100/300/600 store trainable model, optimizer, scheduler, all RNG,
sample cursor and auxiliary schedule counter. No auto-resume of unknown/latest files.

Diagnostics at0/300/600 replay the same full training batch and RNG separately for
action and raw auxiliary objectives. They use shared VLA/proprio/head-trunk parameters,
exclude dedicated readouts and position/bridge gates, intersect graph support, and
preserve parameters/buffers/RNG. Norms are unscaled gradients, not AdamW updates.

Eight final policies x20Clean states =160 trials. Only if one method reaches12/20
in **both** orders are all eight models scored on all four frozen backgrounds
(640 additional trials). No task-failure retries or protocol changes. Infrastructure
failure stops the supervisor and preserves evidence; no automatic device switch.

All results are repeatedly used development data, not independent tests. Bootstrap
clusters physical state across all appearances and orders. Two orders do not measure
complete training-seed uncertainty. Complete task success, not loss, is the main result.
