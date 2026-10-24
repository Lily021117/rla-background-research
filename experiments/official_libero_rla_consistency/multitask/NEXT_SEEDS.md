# Optional complete-method matched-seed replication — not authorized to run now

The completed first matched seed does not establish stable method effectiveness. Negative results must remain negative; do not search for a coefficient/background/checkpoint that reverses them. The user decides whether the remaining two seeds are scientifically worthwhile after reviewing all three tasks.

## Fixed replication units

Three tasks: historical cookie-box retrieval, already-open-drawer retrieval, stove retrieval. Each retains its frozen20 training/5 offline-development source episodes and20 closed-loop development reset states. Current results remain development evidence, not sealed final test. Do not replace a low-performing task or state.

First full-method seed:20261023. Optional additional matched seeds:20261024 and20261025. For each task and each new seed:

1. Reinitialize **both AE-C and AE-D** from the same verified official LIBERO step90000 parameters. Change AE sampling/dropout RNG to the new seed, common within C/D. Keep dual-view16×64 architecture, frozen DINO features, reconstruction and fixed lambda_cons0.009127798272072366;600 successful updates each. Final600 only. No seed-specific calibration or tuning.
2. Freeze each newly trained AE and generate seed-specific clean/background anchored eight-target sidecars. Never overwrite another seed's sidecars. Recompute that seed's AE-C training-only center/global-scale statistics and share them across C/D; the normalization **rule** is fixed, while statistics follow the new AE. Reuse verified RGB/DINO caches unchanged.
3. Reinitialize policies **B/C/D from original Prismatic and original official head/LoRA initialization**, never previous trained policies. Use the new policy seed. Construct one common initialization and RNG receipt per seed; fork new RLA-module RNG deterministically (seed+1) without changing common action parameters or main RNG. C/D new modules and action-only gradients must match exactly within the accepted tolerance. B has the official baseline structural difference.
4. Use common per-task train-only action/proprio statistics, the same paired exposure rule and matched within-seed sample sequence, microbatch8×accumulation2, BF16/Math SDPA, activation checkpointing, AdamW/LR/scheduler and600 successful updates per policy. Fixed lambda_aux0.22408536585365854, no auto/per-seed recalibration. Record normalized target scales and actual auxiliary gradients; equal coefficients do not imply equal gradient strength.
5. Evaluate all final models under the same frozen20-state/three-appearance protocol:180 valid rollouts/task/seed. Failures remain valid; only documented infrastructure repairs permit the frozen exact-triplet retry rule. No checkpoint selection or additional budget based on scores.

Changing only policy seeds with one fixed AE pair is a **conditional policy-stability** study, not a complete-method replication. Do not mix those two estimands.

## Additional budget if all optional work is approved

-12 AE runs×600 =7,200 effective AE updates.
-18 policy runs×600 =10,800 effective policy updates.
-3 tasks×2 additional seeds×180 =1,080 additional valid development rollouts.
-Report actual compute/replay/failed attempts separately. Do not predict elapsed hours from nominal update counts alone.

## Analysis and stop conditions

Primary D−C per task and appearance; task-equal summary conditional on these three fixed tasks. Report each matched seed separately, plus across-seed variation, absolute Clean/background SR, paired flips and interactions. Cluster appearances by physical state and preserve the training-seed structure; repeated states across seeds are not independent new states. With only three tasks and three seeds, do not claim broad task-population generalization.

Ceiling and floor tasks limit information. Sparse-flip or degenerate bootstrap intervals do not establish equivalence or precise population effects. If an independent expanded development protocol is needed, freeze disjoint legal reset seeds and unseen material parameters before scoring, selecting only on legality—not D victories. No sealed data access.

Any nonfinite update, OOM, new ECC or CUDA fault stops the affected task, preserves complete model/optimizer/scheduler/scaler-if-used/RNG/sample-position state and records replay. No automatic fault-device reuse, parameter changes, second-seed escalation or extra loss. No physical, action-consistency, contrastive or world-model objective is introduced by this plan.
