# Fixed AE / fixed half-weight low-LR continuation

Inventory-resolved six complete policy step600 checkpoints continue to1800.
This is extra training **plus** the already scheduled low-LR phase, not a pure
update-budget ablation. Restore AdamW, original MultiStepLR(last_epoch600), all
RNG and sample cursor. LR stays2e-5 at601–1800; lambda remains0.11204268292682927.
No AE, target, normalization, action, model or deployment changes.

Independent worktree based on9f2a338; all historical assets read-only. Prepare
checks checkpoint hash/state and actual prior LR, extends the exact episode-balanced
algorithm to720 episode cycles, and verifies prefix4800 against both actual traces.
No restarting pending window permutations. Each seed's methods share all14400 indices.

Training reuses the historical implementation with asserted, recorded source edits
for the bounded continuation. Before first update, full-batch outputs are compared
before/after strict step600 reload; RNG and buffers are restored. No extra gradient
diagnostics. Validation at900/1200/1500/1800 preserves training RNG. Checkpoint at600
in new directory is a verified restored snapshot, not a new training run.

Use existing analysis/VLA Python environments. Run prepare.py, publish.py ready,
then run.py with the established VLA environment. Supervisor serially executes
six1200-update continuations and600 state-paired rollouts, analyzes and publishes.
GPU UUID/ECC/26GiB checks remain. Stop on hardware errors; never silently retry.

Reuse all historical600-step rollouts without rerun. Final1800 only, no early stop
or checkpoint selection. Statistics cluster physical states across conditions,
seeds and endpoints; seed SD/range reported separately. Reused development is not
independent test. More updates and lower LR are not separately identified.
Preserve upstream licenses; publish only sanitized small summaries and source.
