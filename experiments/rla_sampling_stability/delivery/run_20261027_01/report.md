# C sampling stability intervention

Same historical policy step0 and model training RNG for both new arms. Only balanced sampler order seed differs. This changes episode/window weights versus historical uniform-window sampling; it is not order-only versus history.

Historical sampling counts are reconstructed from frozen index files, checked against saved terminal configuration/cursor and update logs. Full per-item historical trace is unavailable; new runs record actual IDs for all600 updates.

| Group | Clean | Seen | Unseen |
|---|---|---|---|
| history_I23_T23 | 20/20 | 19/20 | 20/20 |
| history_I29_T29 | 2/20 | 2/20 | 0/20 |
| history_I30_T30 | 10/20 | 11/20 | 13/20 |
| cross_I23_T29 | 1/20 | 0/20 | 0/20 |
| cross_I29_T23 | 20/20 | 19/20 | 17/20 |
| balanced_20261101 | 20/20 | 20/20 | 20/20 |
| balanced_20261102 | 13/20 | 16/20 | 13/20 |

New order-seed equal-appearance rates: [1.0, 0.7]; report both, no winner selection.
Two order seeds, one task, reused/tuned development states. Good performance in both is preliminary stabilization evidence only. If performance remains highly variable, the intervention is insufficient; no automatic further seed/loss search.
No semantic phase oversampling. Relative-time deciles are not task-stage ground truth. Rollout stages retain historical diagnostic proxies, official success remains primary.
See paired_statistics.json, per_demo_validation_summary.csv and raw curves. History includes all relevant C conditions, not only the worst baseline.
Completed1200 updates/120 rollouts; AE/D updates0, sealed reads0, original inputs unchanged. Stop after this experiment.
