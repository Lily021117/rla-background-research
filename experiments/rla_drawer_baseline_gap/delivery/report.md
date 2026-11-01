# Drawer capability and appearance-gap development study

Completed 10800 successful policy updates and 600 valid rollouts. AE/DINO updates: 0.

## Per-seed results

| Method | Seed | Appearance | Success |
|---|---:|---|---:|
| BC-standard | 31001 | clean | 18/20 |
| BC-standard | 31001 | new_background_1 | 19/20 |
| BC-standard | 31001 | new_background_2 | 19/20 |
| BC-standard | 31001 | new_background_3 | 19/20 |
| BC-standard | 31001 | new_background_4 | 19/20 |
| BC-standard | 31002 | clean | 20/20 |
| BC-standard | 31002 | new_background_1 | 20/20 |
| BC-standard | 31002 | new_background_2 | 3/20 |
| BC-standard | 31002 | new_background_3 | 19/20 |
| BC-standard | 31002 | new_background_4 | 20/20 |
| BC-standard | 31003 | clean | 18/20 |
| BC-standard | 31003 | new_background_1 | 19/20 |
| BC-standard | 31003 | new_background_2 | 8/20 |
| BC-standard | 31003 | new_background_3 | 18/20 |
| BC-standard | 31003 | new_background_4 | 18/20 |
| C-low | 31001 | clean | 18/20 |
| C-low | 31001 | new_background_1 | 19/20 |
| C-low | 31001 | new_background_2 | 7/20 |
| C-low | 31001 | new_background_3 | 14/20 |
| C-low | 31001 | new_background_4 | 16/20 |
| C-low | 31002 | clean | 20/20 |
| C-low | 31002 | new_background_1 | 20/20 |
| C-low | 31002 | new_background_2 | 19/20 |
| C-low | 31002 | new_background_3 | 20/20 |
| C-low | 31002 | new_background_4 | 20/20 |
| C-low | 31003 | clean | 16/20 |
| C-low | 31003 | new_background_1 | 18/20 |
| C-low | 31003 | new_background_2 | 20/20 |
| C-low | 31003 | new_background_3 | 20/20 |
| C-low | 31003 | new_background_4 | 20/20 |

## Decision

C qualifies without the prespecified overall gap: stop method search on this task/intervention combination.

## Interpretation

C uses frozen scratch-trained AE-C final2000 from three bowl pick/place tasks, not drawer-trained AE. BC/C architecture differs: this is not a pure auxiliary-loss causal estimate. Both use the same 20 training demos, paired appearances and seed-specific balanced indices. 1800 is a fixed limited budget, not demonstrated convergence. State-cluster intervals retain every seed/appearance but do not quantify training-seed, AE or task uncertainty. Reused development states/backgrounds are not independent tests. No D was trained. The earlier stove new-state negative result remains archived unchanged.
