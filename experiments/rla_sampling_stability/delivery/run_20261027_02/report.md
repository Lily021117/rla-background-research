# Balanced sampling D-center paired completion

C reused unchanged; new D-center uses matching C saved step0, exact training RNG and actual sample sequence. Historical AE-D, own mu_D and common s_C; no inv03, AE update or new coefficient.

| Order | Method | Clean | Seen | Unseen | Mean |
|---|---|---|---|---|---|
| 20261101 | C | 20/20 | 20/20 | 20/20 | 100.0% |
| 20261101 | D-center | 16/20 | 18/20 | 16/20 | 83.3% |
| 20261102 | C | 13/20 | 16/20 | 13/20 | 70.0% |
| 20261102 | D-center | 3/20 | 0/20 | 1/20 | 6.7% |

## Interpretation
- Order 20261101: D-center minus C = -16.7 percentage points.
- Order 20261102: D-center minus C = -63.3 percentage points.
- Two-order equally weighted difference: -40.0 percentage points.
- High absolute D success below C is recovery, not evidence that consistency improves control.
- If only one order favors D, the effect is order dependent. Two orders cannot establish general stability.
- Background interaction must be interpreted with absolute Clean/background rates; loss of Clean ability is not robustness improvement.
- Single task, fixed AE and initialization, reused/tuned development states. C/D target scale/representation differs; common coefficient does not equate gradient strength.
- Historical uniform-window results are references only; balanced sampling changes weights, not merely ordering.
- Completed1200 effective new updates,1300 actual successful computations including100 replayed updates, and120 new rollouts; C/AE updates0, no extra experiment or sealed data.
- GPU1 ECC guard stopped order20261102 at400. Complete model/optimizer/scheduler/RNG/sampler checkpoint300 restored on user-authorized GPU0; no hyperparameter changes. All failure logs preserved. Both D evaluations ran on GPU0; historical C evaluations ran on GPU1, a disclosed device difference.
