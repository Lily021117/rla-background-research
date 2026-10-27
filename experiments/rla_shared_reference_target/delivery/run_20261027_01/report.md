# Shared Clean AE-C target: paired experiment

| Order | Method | Clean | Seen | Unseen | Mean |
|---|---|---|---|---|---|
| 20261101 | C | 20/20 | 20/20 | 20/20 | 100.00% |
| 20261101 | D-center | 16/20 | 18/20 | 16/20 | 83.33% |
| 20261101 | C-shared-target | 19/20 | 20/20 | 20/20 | 98.33% |
| 20261102 | C | 13/20 | 16/20 | 13/20 | 70.00% |
| 20261102 | D-center | 3/20 | 0/20 | 1/20 | 6.67% |
| 20261102 | C-shared-target | 1/20 | 2/20 | 3/20 | 10.00% |

## Primary paired effects
- 20261101: shared minus C, equal appearance: -1.67 pp, state-cluster95%CI [-5.00, +0.00].
- 20261102: shared minus C, equal appearance: -60.00 pp, state-cluster95%CI [-78.33, -40.00].
- two orders: shared minus C: -30.83 pp, state-cluster95%CI [-40.00, -20.83].
- 20261101: shared minus D-center, equal appearance: +15.00 pp, state-cluster95%CI [+3.33, +26.67].
- 20261102: shared minus D-center, equal appearance: +3.33 pp, state-cluster95%CI [-10.00, +16.67].
- two orders: shared minus D-center: +9.17 pp, state-cluster95%CI [+0.00, +18.33].

## Interpretation and limitations
- Recovery relative to D is not an improvement over C. Target equality is imposed, not evidence of learned invariance: actual policy prediction differences and errors to the fixed Clean reference are in representation_summary.csv.
- All successes/failures retained. Stage labels are frozen distance/lift proxies, not ground-truth causal failure diagnoses.
- The two orders are not independent training seeds; bootstrap clusters the same physical state jointly across orders/appearances. These20 development states have been repeatedly used in development, not an independent test.
- Only Background target appearance changes. Existing AE-C, mean/scale, auxiliary coefficient, observations, labels and deployment unchanged. AE/teacher/keep/additional losses absent.
- New runs/evaluation use GPU0; historical C used GPU1, D reference evaluated on GPU0. This execution difference is disclosed.
-1200 successful updates,120 valid new rollouts, no replay, no invalid trial, no AE/C update or sealed-set read. Historical protected hashes unchanged. No automatic extension.
