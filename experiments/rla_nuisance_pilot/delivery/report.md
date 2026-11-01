# Frozen C/D new appearance exploration

Completed 900 new valid trials; reused 150 matching Clean trials. No training or backward.

## Main and secondary results

| Category | C mean | D mean | D-C | Paired-state 95% CI |
|---|---:|---:|---:|---|
| clean | 89.33% | 94.67% | 5.33 pp | [-1.33, 12.00] pp |
| background | 94.22% | 91.56% | -2.67 pp | [-7.56, 2.67] pp |
| lighting | 93.78% | 92.44% | -1.33 pp | [-6.22, 3.57] pp |

## Full per-seed outcomes

| Model | Seed | Condition | Success |
|---|---:|---|---:|
| C-low | 31001 | clean | 23/25 |
| D-low | 31001 | clean | 24/25 |
| C-low | 31001 | background_1 | 24/25 |
| D-low | 31001 | background_1 | 23/25 |
| C-low | 31001 | background_2 | 24/25 |
| D-low | 31001 | background_2 | 21/25 |
| C-low | 31001 | background_3 | 24/25 |
| D-low | 31001 | background_3 | 20/25 |
| C-low | 31001 | lighting_1 | 23/25 |
| D-low | 31001 | lighting_1 | 25/25 |
| C-low | 31001 | lighting_2 | 20/25 |
| D-low | 31001 | lighting_2 | 24/25 |
| C-low | 31001 | lighting_3 | 23/25 |
| D-low | 31001 | lighting_3 | 24/25 |
| C-low | 31002 | clean | 22/25 |
| D-low | 31002 | clean | 24/25 |
| C-low | 31002 | background_1 | 23/25 |
| D-low | 31002 | background_1 | 25/25 |
| C-low | 31002 | background_2 | 24/25 |
| D-low | 31002 | background_2 | 25/25 |
| C-low | 31002 | background_3 | 23/25 |
| D-low | 31002 | background_3 | 25/25 |
| C-low | 31002 | lighting_1 | 23/25 |
| D-low | 31002 | lighting_1 | 24/25 |
| C-low | 31002 | lighting_2 | 25/25 |
| D-low | 31002 | lighting_2 | 24/25 |
| C-low | 31002 | lighting_3 | 25/25 |
| D-low | 31002 | lighting_3 | 22/25 |
| C-low | 31003 | clean | 22/25 |
| D-low | 31003 | clean | 23/25 |
| C-low | 31003 | background_1 | 24/25 |
| D-low | 31003 | background_1 | 22/25 |
| C-low | 31003 | background_2 | 23/25 |
| D-low | 31003 | background_2 | 23/25 |
| C-low | 31003 | background_3 | 23/25 |
| D-low | 31003 | background_3 | 22/25 |
| C-low | 31003 | lighting_1 | 24/25 |
| D-low | 31003 | lighting_1 | 22/25 |
| C-low | 31003 | lighting_2 | 23/25 |
| D-low | 31003 | lighting_2 | 22/25 |
| C-low | 31003 | lighting_3 | 25/25 |
| D-low | 31003 | lighting_3 | 21/25 |

## Interpretation boundaries

Background is primary; lighting is secondary cross-intervention transfer, not an AE training intervention. Prior stove and drawer negative conclusions remain unchanged. These 25 states were previously scored: new-appearance exploration, not a new sealed test. All conditions/seeds retained. State bootstrap does not include policy-seed, AE, nuisance-sampling or task uncertainty. Per-demo offline summaries and all per-anchor/horizon values are supplied; raw latent MSE is not a cross-AE quality ranking. Lower action drift is not improvement unless absolute demonstration error is retained. No gradients or interventions were used to identify causes.

A/B: see primary/secondary effect sizes, per-seed direction and Clean cost; no isolated seed or condition establishes efficacy. C: label, predicted-latent and action measurements are separated in offline_by_demo.csv. D: no automatic next experiment; independent replication is justified only as a candidate if aggregate gains are not driven by a single seed/condition and Clean cost is explicit. Otherwise retain the existing negative archive rather than tune to this pilot.

## Same-state measurement chain

| Type / method | AE normalized MSE | Policy normalized latent MSE | Action drift xyz / rotation / gripper | Absolute action L1 xyz / rotation / gripper |
|---|---:|---:|---|---|
| background / C-low | 0.05794 | 0.00196 | 0.03331 / 0.02971 / 0.01998 | 0.17271 / 0.21282 / 0.06085 |
| background / D-low | 0.03988 | 0.00174 | 0.03198 / 0.02940 / 0.01822 | 0.18013 / 0.20804 / 0.06669 |
| lighting / C-low | 0.02444 | 0.00280 | 0.04034 / 0.04026 / 0.02632 | 0.17425 / 0.21214 / 0.06198 |
| lighting / D-low | 0.01538 | 0.00230 | 0.03507 / 0.03125 / 0.02144 | 0.17968 / 0.20944 / 0.06695 |

Background normalized label distance decreases in D: True; predicted-latent distance decreases in D: True. These are finite-sample associations, not causal mediation. Movement axes and gripper errors must be read separately; no mixed physical-unit error is used.

## Unique next-step recommendation

One priority: retain/archive the current implementation and this exploratory result; no consistent chain or independently established control benefit justifies automatic further tuning.
