# Task-relative geometry auxiliary supervision

Process completed; method validity requires reading all paired results.

| Order | Method | Clean | BG1 | BG2 | BG3 | BG4 |
|---|---|---|---|---|---|---|
| 20261101 | C-reference | 20/20 | 20/20 | 15/20 | 20/20 | 18/20 |
| 20261101 | Action-only-match | 0/20 | 0/20 | 0/20 | 1/20 | 2/20 |
| 20261101 | Geometry-target | 17/20 | 18/20 | 15/20 | 15/20 | 15/20 |
| 20261102 | C-reference | 13/20 | 15/20 | 10/20 | 16/20 | 12/20 |
| 20261102 | Action-only-match | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| 20261102 | Geometry-target | 10/20 | 3/20 | 3/20 | 7/20 | 6/20 |

## Scope
Geometry versus architecture-matched action-only is the primary controlled comparison. Geometry versus C also changes supervision meaning, effective output dimension and normalization. Extra simulation labels are used only for training, not deployment. This is not DBC or a demonstrated RLA algorithm improvement. No explicit invariance switch or future-only ablation was tested.

All20 development states have been reused in research. Two sampling orders are not independent full training seeds. No sealed split was read. Final600 only; no checkpoint selection by outcomes.

2400 new successful updates and400 valid new rollouts; C200 results reused. AE and teacher updates0. Stop after this run, no coefficient search.

## Paired numerical comparisons (percentage points)
- 20261101 Geometry minus Action-only-match clean: +85.00; clustered95%CI [+70.00, +100.00].
- 20261101 Geometry minus Action-only-match four_background_mean: +75.00; clustered95%CI [+61.25, +87.50].
- 20261102 Geometry minus Action-only-match clean: +50.00; clustered95%CI [+30.00, +70.00].
- 20261102 Geometry minus Action-only-match four_background_mean: +23.75; clustered95%CI [+10.00, +38.75].
- two orders Geometry minus Action-only-match four_background_mean: +49.38; clustered95%CI [+40.62, +56.88].
- 20261101 Geometry minus C-reference clean: -15.00; clustered95%CI [-30.00, +0.00].
- 20261101 Geometry minus C-reference four_background_mean: -12.50; clustered95%CI [-22.50, -3.75].
- 20261102 Geometry minus C-reference clean: -15.00; clustered95%CI [-45.00, +15.00].
- 20261102 Geometry minus C-reference four_background_mean: -42.50; clustered95%CI [-58.75, -25.00].
- two orders Geometry minus C-reference four_background_mean: -27.50; clustered95%CI [-36.25, -18.12].

## Decision boundary
These are exploratory development comparisons. No independent test or complete training-seed uncertainty is represented. An improvement over action-only without improving C is not an RLA-method advance. Single-order gains are order-dependent. No gains or harm warrants archiving this fixed target, without automatically expanding labels/losses. No follow-up is launched.
