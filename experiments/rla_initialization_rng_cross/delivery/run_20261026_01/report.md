# C initialization / training randomness crossover

Two new crossover cells; historical diagonal cells reused. All results are reused-development diagnostics, not independent tests. Initial parameters and saved RNG/order were crossed, not regenerated. Other hyperparameters unchanged.

| Cell | Clean | Seen | Unseen |
|---|---|---|---|
| I23_T23 | 20/20 | 19/20 | 20/20 |
| I29_T29 | 2/20 | 2/20 | 0/20 |
| I23_T29 | 1/20 | 0/20 | 0/20 |
| I29_T23 | 20/20 | 19/20 | 17/20 |

Effects are new minus historical level, conditional on the other factor. Negative initialization effects indicate the new initialization is worse at that fixed training RNG. Interaction means the two effects need not be additive.

{
  "init_effect_training23": {
    "difference": -0.05,
    "state_cluster_CI95": [
      -0.11666666666666667,
      0.016666666666666666
    ],
    "gains": 1,
    "losses": 4
  },
  "init_effect_training29": {
    "difference": 0.05,
    "state_cluster_CI95": [
      0.0,
      0.13333333333333333
    ],
    "gains": 3,
    "losses": 0
  },
  "training_effect_init23": {
    "difference": -0.9666666666666667,
    "state_cluster_CI95": [
      -1.0,
      -0.9166666666666666
    ],
    "gains": 0,
    "losses": 58
  },
  "training_effect_init29": {
    "difference": -0.8666666666666667,
    "state_cluster_CI95": [
      -0.9666666666666667,
      -0.75
    ],
    "gains": 0,
    "losses": 52
  },
  "interaction": {
    "difference": 0.1,
    "state_cluster_CI95": [
      0.0,
      0.21666666666666667
    ]
  }
}

Only two factor levels, one task and previously tuned development states. No unique-cause claim; hardware/time differences in historical cells remain a limitation. No AE updates, loss changes or post-result extension.
