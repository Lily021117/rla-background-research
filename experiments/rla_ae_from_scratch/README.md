# Untrained AE-C/D feasibility experiment

This is **not** the historical step90000 + 600-update fine-tune. Both AE modules
are constructed with the official LIBERO dual-view 16x64 configuration. Loading
any checkpoint during construction is explicitly forbidden. Only our newly
saved common initialization is loaded for the two formal arms. The DINO feature
cache comes from the previously accepted frozen pretrained DINO pipeline.

## Frozen protocol

- Three previously accepted tasks: cookie-box bowl, open-drawer bowl, stove bowl.
  Each retains 20 train and 5 exploratory development episodes. New middle-drawer
  operation data is inventoried but excluded from this cached-pipeline trial.
- C: mean of clean/background future-feature L1+MSE reconstruction.
- D: same reconstruction plus raw paired latent MSE.
- Decoder output zero initialization is preserved. Reconstruction gradient to
  the encoder at step0 is expected to be zero; it is not used to calibrate lambda.
- One isolated 60-update reconstruction-only pilot. At pilot10/30/60, the same
  six training transitions supply encoder reconstruction and consistency gradient
  norms. Freeze lambda using the median 10%-gradient-ratio candidate, capped by
  the smallest 10%-loss-ratio candidate. No development-based calibration.
- Formal C and D restart the common **untrained** step0/RNG, each2000 updates.
  Independent sampler RNG produces an identical frozen index file; batch8 paired
  transitions, micro1, accumulation8. D consistency starts at formal update1.
- Official mixed-FP16 structure, FP32 master weights, fixed loss scale256,
  Math SDPA, AdamW1e-4/WD0, no scheduler. No clipping or auxiliary objectives.
- Checkpoints/validation0,50,250,500,1000,1500,2000. Additional250-period states
  are recovery only; no best-validation selection. Six-hour per-process cap.
- Eval real/zero/shuffled latent. Shuffling is a fixed bijection across episodes
  of the same task/split/horizon. No donor from train is used for dev.
- Paired uncertainty uses development episode clusters, not camera/horizon rows
  as independent episodes. Does not include seed or task population uncertainty.

## Execution

`launch.py prepare init pilot` then, once frozen_plan exists, `launch.py C D`.
All subprocess logs are exclusive-created and failed attempts are preserved.
Do not rerun completed stages. A future explicit resume must use model, FP32
master, Adam state, RNG and sample cursor from one complete `step*.pt`; never use
weight-only restart. `reload_check` strictly validates the saved parameter hash,
optimizer step and RNG at the checkpoint boundaries.

`summarize.py` runs only after both formal completion files exist. It reads logs,
generates curves and episode statistics, and rechecks protected manifests.

No policy training, sidecar regeneration, simulator, rollout, held-out access,
download or new rendering. No public upload of caches or checkpoints. CPU caches
and all newly created files are confined to the data disk.
