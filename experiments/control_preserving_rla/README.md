# Control-preserving RLA policy post-training

Four students share trained stove C parameters. Only the independent RLA readout
(`model.layer_norm2_rla`, `model.fc2_rla`) is reset under a common initialization.
The RLA token trunk/gates are shared with the action path and remain loaded from C.
Step0 action equality with C is checked before any update.

| Arm | Target | Keep coefficient |
|---|---|---:|
| E-C | AE-C, (z_C-mu_C)/s_C | 0 |
| E-D | AE-D, (z_D-mu_D)/s_C | 0 |
| E-C-keep | AE-C, (z_C-mu_C)/s_C | 1 |
| E-D-keep | AE-D, (z_D-mu_D)/s_C | 1 |

Every arm uses action BC plus the original fixed policy RLA coefficient
0.22408536585365854. Keep is mean L1 over valid normalized 8x7 action elements.
Students see the sampled clean/background observations; frozen eval C supplies
the same-state clean teacher action. Teacher outputs are cached for the 20 train
episodes only and are not physical ground truth. No development teacher targets.

The common C training cost is 600 policy updates. Each student starts a fresh
AdamW/scheduler and adds exactly600 updates with the same sampling/RNG protocol.
AE and teacher parameters are frozen. No new physical loss, target rescaling or
automatic coefficient calibration. All four compute the keep diagnostic.

Use the accepted official Prismatic/VLA environment; resolve local model/data
paths in prepare.py to authorized data-volume inputs. Set all cache roots there.
Run `prepare.py --out DATA_VOLUME/unique_results`, then `run.py --out ...`.
The runner performs teacher caching, zero-update full-batch gate, four serial
trainings, then exactly240 frozen development rollouts. It stops on any failure.
Checkpoint resumes must restore model/optimizer/scheduler/RNG/sampling position.
No automatic device/backend/parameter fallback or trial retry.

The original20 development states are reused, with clean/seen/unseen material
conditions, reflection1 and MSAA4. Results are single-seed development evidence.
Analysis resamples physical states jointly across appearances. Recomputed contact
metrics never replace recorded online official task success labels.

Lineage: mlzxy/rla-wm commit0fc6e97212442ebccffa06f1d42f5481e5f22950,
official rebuttal/exp2_vla_adapter overlay and this repository's accepted stove
trainer, dataset and evaluator. Existing licenses/source attribution apply.
