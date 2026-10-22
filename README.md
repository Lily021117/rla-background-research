# RLA background research — private reproducibility snapshot

Controlled experiments using ManiSkill-derived RLA targets as VLA-Adapter auxiliary supervision. This is a custom cross-domain experiment. It is not an exact reproduction of the author Original LIBERO RLA policy.

## Contents

- `source/vla`: the actual unified head/objective, Step5.7 Math SDPA trainer, Step5.8 rollout entry and required VLA-Adapter source.
- `source/data`: the real sparse dataset/target adapters used to construct cached batches. The old SparseHead is historical; Step5.7 uses UnifiedHead.
- `source/preparation`: exact same-state FK, batch construction, DINO and target-generation lineage.
- `source/autoencoder`: custom B/C reconstruction/consistency experiment code; official encoder/decoder implementation remains an external prerequisite.
- `configs`: frozen training/evaluation templates with server paths removed.
- `diagnostics/step5_9`: diagnostic scripts, aggregate evidence, report and next proposal.
- `source_inventory.csv`: source and sanitized-file SHA-256, with origins.

## Dependencies and external inputs

VLA-Adapter checkout: https://github.com/OpenHelix-Team/VLA-Adapter at `23fa0c9c159e2aa04341cdd3e924f44061311060`. The included VLA-derived source keeps its original MIT license in `LICENSES`.

RLA official checkout: https://github.com/mlzxy/rla-wm at `0fc6e97212442ebccffa06f1d42f5481e5f22950`. The RLA repository has no root license file in the audited checkout, so its core model/trainer source and official overlay are not vendored or relicensed here. Obtain it separately and preserve its terms. The custom scripts import its SimpleTokenTransformer and official `rla.data`/sidecar integration.

LIBERO-Plus: https://github.com/sylvestf/LIBERO-plus. Public Spatial-Pro model revision `cb023c6771fc62ea729d218f09ac0583c14f59d4` is an external input; no model files are included.

Use the VLA project's `pyproject.toml` plus the measured experiment runtime: Python3.10, PyTorch2.2.0+cu121, transformers4.40.1, peft0.11.1, NumPy1.26.4, TensorFlow2.15.0, mujoco3.3.2, robosuite1.4.1, h5py3.16.0, Pillow12.3.0, imageio2.38.0 and ffmpeg. DINO extraction additionally uses decord from the RLA environment; it has no distribution metadata in the VLA diagnostic environment. `diagnostics/step5_9/runtime_versions.json` is the measured inventory, not a lockfile. Do not reinstall an accepted environment solely to match these notes.

## Localizing and reproducing

This is a source snapshot, not a data release. The exact accepted batch/feature/target/state manifests, raw HDF5, model checkpoints and target normalization NPZ files must be provided locally. They are intentionally excluded. Source hashes identify the original scripts even though server paths have been replaced by `${RLA_PROJECT_ROOT}`, `${VLA_REFERENCE_ROOT}`, `${LIBERO_ROOT}`, `${RLA_CACHE_ROOT}`, `${DATA_ROOT}`, `${GPU_UUID}` and environment package placeholders. Replace these placeholders in a new local working copy before execution, retaining a record of the mapping. Python literal placeholders are not automatically shell-expanded.

For the policy: locate `source/vla` as the VLA worktree; preserve the experiment-relative directory layout referred to by the source; expose the external official overlay and custom sparse data module on PYTHONPATH. Start from `prepare_acceptance.py` and the saved acceptance conditions before considering the historical `run_math_step57.sh` training command. The Step5.9 audit itself never creates an optimizer, calls backward, runs rollouts, or updates weights. Do not run historical training/rollout entries as part of the diagnostic.

## Scope and limitations

Single training seed; 19 train episodes/190 transitions; three validation episodes without stove coverage; 20 historical exploratory evaluation states. Step5.8 results: A16/20 clean and18/20 background; B2/20 and3/20; C4/20 and4/20; D7/20 and6/20. No robust causal benefit has been established. D-vs-C background interaction is -5pp with a CI spanning zero.

The snapshot has no parent project history, checkpoints, datasets, images/videos, raw trajectories, credential files or secret material. All upstream attribution remains in source and LICENSES. No new open-source license is assigned to the unpublished custom research code by this private archival upload.
