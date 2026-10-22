# BC-ResNet policy appearance consistency

Independent from the previous Spatial-Pro repair route. The executable policy
comes from `mlzxy/rla-wm` commit `0fc6e97212442ebccffa06f1d42f5481e5f22950`,
class `policies.policy.vla_bc_policy_rla_unified.VLABCPolicyRLAUnified`,
with `use_latent_head=False`. Despite its class name, no RLA/DINO modules are
created. Read `baseline_provenance.md` before using the LIBERO adapter.

Reproduction requires an independent checkout at that commit (never switch an
existing worktree), its pinned dependencies, and the privately held frozen
Step4.4.1 dataset. The dataset is intentionally not included. Tested CPU stack:
torch2.8.0+cu128, torchvision from the existing RLA environment, Python3.10.
Torch's CUDA build does not imply CUDA execution: `CUDA_VISIBLE_DEVICES=''`.

```bash
export RLA_PROJECT_ROOT=/path/on/data-volume/rla-wm
export RUN_DIR=/path/on/data-volume/new-unique-run
bash experiments/policy_appearance_consistency/run_audit.sh
```

The script refuses output overwrite via exclusive output creation. It expects
the original checked-out code at `worktrees/rla-official-0fc6e97` and datasets at
the documented experiment-relative locations. Adapt these path bindings for
your machine without altering source-ID semantics. Historical CSV paths may
need an audited prefix relocation. All caches/temp must remain on the data
volume. No model weights are downloaded; initialization is random.

`method.py` defines matched-RNG paired forward and masked objectives;
`dataset.py` reads real current images/FK/actions and fits train-only stats;
`entry.py` exposes `step_loss` and a CPU/no-update dry-run. `entry_audit.py`
verifies separately initialized policies using this exact entry. Run it with
the same environment/PYTHONPATH as `run_audit.sh`. There is no optimizer loop
enabled in this delivery. The next training budget is a proposal only.

This code imports the authoritative upstream implementation, not the old
custom VLA wrapper or a newly invented surrogate network. Original repository
metadata declares MIT; no independent LICENSE file exists at the pinned commit.
Retain upstream attribution and all existing repository license notices.
