# Coverage and gripper diagnosis

This is an offline diagnostic and derived-data export workflow, not training.
Optimizer updates, backward calls, and new policy rollouts are all zero.

Run scripts with `--help` for explicit input/output paths. Use a fresh output
directory on the data volume and configure HF_HOME, TORCH_HOME, XDG_CACHE_HOME,
and TMPDIR there before loading models. Do not include protected episodes.

Workflow:

1. `prepare_pool.py`: inspect source inventory and protected identities; freeze
   selection before model scoring. Preserve existing train/development roles.
2. `render_pool.py`: export selected saved states using the guarded
   `../recover_sequence.py --selection-manifest` path. Reuse accepted exports.
3. `freeze_pool.py`: validate exports, freeze provenance and episode splits,
   and calculate training-only normalization statistics.
4. `loader_audit.py` / `dataset.py`: exercise the official normalization,
   action chunking, batch transform, and collator with the frozen manifest.
5. `label_rollout_audit.py`: audit original labels, recorded sampling exposure,
   and the two existing trajectories. No simulator dynamics are stepped.
6. `checkpoint_inference.py`: frozen step0/45/135/270 inference on a fixed
   sample manifest using Math SDPA, including deployment-path comparisons.
7. `transition_timing.py`, `execution_response.py`, and `plot_diagnosis.py`:
   summarize current/future chunk closure timing and saved command execution.
8. `make_report.py`: produce local delivery and sanitized small summaries.

The raw-action gripper convention is -1=open, +1=close; the official training
transform produces 1=open, 0=close. The last action normalization mask is false.
The deployed command is `-sign(2*p-1)`, including a neutral command at p=0.5.
Do not replace this with an unrecorded threshold or extra gripper loss weight.

The `delivery/` directory contains sanitized aggregate findings and proposed,
unexecuted next-run configurations. Detailed manifests, figures, checkpoints,
prediction arrays, and source trajectories remain local and are not uploaded.
Source and dependency provenance follows the parent experiment documentation;
retain the upstream license notices in this repository.
