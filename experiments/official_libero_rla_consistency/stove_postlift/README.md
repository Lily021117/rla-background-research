# Frozen stove post-lift diagnosis

No training or backward. Uses the already verified stove C final600 and
D-center final600, not other-task policies. Source entry uses the accepted
official model bootstrap and exact inference/deployment path. The diagnostic
branch is separate from formal success-rate tables.

Order (all outputs/caches on the data disk; original isolated environments):

1. `prepare.py --out NEW_DIRECTORY`: freeze first five manifest states, limits,
   sample rules, provenance, and the strict restoration tolerances.
2. CPU `history.py`, `demo_coverage.py`, `phase_logs.py` with the same `--out`.
   Set `CUDA_VISIBLE_DEVICES=''`, `MUJOCO_GL=osmesa`, LIBERO paths and data-disk
   TMP/cache variables. `history.py` uses saved states and `mj_forward` only.
3. `launch.py --out NEW_DIRECTORY`: UUID/free-memory/ECC guarded GPU1 bootstrap;
   at most10 prefixes and20 continuations. No optimizer construction/update.
   Self continuation must reproduce historical actions/states/termination;
   otherwise the process stops before accepting cross-handoff attribution.
4. CPU `events.py`, `condition_sequence.py`; then `finalize.py` once the GPU
   process has exited and its receipt is present.

Snapshots include full MuJoCo integration input state, controller primitives,
robot buffers, gripper current action, observation caches and all RNG streams.
Static simulator/model references remain bound to the same unchanged environment.
There are no controller interpolators in this accepted configuration; this is
asserted. At switching boundaries the action queue is empty. Policy inference
uses a single current dual-view observation, language and synchronized robot FK
proprio, with no target or future frame input. It has no persistent KV cache.

The actual runtime inventory records saved versus static/skipped attributes.
Restoration is an empirical zero-error gate on the complete remaining self
continuation, not an assumption based on qpos shape. Same-controller continuation
and cross-controller continuation have identical prefix-specific remaining
control budgets. Cross-handoff results are not model-selection evidence.

Only sanitized documents and small derived metrics are published. Full videos,
raw states, snapshots, action trajectories and weights stay local. Original
project licenses/source attribution remain unchanged. This is a LIBERO-derived
custom material intervention, not official LIBERO-Plus.
