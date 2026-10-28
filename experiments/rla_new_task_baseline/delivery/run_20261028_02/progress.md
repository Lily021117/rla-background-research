# Official-source middle-drawer experiment: preparation and compatibility repair

This is an in-progress report, not a training or success-rate result.

## Data admission

Only the authorized `libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5`
was downloaded from `yifengzhu-hf/LIBERO-datasets`, dataset revision
`97773100c1474cd0d686ebd173cc0e4fd5442466`.
Size: 702223367 bytes. SHA-256:
`20252c7cf98cd7437061f7f200ae7b6cb6219fabbd53b4536dfaa8abda6ab737`.

The official file is a new source, not a repair of the previous Cosmos candidate
episodes. Numeric episodes 0–19 form training; 20–24 form development.
All 25 selected episodes passed the derived paired-render admission: 3470 saved
physical frames, 2647 complete training anchors and 623 development anchors.
Incomplete future windows are excluded from policy samples.

The source exporter records post-action observations alongside pre-action saved
states and actions. Adjacent state/observation robot joints and fingers match
exactly; selected source states have 0.05-second spacing. Both camera images and
robot-only FK proprio are rederived at the same saved pre-action state. This does
not claim the new content has the original RLDS content identity.

An initial admission attempt incorrectly guessed the drawer joint name. Its
failure record is preserved. The XML identifies the actual joint as
`wooden_cabinet_1_middle_level`; the corrected export is a new data version.
It does not change the selected episodes or task goal.

## Startup repairs

The first encoder attempt failed because the policy environment's PyTorch does
not expose `device_properties.uuid`. The guard now obtains the process-visible
device UUID through `cuDeviceGetUuid_v2`, matching the existing policy bootstrap.
The expected device check remains mandatory; it was tested successfully.

A subsequent attempt revealed that the extraction entry had incorrectly used
the policy environment, which lacks `lxml`. Extraction now uses the established
isolated sidecar environment, without installing/upgrading packages in either
environment. Both failures occurred before policy updates and are preserved.

The GPU gate requires at least 26 GiB free for three consecutive checks and
unchanged ECC counters. No other process is terminated. The known faulty GPU
is not an alternative. A gate is not a scheduler reservation.

## Fixed continuation

Original official step90000 AE and DINO are frozen. Each appearance retains its
own anchored eight-horizon targets; all policy normalization statistics are fit
only on the new task's training split. No stove-finetuned AE is used.

The standard action head and the RLA action head have different structures.
CPU comparison confirmed their step0 VLA and proprio parameters match exactly.
BC-matched and C-frozen-RLA load the same complete RLA step0 and RNG.

After target extraction and real-batch acceptance, the bounded coordinator runs
BC-standard, BC-matched, C-frozen-RLA for sampling orders 20261101 and 20261102:
600 successful updates each, 3600 total. No AE updates. Final step600 only.
Clean scoring is 120 trials at the LIBERO-Goal 300-step limit. Only if one method
achieves at least 12/20 in both orders are all six policies evaluated on all four
prefrozen backgrounds (480 additional trials). Otherwise background scoring stops.

Any failed prerequisite stops dependent work with logs retained. No sealed
initial states, held-out trajectories, or extra task downloads are used.

## Reproduction entry points

`download_official.py`, `prepare_official.py`, `prepare_states.py`, and
`freeze_data.py` construct and accept derived data. `gpu0_queue.py` gates
`extract_frozen.py`; `prepare_policy.py` verifies initialization and freezes
plans. `continue_pipeline.py` serially gates real-batch acceptance, training,
Clean scoring, and conditional background scoring. `summarize.py` reports all
methods/orders and paired state statistics. Paths are supplied through `--run`
and `--project`; local dependency locations follow the existing project setup.

Only code and this sanitized progress report are published. Images, simulator
states, checkpoints, target arrays, and videos remain outside the repository.

## Verified continuation milestones

Frozen encoding completed for all 25 episodes (50 appearance-specific files).
All 150 source-image recomputation checks and 50 deterministic repeat checks
passed. Maximum absolute cached/recomputed latent difference was
0.0038967132568359375; this is an implementation consistency check, not a claim
of representation quality. AE, DINO, and policy updates during extraction: 0.

Both full-batch preflights passed (microbatch 8, accumulation 2). With the
auxiliary coefficient zero, target replacement preserved exact action-output,
action-loss, and action-gradient equality. Weighted auxiliary gradients reached
the shared backbone in the RLA path (measured norm approximately 0.187).
Each preflight verified unchanged initial parameters and performed zero updates.
Peak allocated memory was approximately 18.47 GiB; the 26 GiB startup guard is
unchanged. The bounded coordinator then advanced to formal policy training.

No completed-policy or closed-loop performance conclusion is available at this
progress checkpoint. In particular, a successful preflight does not establish
middle-drawer task learning or appearance robustness.
