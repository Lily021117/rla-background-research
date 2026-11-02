# RLA paired-data V2 prototype

CPU-only saved-state rendering for stove bowl-to-plate and opening the middle drawer. No policy/model imports or simulator stepping. This is a data prototype, not a method evaluation.

## Reproduction

Set `RLA_PROJECT_ROOT` to the existing accepted project data tree, `RLA_LIBERO_ROOT` to its validated LIBERO-plus checkout and `PAIRED_V2_ROOT` to a **new empty output directory**. Use the previously validated environment with numpy, scipy, Pillow, h5py, mujoco, robosuite and imageio/ffmpeg. Set `CUDA_VISIBLE_DEVICES=''`, `MUJOCO_GL=osmesa`, `PYOPENGL_PLATFORM=osmesa`, `LP_NUM_THREADS=2`, `OMP_NUM_THREADS=2`.

1. `python prepare.py` — source registration, bounded admission, material/light XML creation. Uses accepted historical manifests; it does not download data.
2. `python render.py --mode sample` — fixed first source per task, five uniform times, RGB/depth/GEOM segmentation.
3. Review the fixed sample images and gates; explicitly save `sample_review.json` with approval/reason. Never approve based on policy performance.
4. `python batch_render.py` — two CPU subprocess workers, five full variants per source.
5. `python finish.py`, `python verify.py`, `python package.py` — pairing index, fixed videos, integrity checks and sanitized reports.

Scripts deliberately refuse overwriting outputs. A failed attempt must retain its original directory; technical revisions use a new directory. The delivered run's `revision2` follows a retained first attempt whose stove material failed the brightness gate. `finish.py` includes sample results in its parent directory when present; adapt this accounting explicitly for a different run layout, not by deleting history.

## Format

Each `<episode>/<appearance>.npz` contains `rgb` of shape `[N,2,224,224,3]`, uint8, and `timesteps`. Camera order is agentview then robot0_eye_in_hand. `physical.npz` is shared across appearances and includes original state/action arrays, synchronized FK proprio, target/eef pose, plate or drawer joint and per-frame camera matrices. RGB is horizontally flipped under the accepted convention, with corresponding negative x focal entry in K. Consumers needing CHW must transpose explicitly.

The JSONL pairing manifest uses source episode/timestep/appearance/version identity, anchored future horizons 1..8 and validity masks. No padded copies of the last frame are created. This work does not extract AE targets or exercise a training loader.

Recorded positions use world coordinates in meters: `gripper0_grip_site` for eef, `akita_black_bowl_1_main` and `plate_1_main` body origins for stove objects, `wooden_cabinet_1_cabinet_middle` body origin and `wooden_cabinet_1_middle_level` joint for the drawer. Body origins are not claimed to be geometric centroids. Quaternion records use MuJoCo wxyz; proprio rotation uses the accepted robot hand rotation-vector convention, followed by the two finger joint positions. Raw seven-dimensional source actions are preserved, not replaced by gripper position or normalized in this prototype. Physical labels are audit records, not new supervision losses.

Only code and small sanitized metadata are versioned. Images, videos, XML/source states, original HDF5 and generated texture caches remain local. Preserve repository license/source attribution; original LIBERO/robosuite assets are not redistributed here.
