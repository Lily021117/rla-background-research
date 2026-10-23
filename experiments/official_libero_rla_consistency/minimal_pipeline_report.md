# Minimal pipeline result: PARTIAL / training BLOCKED

Completed: independent worktree; official/upstream pin audit; real220-row train/validation manifest audit; fixed real training-pair array/hash checks; official released90000 archive download with exact published SHA; Encoder and Decoder strict CPU load; source-based action/proprio/camera/sidecar audit. Historical reports reused only for their actual scope.

Not completed: native224 dual-view paired replay recovery, new paired h1…8 sidecars, full official Prismatic baseline/RLA fixed-batch forward, paired backward and end-to-end RNG/checkpoint resume test. No claim of pipeline PASS.

Concrete blockers:
1.220/220 valid transfers have both h8 endpoints;0/220 have complete h1…8 images. Historical32×64 targets cannot be reused.
2.Per-camera arrays store camera_K with shape(2,3,3), because pilot_generate.py:107 indexes the time-major geometry array with camera index rather than [time,camera]. Original geometry writer prepare_pairs.py:99,134 establishes [time,camera,3,3]. Recover from source, do not squeeze/guess.
3.Geometry convention includes mirrored horizontal intrinsic sign for historical training pixels, whereas paired renderer returns modern MuJoCo RGB directly. Orientation-to-ray equivalence is not certified. Upsampling128px to224 is not lossless.
4.FK state is same recorded physical state, but differs numerically from raw observed EEF proprio. The official state-hash join cannot reuse RLDS hashes for FK streams.
5.Local Prismatic base directory contains only README, not its weights. Public trained Spatial-Pro is only a reference fixture, not approved training initialization.
6.Official paper40000 unavailable in downloaded archive; released90000 is now a verified reference, not exact paper replication.
7.Stove validation remains missing (3 validation demos).

The official full multiview forward is deliberately not fed malformed geometry or random substitute input and reported as successful. Existing P6 step750 real-RLDS checks demonstrate old plumbing only; they do not certify the new data.

GPU1 resource/ECC preflight: free81065MiB; historical uncorrectable parity1/SM1, correctable2, threshold not exceeded. No CUDA computation launched. All new computation CPU; no optimizer, backward, rollout or held-out read.
Prior BC-ResNet branch retained, no further training launched.

