# Single trajectory official VLA-RLA acceptance

**PASS — bounded single-trajectory interface acceptance, not training or policy quality.**

## Data and time

Selected first entry in the frozen TRAIN order: `sealed_163_demo_12` / `demo_12`. All93 saved states restored independently,186 dual-appearance camera pairs /372 RGB images,224px;85 complete8-step windows per appearance. Prefix17frames/9complete chunks passed before completing the trajectory. No other trajectory was rendered. The identifier's historical `sealed_` prefix does not mean this is held-out: it is explicitly in the pre-existing train split.

At derived timestep s: image and FK use recorded state[s]; actions[s:s+8] connect to state[s+8],0.05s/control and0.4s/8steps. mj_forward/render do not integrate or advance time; qpos/qvel/time and paired body/camera arrays checked exactly. Nominal raw obs[s-1] remains a cached post-action observation, not independently timestamped truth. State0 has no preceding raw observation and no fabricated comparison.

This is **基于 LIBERO 示范构建的跨外观配对数据**, evaluated under **基于 LIBERO 的自定义背景材质干预协议**, not official LIBERO-Plus. New version dual224_fk_v2 preserves source HDF5/demo/state/action indexes and appearance; the old RLDS identity is explicitly UNMATCHED (Step4.2), not guessed. The new full-FK state hash is not an old RLDS sample hash.

## Geometry and proprio

Native224 renders, modern top-down RGB horizontally mirrored to match negative-fx camera convention; pixel centers index+0.5, K[t,camera], cam-to-world OpenCV, wrist recomputed every frame.18 raster-marker projection checks across3states/2cameras passed, maximum error 3.18e-14px. These are surface points unprojected from depth and independently rasterized as isolated diagnostic markers, not external pose-ground-truth measurements. Formal images contain no markers. Overlay examples and full world coordinates are local only.

Segmentation/depth/physical states match for all186 pairs; background RGB changed in186/186. Reflection1 and MSAA4 held fixed; only tex-table/tex-wall asset references changed. RGB/reflection changes are permitted; this is not strict foreground-RGB invariance. No separate shadow/contact semantic mask is claimed.

Chosen proprio: same-state grip-site xyz, right-hand axis-angle, two finger qpos.93-frame axis-angle comparison with official robosuite conversion agrees exactly after float32. No independent EEF pose truth. Raw-vs-FK maximum position-component residual 0.000472400m, rotation-component residual 0.001218387rad; neither was hidden by a temporal offset or tolerance. The new choice resolves the derived image/proprio timestamp, not the historical raw sensor refresh phase. Official published Spatial Q99 statistics used unchanged for both appearances; formal derived-data distribution suitability still needs a training-only audit.

## AE and sidecar

Official LIBERO step90000 (not paper40000), joint dual-view16x64 Plücker AE. Real input[1,2,2,3,224,224],latent[1,16,64],decoder[1,2,196,1024]. Repeat inference exact. Per-camera reconstruction metrics and latent norm/variance in ae_forward_results.json; finite losses do not prove useful representation.

Two sidecars each[93,8,16,64], official anchored t→min(t+h,N−1),h1…8, same frozen AE, not trained AE-C/D. Separate appearance roots avoid same-FK-hash collisions. Six fixed source-image recomputations passed official2%-scale numerical tolerance (maximum absolute error 0.006502); batch-shape/fp16 storage error reported rather than claimed exact. Validity mask excludes non-existent futures. Official action chunker yields86 action-complete windows but only85 have all future images; last window is NOT admitted for full auxiliary supervision. Stats use85×8 complete targets; reader center mode is per-coordinate mean/global std.

## Official initialization and real forward

Pinned RLA commit0fc6e97212442ebccffa06f1d42f5481e5f22950; upstream23fa0c9c159e2aa04341cdd3e924f44061311060. Prismatic repo revision5cfd2cc6da00c06e0be7abf35d43ec792d8e9498, checkpoint SHAd86525b8d21865b2e83d4a403ee9d1b641579bb41259111d7e3cc7f7ed46564a. Official loader restores LLM/projector plus separately specified DINOv2/SigLIP pretrained vision; not pure Qwen replacement and not Spatial-Pro. CPU training-load attempt hit missing flash_attn; official inference-load succeeded. Policy conversion follows finetune key mapping, only action_queries.weight intentionally new; assembled state strictly loaded, fresh action/proprio/LoRA per recipe. These are valid initialized networks, not trained robot checkpoints.

Real trajectory → official TF normalization and chunking → RLDSBatchTransform/RlaBatchTransform → official collators → official finetune.run_forward_pass / RLA loss wrapper. This is an in-memory derived-trajectory adapter, not a claim that a full TFDS export was built. Appearance selection is explicit before the official hash/slot hook. Batch image[1,12,224,224],actions[1,8,7],RLA[1,8,1024]; collator squeezes batch1 proprio to[8], official head reshapes it.

Original upstream baseline/RLA forward succeeded for Clean and Background, separately preserved. MiniVLM training transform omits STOP while deployment appends it; therefore an explicit caller-side STOP/mask repair was additionally tested for BOTH paths: append STOP2 after64action slots, mask all prompt labels. This is a documented data-boundary departure, not an official model/head rewrite or paper-exact replication. Token random-padding RNG matched between paths. STOP-aligned forward also passed. Same-RNG replacement of numeric action labels and z targets leaves action output exactly unchanged; future RGB never enters policy batch. Code replaces masked action embeddings with query embeddings. This does not claim full deployment equivalence or closed-loop quality.

## Resources and limits

GPU1 UUID verified in-process,26GiB threshold; AE peak allocated 2.02GiB, VLA 3.76GiB. Historical ECC retained, no increment. CPU/OSMesa rendered, isolated frozen GPU forwards only. Initial failures (UUID formatting guard, missing FlashAttention, unmatched token-padding RNG, single-process CPU labels) were stopped before updates, logs retained; final runtime fixes do not alter weights. All data/cache writes under data volume; existing content-addressed HF cache received dependency additions, no historical blobs overwritten.

Optimizer updates=0; backward=0; new policy rollouts=0; held-out data reads=0. Recorded historical source hashes reverified; no exhaustive unrelated history rehash claimed. Projection, FK and valid-mask checks justify single-trajectory extension preparation only. This does not validate all220 previous windows or prove enough data for policy training.
