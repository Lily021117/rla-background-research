"""Summarize completed read-only run; publish only aggregate sanitized outputs."""
import argparse,json,csv,hashlib,subprocess,os,re,importlib.metadata
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--project',type=Path,required=True);a=p.parse_args();o=a.out;src=Path(__file__).resolve().parent;pub=src/'single_trajectory_acceptance';pub.mkdir()
def load(n):return json.load(open(o/n))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def js(n,x):
 with (o/n).open('x') as f:json.dump(x,f,indent=2)
t=load('trajectory_provenance.json');g=load('camera_geometry_audit.json');ae=load('ae_forward_results.json');v=load('official_forward_results_stop_aligned.json');sc=load('sidecar_alignment_audit.json');init=load('initialization_inventory.json')
assert all(sha(path)==h for path,h in t['source_hashes'].items())
previous=load('../run_20261023_01/source_provenance.json');assert all(sha(path)==h for path,h in previous['hashes'].items())
dependency=[]
cache=Path(os.environ['DATA_ROOT'])/'hf_cache'
for rel in ['hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m','hub/models--timm--ViT-SO400M-14-SigLIP','transformers/models--Qwen--Qwen2.5-0.5B']:
 root=cache/rel;revision=(root/'refs/main').read_text().strip();files=[]
 for f in sorted((root/'snapshots'/revision).iterdir()):
  if f.is_file():
   actual=sha(f);name=f.resolve().name
   if len(name)==64:assert actual==name
   files.append(dict(file=f.name,bytes=f.stat().st_size,SHA256=actual))
 dependency.append(dict(repo=root.name,revision=revision,files=files))
configs=o/'vla-official/pretrained_models/configs'
processor_files=[dict(file=p.name,SHA256=sha(p)) for p in sorted(configs.iterdir()) if p.is_file() and p.suffix in ['.json','.txt']]
js('initialization_dependencies.json',dict(pretrained_components=dependency,processor_files=processor_files,processor_source_commit='23fa0c9c159e2aa04341cdd3e924f44061311060',note='Official loader used append-only existing data-disk HF cache due inherited HF_HUB_CACHE/TRANSFORMERS_CACHE; no system cache or historical model replacement. HF_HOME/TORCH/XDG/TMP were task-specific.'))
pro=list(csv.DictReader(open(o/'proprio_alignment_audit.csv')));residual={k:max(float(r[k]) for r in pro if r[k]) for k in ['xyz_max_m','rot_max_rad','finger_max_m']}
js('same_state_fk_rotation_check.json',dict(frames=93,official_robosuite_axisangle_float32_max_abs=0.,external_pose_truth=False,raw_residual=residual))
aeck=load('../run_20261023_01/checkpoint_acceptance.json');js('target_checkpoint_provenance.json',dict(models=aeck['models'],resolved_step=90000,not_paper_step40000=True,paths_relative_to_previous_run='downloads/extracted/runs/16x64_libero/20260725_00-39-52/ckpts',sidecars={appearance:sha(o/('sidecar_'+appearance)/'libero_spatial/ep_000000.npy') for appearance in ['clean','background']}))
report=f'''# Single trajectory official VLA-RLA acceptance

**PASS — bounded single-trajectory interface acceptance, not training or policy quality.**

## Data and time

Selected first entry in the frozen TRAIN order: `{t['selected']}` / `{t['source_demo']}`. All93 saved states restored independently,186 dual-appearance camera pairs /372 RGB images,224px;85 complete8-step windows per appearance. Prefix17frames/9complete chunks passed before completing the trajectory. No other trajectory was rendered. The identifier's historical `sealed_` prefix does not mean this is held-out: it is explicitly in the pre-existing train split.

At derived timestep s: image and FK use recorded state[s]; actions[s:s+8] connect to state[s+8],0.05s/control and0.4s/8steps. mj_forward/render do not integrate or advance time; qpos/qvel/time and paired body/camera arrays checked exactly. Nominal raw obs[s-1] remains a cached post-action observation, not independently timestamped truth. State0 has no preceding raw observation and no fabricated comparison.

This is **基于 LIBERO 示范构建的跨外观配对数据**, evaluated under **基于 LIBERO 的自定义背景材质干预协议**, not official LIBERO-Plus. New version dual224_fk_v2 preserves source HDF5/demo/state/action indexes and appearance; the old RLDS identity is explicitly UNMATCHED (Step4.2), not guessed. The new full-FK state hash is not an old RLDS sample hash.

## Geometry and proprio

Native224 renders, modern top-down RGB horizontally mirrored to match negative-fx camera convention; pixel centers index+0.5, K[t,camera], cam-to-world OpenCV, wrist recomputed every frame.18 raster-marker projection checks across3states/2cameras passed, maximum error {g['max_projection_error_px']:.3g}px. These are surface points unprojected from depth and independently rasterized as isolated diagnostic markers, not external pose-ground-truth measurements. Formal images contain no markers. Overlay examples and full world coordinates are local only.

Segmentation/depth/physical states match for all186 pairs; background RGB changed in186/186. Reflection1 and MSAA4 held fixed; only tex-table/tex-wall asset references changed. RGB/reflection changes are permitted; this is not strict foreground-RGB invariance. No separate shadow/contact semantic mask is claimed.

Chosen proprio: same-state grip-site xyz, right-hand axis-angle, two finger qpos.93-frame axis-angle comparison with official robosuite conversion agrees exactly after float32. No independent EEF pose truth. Raw-vs-FK maximum position-component residual {residual['xyz_max_m']:.9f}m, rotation-component residual {residual['rot_max_rad']:.9f}rad; neither was hidden by a temporal offset or tolerance. The new choice resolves the derived image/proprio timestamp, not the historical raw sensor refresh phase. Official published Spatial Q99 statistics used unchanged for both appearances; formal derived-data distribution suitability still needs a training-only audit.

## AE and sidecar

Official LIBERO step90000 (not paper40000), joint dual-view16x64 Plücker AE. Real input[1,2,2,3,224,224],latent[1,16,64],decoder[1,2,196,1024]. Repeat inference exact. Per-camera reconstruction metrics and latent norm/variance in ae_forward_results.json; finite losses do not prove useful representation.

Two sidecars each[93,8,16,64], official anchored t→min(t+h,N−1),h1…8, same frozen AE, not trained AE-C/D. Separate appearance roots avoid same-FK-hash collisions. Six fixed source-image recomputations passed official2%-scale numerical tolerance (maximum absolute error {max(r['max_abs_error'] for r in sc['recompute_checks']):.6f}); batch-shape/fp16 storage error reported rather than claimed exact. Validity mask excludes non-existent futures. Official action chunker yields86 action-complete windows but only85 have all future images; last window is NOT admitted for full auxiliary supervision. Stats use85×8 complete targets; reader center mode is per-coordinate mean/global std.

## Official initialization and real forward

Pinned RLA commit0fc6e97212442ebccffa06f1d42f5481e5f22950; upstream23fa0c9c159e2aa04341cdd3e924f44061311060. Prismatic repo revision{init['revision']}, checkpoint SHA{init['checkpoint_SHA256']}. Official loader restores LLM/projector plus separately specified DINOv2/SigLIP pretrained vision; not pure Qwen replacement and not Spatial-Pro. CPU training-load attempt hit missing flash_attn; official inference-load succeeded. Policy conversion follows finetune key mapping, only action_queries.weight intentionally new; assembled state strictly loaded, fresh action/proprio/LoRA per recipe. These are valid initialized networks, not trained robot checkpoints.

Real trajectory → official TF normalization and chunking → RLDSBatchTransform/RlaBatchTransform → official collators → official finetune.run_forward_pass / RLA loss wrapper. This is an in-memory derived-trajectory adapter, not a claim that a full TFDS export was built. Appearance selection is explicit before the official hash/slot hook. Batch image[1,12,224,224],actions[1,8,7],RLA[1,8,1024]; collator squeezes batch1 proprio to[8], official head reshapes it.

Original upstream baseline/RLA forward succeeded for Clean and Background, separately preserved. MiniVLM training transform omits STOP while deployment appends it; therefore an explicit caller-side STOP/mask repair was additionally tested for BOTH paths: append STOP2 after64action slots, mask all prompt labels. This is a documented data-boundary departure, not an official model/head rewrite or paper-exact replication. Token random-padding RNG matched between paths. STOP-aligned forward also passed. Same-RNG replacement of numeric action labels and z targets leaves action output exactly unchanged; future RGB never enters policy batch. Code replaces masked action embeddings with query embeddings. This does not claim full deployment equivalence or closed-loop quality.

## Resources and limits

GPU1 UUID verified in-process,26GiB threshold; AE peak allocated {ae['peak_GPU_allocated']/2**30:.2f}GiB, VLA {v['peak_GPU_allocated']/2**30:.2f}GiB. Historical ECC retained, no increment. CPU/OSMesa rendered, isolated frozen GPU forwards only. Initial failures (UUID formatting guard, missing FlashAttention, unmatched token-padding RNG, single-process CPU labels) were stopped before updates, logs retained; final runtime fixes do not alter weights. All data/cache writes under data volume; existing content-addressed HF cache received dependency additions, no historical blobs overwritten.

Optimizer updates=0; backward=0; new policy rollouts=0; held-out data reads=0. Recorded historical source hashes reverified; no exhaustive unrelated history rehash claimed. Projection, FK and valid-mask checks justify single-trajectory extension preparation only. This does not validate all220 previous windows or prove enough data for policy training.
'''
with (o/'report.md').open('x') as f:f.write(report)
plan='''# Next plan — not executed

1. Extend the accepted native224/geometry/FK/appearance-ID pipeline to the remaining18 TRAIN and3 development episodes, in batches with per-episode checks. Do not reintroduce the excluded stove validation episode or touch historical holdouts. These21 episodes are candidates, not automatically accepted. Existing220 sparse windows require all real h1…8 images.
2. Preserve broad original LIBERO supervision:432 local Spatial episodes cover10tasks; current paired set covers only a subset. Before mixing, audit original vs derived episode identity/deduplication, and keep original clean data exposure identical in B/C/D. Do not treat an UNMATCHED HDF5 demo as a new independent trajectory relative to RLDS. Other3suites and a suitable stove development cohort remain missing for full-paper-scale claims.
3. AE-C/D: official multiview16x64 architecture, same initialization/RNG/data/budget. Current step90000 is a frozen interface reference. A shared90000 continuation must be explicitly named as a released-checkpoint warm start; official from-scratch alternative uses common random initialization. Neither is paper40000 exact replication. Average own-appearance reconstruction for both; only AE-D adds latent consistency. Freeze initialization choice and training-only loss scale before training; no automatic launch.
4. Policy B/C/D use the now-verified Prismatic source plus identical shared initialization and official heads/LoRA recipe. B noRLA; C/D official RLA and targets from respective AEs. Same observations/language/action labels/sampling/cumulative exposure. Fresh heads require meaningful training, not100-step finished-policy repair. Stage1/Stage2 cost accounting and reference20K+10K budgets remain separate from resource-approved pilot decisions.
5. Adopt or independently review the documented STOP/mask repair uniformly; do not silently mix upstream-noSTOP and repaired training. Add a real TFDS/stream export, per-variant store routing and a strict complete-future mask at dataset selection. Complete training-only normalization and auto:0.25 effective-weight/confound audit. Install/verify pinned FlashAttention if reproducing the official training backend; no environment upgrades were made this round.
6. No training is released automatically. Independent evaluation states/unseen backgrounds and standard LIBERO/custom-background protocols must be frozen later. Historical20 states remain exploratory development. No new rollout, sampling or additional seed authorized by this report.
'''
with (o/'next_training_plan.md').open('x') as f:f.write(plan)
verification=dict(status='PASS_SINGLE_TRAJECTORY_INTERFACE',formal_training_ready=False,frames=93,valid_full_chunks=85,RGB_images=372,optimizer_updates=0,backward=0,policy_rollouts=0,heldout_reads=0,historical_source_hashes_unchanged=True,GPU_ECC_increment=False,official_source_unmodified=subprocess.check_output(['git','-C',str(o/'vla-official'),'status','--short'],text=True).strip()=='',documented_changes=['correct K timestep-camera index','native224 horizontal-mirror geometry convention','same-state FK derived identity','variant-isolated stores','STOP/mask caller boundary repair','DDP-equivalent device transfer for no-DDP smoke'])
js('delivery_verification.json',verification)
for n in ['report.md','next_training_plan.md','ae_forward_results.json','official_forward_results.json','official_forward_results_stop_aligned.json','initialization_inventory.json','initialization_dependencies.json','same_state_fk_rotation_check.json','delivery_verification.json']:
 s=(o/n).read_text();s=s.replace(str(a.project),'$PROJECT_ROOT').replace(os.environ['DATA_ROOT'],'$DATA_ROOT')
 with (pub/n).open('x') as f:f.write(s)
with (pub/'README.md').open('x') as f:f.write('Run recover_sequence.py, extract_sequence.py, prepare_prismatic.py and official_forward.py (both --stop-contract modes) in fresh --out under the data volume. Set RLA_GPU_UUID and CUDA_VISIBLE_DEVICES to the authorized checked UUID, PYTHONDONTWRITEBYTECODE=1, and all HF/Torch/XDG/TMP caches to data-disk paths. Renderer uses existing policy CPU environment; AE uses existing sidecar environment. Pin commits in report. Local frames, sidecars and resolved initialization are deliberately not committed. All scripts refuse conflicting final outputs; retain failed-attempt logs. No optimizer or backward calls.\n')
print('DELIVERY_PASS',flush=True)
