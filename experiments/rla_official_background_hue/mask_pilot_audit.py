"""CPU-only pilot accounting, review register and original-loader integration.

Review decisions are explicit AI visual observations, not human truth or metrics
of segmentation accuracy. Full sequences remain quarantined.
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np, cv2, decord
from PIL import Image, ImageDraw
from audit import sha, write_json, write_csv
from paired_hue import VerifiedMaskProvider, PairedHueReader, array_hash

def final_dir(run, video):
    if video=='libero_10_000011_agentview_camera': return run/'v2_recovered_front'/video
    if video=='libero_10_000011_wrist_camera': return run/'v2_tail'/video
    return run/'v2'/video

# These bounded review judgments do not claim full-video semantic certification.
WRIST = {
 'libero_spatial_000010': ('NEEDS_CORRECTION','t66 plate rim suspected contamination; fragmented first/middle keyframes', [66], [16,33]),
 'libero_spatial_000011': ('NEEDS_CORRECTION','t27 plate interior incorrectly included; long near-empty segments despite visible table', [27,41,55,69,83], []),
 'libero_object_000010': ('NEEDS_CORRECTION','t0/23/47 mostly empty; table recovered later; early coverage missing', [0,23,47], [94,118]),
 'libero_object_000011': ('FAILED_TRACKING','Most fixed frames empty despite visible table; only middle prompt fragments', [0,27,54,108,135,163], []),
 'libero_goal_000010': ('FAILED_TRACKING','Visible table mostly unsegmented throughout fixed checks', [0,18,37,56,75,94,113], []),
 'libero_goal_000011': ('FAILED_TRACKING','Early/late table missed; t40 possible stovetop boundary inclusion', [0,13,26,40,66,80], []),
 'libero_10_000010': ('NEEDS_CORRECTION','First and middle prompt frames fragmented; later fixed frames have useful conservative table', [0,191], [191,254]),
 'libero_10_000011': ('NEEDS_CORRECTION','t0/109 prompt frames fragmented; remaining fixed checks useful conservative table', [0,109], [144,180]),
}

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--official',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--output-name',default='audit');a=p.parse_args()
 out=a.run/a.output_name;out.mkdir(exist_ok=False)
 protocol=json.loads((a.run/'selection_and_protocol.json').read_text()); identity=json.loads((a.run/'v2/model_identity.json').read_text())
 reviews=[]; metrics=[]; evidence=[]; all_rows=[]; abnormal=[]; timings=[]
 for v in protocol['videos']:
  vd=final_dir(a.run,v['id']); summary=json.loads((vd/'summary.json').read_text()); rows=list(csv.DictReader(open(vd/'frame_metrics.csv')))
  assert len(rows)==v['frames'] and sha(v['path'])==v['sha256']
  assert np.load(vd/'masks_224.npz')['masks'].shape==(v['frames'],224,224)
  base=f"{v['suite']}_{v['trajectory_id']}";status,note,issues,approved=WRIST[base]
  if v['camera']=='agentview_camera': status='PARTIAL_USABLE_FIXED_FRAMES'; note='Table visible in later fixed frames; prompt frame fragmented; no obvious interior foreground contamination observed in fixed sheets';issues=[0]
  reviews.append(dict(video_id=v['id'],status=status,notes=note,semantic_review='AI visual review, NOT human ground truth',
    fixed_frames_viewed=v['fixed_review_frames'],known_issue_frames=issues,prototype_pair_frames=approved,
    full_video_human_review=False,human_reviewed_frames=0,all_frames_semantically_verified=False,
    evidence=str(vd/'overlay_review.png'),evidence_sha256=sha(vd/'overlay_review.png')))
  f=lambda key:np.array([float(r[key]) for r in rows if r[key] not in ('','None')])
  coverage=f('coverage');nonempty=[r for r in rows if float(r['coverage'])>=.001]
  weak_nonempty=sum(r['weak_change']=='True' for r in nonempty)
  grayweak=sum(r['weak_change']=='True' and float(r['low_saturation_fraction'] or 0)>.5 for r in nonempty)
  metrics.append(dict(video_id=v['id'],camera=v['camera'],frames=v['frames'],status=status,
    mean_coverage=float(coverage.mean()),min_coverage=float(coverage.min()),max_coverage=float(coverage.max()),
    near_empty_frames=int((coverage<.001).sum()),near_full_frames=int((coverage>.95).sum()),
    mean_adjacent_mask_iou=float(f('adjacent_mask_iou').mean()) if len(f('adjacent_mask_iou')) else None,area_jump_frames=int((f('area_jump')>.15).sum()),
    weak_color_all_frames=sum(r['weak_change']=='True' for r in rows),weak_color_nonempty=weak_nonempty,
    weak_and_majority_low_saturation=grayweak,mean_low_saturation_fraction=float(f('low_saturation_fraction').mean()) if len(f('low_saturation_fraction')) else None,
    delta_h=summary['delta_h'],mean_masked_rgb_change=float(f('mean_rgb_change').mean()),full_overlay=str(vd/'overlay_full.mp4')))
  for r in rows: all_rows.append(dict(video_id=v['id'],**r))
  # Supplemental anomaly review index: first and last auto-flag, not replacements for fixed frames.
  flagged=[int(r['frame']) for r in rows if r['flags']]; selected=sorted(set(flagged[:1]+flagged[-1:]))
  vr=decord.VideoReader(v['path'],num_threads=1);masks=np.load(vd/'conservative_masks_native.npz')['masks']
  sheet=Image.new('RGB',(max(1,len(selected))*256,282),'#222222');draw=ImageDraw.Draw(sheet)
  for j,t in enumerate(selected):
   rgb=vr[t].asnumpy();m=masks[t].astype(bool);rgb[m]=(rgb[m]*.55+[0,114.75,0]).astype(np.uint8)
   sheet.paste(Image.fromarray(rgb),(j*256,26));draw.text((j*256+2,3),f"{v['camera']} t={t}",fill='white')
  sheet.save(out/f"abnormal_{v['id']}.png")
  abnormal.append(dict(video_id=v['id'],all_automatic_flags=str(vd/'abnormal_frames.json'),supplementary_frames=selected,
    supplemental_image=str(out/f"abnormal_{v['id']}.png")))
  timings.append(dict(video_id=v['id'],inference_seconds=summary['inference_seconds'] if 'recovered_front' not in str(vd) else None,
    export_plus_inference_seconds=summary['total_seconds'],timer_note='CPU export only; original interrupted inference time unknown' if 'recovered_front' in str(vd) else 'measured',
    peak_allocated_bytes=summary['peak_gpu_allocated_bytes']))
  for name in ['summary.json','frame_identity.json','masks_224.npz','raw_model_masks.npz','overlay_full.mp4']:
   evidence.append(dict(path=str(vd/name),sha256=sha(vd/name)))
 write_json(out/'semantic_review.json',reviews);write_csv(out/'coverage_by_video.csv',metrics);write_csv(out/'all_frame_metrics.csv',all_rows)
 write_json(out/'abnormal_review_index.json',abnormal);write_json(out/'timings.json',timings)
 write_json(out/'trajectory_status.json',[dict(trajectory=k,status=v[0],reason=v[1],complete_trajectory_training_admission=False) for k,v in WRIST.items()])
 write_json(out/'alignment_evidence.json',dict(native_resolution=[256,256],official_resolution=[224,224],decoder='decord native RGB',
   resize_rgb='cv2 INTER_LINEAR',resize_mask='INTER_NEAREST, then 3x3 erosion',additional_flip=False,
   video_hashes_unchanged=True,all_lengths_verified=True,source_frames=2576,sim_geometry_claim=False))
 # Only already-viewed fixed frames may be explicitly opted into the prototype loader.
 registry={}
 for v,r in zip(protocol['videos'],reviews):
  vd=final_dir(a.run,v['id']); hashes=json.loads((vd/'frame_identity.json').read_text())
  for t in r['prototype_pair_frames']:
   assert t in v['fixed_review_frames']
   registry[f"{v['suite']}/{v['trajectory_id']}/{t}/{v['camera']}"]=dict(source_kind='reviewed_model_segmentation',
    semantic_status='prototype_fixed_frame_pass',human_ground_truth=False,reviewer='AI visual inspection; human signoff absent',
    review_evidence_sha256=sha(out/'semantic_review.json'),model_checkpoint_sha256=identity['checkpoint_sha256'],prompt_sha256=identity['prompt_sha256'],
    alignment_status='reviewed_pass',alignment_evidence_sha256=sha(out/'alignment_evidence.json'),allowed_classes=['table','wall'],actual_classes=['table'],
    original_rgb_sha256=hashes[t]['rgb_224_hash'],path=str(vd/'masks_224.npz'),sha256=sha(vd/'masks_224.npz'),frame_index=t)
 write_json(out/'prototype_mask_registry.json',registry)
 # Actual official RGB reader, CPU only; no model or feature extraction.
 sys.path.insert(0,str(a.official));from src.datasets.trajectory_dataset import TrajectoryDataset
 from omegaconf import OmegaConf
 cfg=OmegaConf.to_container(OmegaConf.load(a.official/'rebuttal/exp2_vla_adapter/configs/rla/16x64_libero.yaml'),resolve=True)
 args=cfg['dataset']['args'];args.update(root='/data/lyc/projects/rla-wm/data/libero_converted',use_mp_for_indexing=False,
   trajectory_info_cache_file=str(a.previous/'official_train_index.json'))
 ds=TrajectoryDataset(**args);results=[]
 for tr in protocol['trajectories']:
  base=f"{tr['suite']}_{tr['trajectory_id']}";chosen=WRIST[base][3];frames=chosen or [0,1]
  t=next(t for t in ds.trajectory_info if t['group_name']==tr['suite'] and t['traj_id']==tr['trajectory_id'])
  h=dict(version=1,dataset_idx=t['dataset_idx'],group_name=tr['suite'],traj_id=tr['trajectory_id'],frame_id=frames[0],
   sampled_horizon=frames[1]-frames[0]+1,feasible_horizon=frames[1]-frames[0]+1,camera_keys=['agentview_camera','wrist_camera'],
   rgb_variant='base',interaction_frame_indices=None,dataset_root=ds.datasets[t['dataset_idx']].root)
  assert 2<=h['sampled_horizon']<=64 and frames[-1]<int(tr['frames'])-1
  x=ds.getitem_from_handle(h,apply_augmentation=False)
  off=PairedHueReader(ds,VerifiedMaskProvider(registry),enabled=False).read(h,0,'train')
  assert np.array_equal(x['rgbs'],off['x_prime']['rgbs']);assert np.array_equal(x['frame_id'],frames)
  for optin in [False,True]:
   try: pair=PairedHueReader(ds,VerifiedMaskProvider(registry,allow_reviewed_model=optin)).read(h,0,'train')
   except RuntimeError:
    assert not optin or not chosen
    results.append(dict(trajectory=base,prototype_optin=optin,result='EXPECTED_BLOCK',frames=frames));continue
   assert chosen and optin
   repeat=PairedHueReader(ds,VerifiedMaskProvider(registry,allow_reviewed_model=True)).read(h,0,'train')
   assert np.array_equal(pair['x_prime']['rgbs'],repeat['x_prime']['rgbs'])
   rgb=np.moveaxis(pair['x']['rgbs'],2,-1);aug=np.moveaxis(pair['x_prime']['rgbs'],2,-1);m=pair['background_masks']
   assert np.array_equal(rgb[~m],aug[~m]);assert np.array_equal(x['rgbs'],pair['x']['rgbs'])
   unchanged=[]
   for key in pair['x']:
    if key!='rgbs': assert pair['x_prime'][key] is pair['x'][key];unchanged.append(key)
   results.append(dict(trajectory=base,prototype_optin=optin,result='PASS_FIXED_PAIR_ONLY',frames=frames,delta_h=pair['provenance']['delta_h'],
     original_rgb_hash=array_hash(rgb),augmented_rgb_hash=array_hash(aug),outside_mask_exact=True,reproducible=True,
     unchanged_metadata_fields=unchanged,cameras=h['camera_keys'],production_admission=False))
 write_json(out/'paired_loader_validation.json',results)
 counts1=json.loads((a.run/'v1/completion.json').read_text())['forward_counts'];counts2={k:0 for k in counts1}
 for v in protocol['videos']:
  vd=final_dir(a.run,v['id']);s=json.loads((vd/'summary.json').read_text())
  c=s['forward_counts'] if 'recovered_front' not in str(vd) else {k:v['frames'] for k in counts1}
  for k in counts2:counts2[k]+=c.get(k,0)
 storage=sum(f.stat().st_size for f in a.run.rglob('*') if f.is_file())
 write_json(out/'execution_summary.json',dict(physical_trajectories=8,physical_frames=1288,camera_videos=16,unique_camera_frames=2576,
   intentional_segmentation_passes=2,forward_v1=counts1,forward_v2=counts2,forward_total={k:counts1[k]+counts2[k] for k in counts1},
   counts_inferred_not_hook_persisted={'video':'libero_10_000011_agentview_camera','each_component':218},
   segmentation_infrastructure_replay_frames=0,interrupted_processes=1,completed_overlay_videos=32,
   known_v1_wall_seconds=json.loads((a.run/'v1/completion.json').read_text())['seconds'],known_v2_inference_seconds=sum(t['inference_seconds'] or 0 for t in timings),
   unknown_inference_timers=1,peak_gpu_allocated_bytes=max(t['peak_allocated_bytes'] for t in timings),artifact_bytes_at_audit=storage,
   AE_forwards=0,DINO_forwards=0,policy_forwards=0,AE_updates=0,DINO_updates=0,policy_updates=0,backward=0,rollouts=0,simulation_replays=0))
 evidence += [dict(path=str(a.run/f),sha256=sha(a.run/f)) for f in ['selection_and_protocol.json','prompts_v1.json','prompts_v2.json','correction_log_v2.json','infrastructure_incident.json']]
 write_json(out/'evidence_manifest.json',evidence)
 print(json.dumps({'paired_records':results,'counts':counts2},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
