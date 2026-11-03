"""Freeze source scope and acceptance before candidate image comparison."""
import csv,json,hashlib,shutil,subprocess
from pathlib import Path
ROOT=Path('/data/lyc/projects/rla-wm')
RUN=ROOT/'experiments/rla_official_ae_paired_rerender/run_20261103_01'
OLD=ROOT/'experiments/rla_official_background_hue/run_20261102_01'
OFFICIAL=ROOT/'worktrees/rla-official-0fc6e97'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def csvout(p,rows):
 with open(p,'w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 rows=list(csv.DictReader(open(OLD/'audit/trajectory_manifest.csv')))
 assert len(rows)==1693 and sum(r['split']=='train' for r in rows)==1653
 assert subprocess.check_output(['git','-C',str(OFFICIAL),'rev-parse','HEAD'],text=True).strip()=='0fc6e97212442ebccffa06f1d42f5481e5f22950'
 selected=[];manifest=[]
 for suite in ['libero_spatial','libero_object','libero_goal','libero_10']:
  selected.append(min((r for r in rows if r['suite']==suite and r['split']=='train'),key=lambda r:int(r['trajectory_id'])))
  disk=json.loads((ROOT/'data/libero_converted'/suite/'index.json').read_text())['trajectories']
  assert sorted(disk)==sorted(r['trajectory_id'] for r in rows if r['suite']==suite)
 for r in rows:
  p=Path(r['source'])/'metadata.h5';assert p.exists()
  manifest.append(dict(**r,metadata_sha256=sha(p),rlds_status='NOT_CHECKED',hdf5_status='UNKNOWN',state_status='NOT_RECOVERED',clean_reproduction='NOT_RUN',paired_status='NOT_RUN'))
 csvout(RUN/'source_mapping.csv',manifest)
 dump(RUN/'frozen_protocol.json',dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
  rla_code_commit='0fc6e97212442ebccffa06f1d42f5481e5f22950',converted_repo='xyzhang368/RLA-WM',converted_revision='5fc55b89ba14e85c1d8ac064fd8a6eb11b5329f8',
  rlds_repo='openvla/modified_libero_rlds',rlds_revision='6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551',
  hdf5_repo='yifengzhu-hf/LIBERO-datasets',hdf5_revision='97773100c1474cd0d686ebd173cc0e4fd5442466',
  source_manifest_sha256=sha(OLD/'audit/trajectory_manifest.csv'),train=1653,val=40,pilot_selection='numeric minimum training ID per suite, no replacement',pilots=selected,
  acceptance=dict(rlds_mapping='Full action/joint/eef position arrays plus sampled original pixels in both views; full pixels for pilots; enumeration checked, not assumed',
    converted_vs_rlds_max_rgb_error=3,converted_vs_rlds_mean_rgb_error=1.0,
    replay_vs_rlds_joint_max_abs=0.002,replay_vs_rlds_eef_max_m=0.002,
    replay_vs_video_rgb_mean_abs_max=8,replay_vs_video_rgb_p99_abs_max=40,
    replay_vs_video_note='Per camera and full trajectory; image metrics alone insufficient. Review robot, task objects, occlusion and event timing on complete comparison video/fixed trace. Any clear geometry/time mismatch blocks, no deletion/shift.',
    same_state_pair='Exact flattened state/time before and after render, unchanged physical arrays and camera pose. No step between renders.',
    source_state_vs_replay='Distinct provenance. Never treat RLDS 8D observation.state as full state.'),
  hue=dict(range=[-.25,.25],train_seed=20261103,val_seed=20261104,per_trajectory_fixed=True,shared_cameras=True,S_V_unchanged=True),
  limits=dict(initial_replay_candidates_per_pilot=1,alternate_environment_search=0,unconfirmed_sources_blocked=True,batch_only_after_all_four_pass=True),
  forbidden=['SAM','AE/DINO/policy inference','training','backward','policy rollout','split changes'],
  disk_free_bytes=shutil.disk_usage(RUN).free))
 dump(RUN/'sam_pause.json',dict(status='PAUSED_BY_USER',last_commit='5a0fc76860e7fc53394916548d2c144d702de10e',new_segmentation_forwards=0))
 evidence=[OFFICIAL/'rebuttal/exp2_vla_adapter/README.md',OFFICIAL/'rebuttal/exp2_vla_adapter/configs/rla/16x64_libero.yaml',
  OFFICIAL/'rebuttal/exp2_vla_adapter/libero_prep/convert_libero_to_trajectory.py',OFFICIAL/'rebuttal/exp2_vla_adapter/setup/download_vla_adapter_assets.sh',
  OFFICIAL/'rebuttal/DATA.md',OLD/'delivery/report.md',ROOT/'scripts/pilot/download_libero_converted.sh']
 dump(RUN/'initial_evidence.json',[dict(path=str(p),sha256=sha(p)) for p in evidence])
 print(json.dumps(dict(selected=[(x['suite'],x['trajectory_id'],x['task']) for x in selected],scope=len(rows))))
if __name__=='__main__':main()
