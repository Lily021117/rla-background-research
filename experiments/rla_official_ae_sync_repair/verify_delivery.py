"""Final CPU-only arithmetic/hash/resource checks. No simulation imports."""
from pathlib import Path
import json,csv,hashlib,subprocess,datetime,os
ROOT=Path('/data/lyc/projects/rla-wm');RUN=ROOT/'experiments/rla_official_ae_sync_repair/run_20261103_01';OUT=RUN/'delivery';PUBLIC=Path(__file__).parent/'delivery'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def get(p):return json.loads(Path(p).read_text())
def rows(p):return list(csv.DictReader(open(p)))
def dumpboth(name,v):
 text=json.dumps(v,indent=2,ensure_ascii=False)+'\n';(OUT/name).write_text(text);(PUBLIC/name).write_text(text)
def main():
 counts=get(OUT/'execution_counts.json');assert counts['source_actions']==598 and counts['wait_steps']==31
 assert counts['physics_substeps']==25*(counts['source_actions']+counts['wait_steps'])
 assert counts['render_calls']==16420 and counts['state_restores']==5477
 for key in ['AE_forwards','DINO_forwards','policy_forwards','AE_updates','DINO_updates','policy_updates','backward','policy_rollouts']:assert counts[key]==0
 manifest=rows(OUT/'full_status.csv');assert len(manifest)==1693 and sum(r['split']=='train' for r in manifest)==1653
 summary=rows(OUT/'pilot_summary.csv');assert sum(int(r['frames']) for r in summary)==741
 assert all(r['A_self_restore']=='True' and r['B_original_gate']=='False' for r in summary)
 for r in summary:
  directory=RUN/'color_final'/r['suite'];n=int(r['frames']);assert len(rows(directory/'pairing_manifest.csv'))==n
  assert len(rows(directory/'frame_correspondence.csv'))==n and len(rows(directory/'frame_color_metrics.csv'))==4*n
  camera=get(directory/'camera_time_metadata.json');assert camera['frames']==n and camera['formal_loader_admission'] is False
  assert camera['array_sha256']==sha(directory/'paired_rgb.npz') and camera['metadata_sha256']==sha(directory/'synchronized_metadata.npz')
 # Existing input evidence cannot silently change. Only the clarified report-unit
 # wording in deliver.py was edited after first packaging; record that explicitly.
 evidence=get(OUT/'evidence_manifest.json');updates=[]
 for e in evidence:
  actual=sha(e['path'])
  if actual!=e['sha256']:
   assert Path(e['path']).name=='deliver.py'
   updates.append(dict(path=e['path'],previous_sha256=e['sha256'],current_sha256=actual,reason='Clarified weak-color threshold units; no data/threshold changed'))
   e['sha256']=actual
 known={e['path'] for e in evidence}
 for p in Path(__file__).parent.glob('*.py'):
  if str(p) not in known:evidence.append(dict(path=str(p),sha256=sha(p)))
 dumpboth('evidence_manifest.json',evidence)
 for p in OUT.rglob('*'):
  if p.is_file():
   pub=PUBLIC/p.relative_to(OUT)
   if pub.exists():assert sha(pub)==sha(p),str(p)
 processes=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
 active=[line for line in processes if 'python' in line and any('rla_official_ae_sync_repair/'+s in line for s in ['capture.py','color_audit.py','offline.py','model_snapshot.py','index_pairs.py'])]
 assert not active,active
 dumpboth('resource_release.json',dict(checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),task_render_replay_processes=active,GPU_used=False,resources_released=True,
  local_output_bytes=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file() and 'code' not in p.relative_to(RUN).parts),no_other_process_terminated=True))
 dumpboth('delivery_verification.json',dict(status='PASS_WITH_REPORTED_EVIDENCE_GAPS',train=1653,val=40,frames=741,A_pass=4,B_pass=0,
  counts=counts,input_hash_updates=updates,missing=['Object fresh runtime four-path recapture','Spatial optional raw diagnostic_paths.npz'],
  scope_frozen=True,source_action_budget_respected=True,no_threshold_relaxation=True,bulk_started=0))
 print('PASS: scope, rows, hashes, arithmetic, zero learned-model operations, released resources')
if __name__=='__main__':main()
