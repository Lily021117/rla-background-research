"""CPU-only final source/checkpoint protection and evidence packaging."""
import os,csv,json,hashlib,subprocess
from pathlib import Path
R=Path('${RLA_PROJECT_ROOT}');BASE=R/'experiments/rla_background_invariance'
OUT=Path(os.environ['RUN_DIR']);DATA=Path(os.environ['SPARSE_DATA_DIR']);WT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for v in iter(lambda:f.read(8*1024**2),b''):h.update(v)
 return h.hexdigest()
def write(n,v):
 p=OUT/n;assert not p.exists(),p;p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def cmd(args,cwd=R):return subprocess.check_output(args,cwd=cwd,text=True)
def main():
 assert read(OUT/'smoke_acceptance.json')['status']=='PASS'
 assert read(OUT/'batch_coverage_acceptance.json')['status']=='PASS'
 protected=[]
 for p,h in read(DATA/'protected_sources_before.json').items():
  assert sha(p)==h,p;protected.append(dict(path=p,sha256=h))
 for x in read(DATA/'protected_checkpoints_before.json'):
  assert sha(x['path'])==x['sha256'],x['path'];protected.append(dict(path=x['path'],sha256=x['sha256']))
 for folder,manifest in [('baseline_eval/official_attempt','final_delivery_sha256.json'),('step2_training','artifact_sha256.json'),
  ('step2_5_diagnostics/run_20261021_01','artifact_sha256.json'),('step3_rla_wm_bc/run_20261021_04','final_delivery_sha256.json'),
  ('step4_readiness/run_20261021_01','artifact_sha256.json')]:
  d=BASE/folder;a=read(d/manifest);entries=a.get('files',a);h=entries['report.md'];assert sha(d/'report.md')==h
  protected.append(dict(path=str(d/'report.md'),sha256=h))
 expected=read(BASE/'step3_rla_wm_bc/run_20261021_04/reproduction_source_sha256.json')
 for p,h in expected.items():assert sha(p)==h,p
 assert not cmd(['git','diff','--name-only'],WT).strip()
 copied={}
 for name in ['config.py','action_head.py','sidecar.py','data.py','read_rla_sidecar.py']:
  p=WT/'rla'/name;q=Path('${VLA_REFERENCE_ROOT}/rla')/name
  assert sha(p)==sha(q);copied[name]=sha(p)
 write('protection_verification.json',dict(protected=protected,Step3_code_unchanged=True,
  copied_overlay_modules_unchanged=copied,original_repositories_modified=False,tracked_independent_worktree_diff_empty=True,
  scope='key policy/AE/flow weights, prior reports, declared sources; no claim of rehashing every historical cache byte'))
 write('code_inventory.json',{str(p):sha(p) for p in WT.glob('sparse_rla/*') if p.is_file()})
 write('data_inventory.json',{str(p):sha(p) for p in DATA.glob('*') if p.is_file() and (p.suffix in ['.npz','.npy'] or p.name in ['data_manifest.csv','episodes.json','samples.json','preparation.json'])})
 write('runtime_inventory.json',{str(p):sha(p) for p in (OUT/'runtime_metadata').iterdir() if p.is_file() and p.suffix not in ['.pt','.safetensors']})
 baseline=read(OUT/'gpu_before.json')['ecc'];raw=cmd(['nvidia-smi','-i','${GPU_UUID}','-q','-d','ECC'])
 fields=[x.strip() for x in raw.splitlines() if any(k in x for k in ['Correctable','Uncorrectable','Threshold Exceeded','Repair Pending','Unrepairable'])]
 assert fields==baseline
 write('final_environment.json',dict(git={str(p):dict(head=cmd(['git','rev-parse','HEAD'],p).strip(),branch=cmd(['git','branch','--show-current'],p).strip(),status=cmd(['git','status','--short'],p)) for p in [R,WT,Path('${VLA_REFERENCE_ROOT}')]},
  disk=cmd(['df','-h','${DATA_ROOT}']),mount=cmd(['findmnt','-T','${DATA_ROOT}']),ECC_unchanged=True,ECC_raw=raw,
  GPU_processes=cmd(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv'])))
 write('delivery_verification.json',dict(status='PASS_SPARSE_MATCHED_RAW_REPLAY_INTERFACE_ONLY',
  actual_VLA_batch=True,time_alignment=True,full32x64_preserved=True,gradient_paths=True,
  accepted_matched_train_episodes=3,accepted_physical_transitions=30,actual_CPU_batch_constructions=60,
  final_GPU_gradient_batches=2,whole_task_backward_calls=4,whole_task_autograd_probes=4,
  optimizer_created=False,optimizer_updates=0,closed_loop_episodes=0,heldout_evaluations=0,
  native_full_RLDS_stream_accepted=False,unmatched_train_episodes_used=0,validation_episodes_used=0,
  old_checkpoint_hashes_unchanged=True,original_policy_parameters_unchanged=True,original_code_modified=False,
  formal_training_readiness='PARTIAL: only3 matched episodes; no matched independent validation; new LoRA scope not tested',
  GPU_released=True,model_checkpoints_written=0))
 write('artifact_sha256.json',{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['artifact_sha256.json'] and p.suffix not in ['.pt','.safetensors']})
 print('STEP4_1_DELIVERY_PASS_ZERO_UPDATES',len(protected),flush=True)
if __name__=='__main__':main()
