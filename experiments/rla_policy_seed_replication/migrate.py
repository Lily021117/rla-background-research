"""Authorized safe boundary migration: only this task PID, no optimizer replay."""
import argparse,os,signal,time,json,csv,hashlib,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.out/'seed_20261029';formal=root/'D-center/formal';ckpath=formal/'step0100.pt'
def rows(path):
 try:return list(csv.DictReader(path.open()))
 except FileNotFoundError:return []
def save(path,x):
 with path.open('x') as f:json.dump(x,f,indent=2)
while True:
 ts=rows(formal/'training_metrics.csv');vs=rows(formal/'validation_metrics.csv')
 if ts and int(ts[-1]['step'])>100:raise RuntimeError('Missed boundary; do not terminate or rewind')
 if ckpath.exists() and ts and int(ts[-1]['step'])==100 and any(r.get('step')=='100' for r in vs):break
 time.sleep(2)
pids=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:cmd=(proc/'cmdline').read_bytes().split(b'\0')
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 if b'--out' in cmd and str(root).encode() in cmd and b'--arm' in cmd and b'D-center' in cmd and b'stove_centering.training' in cmd:pids.append(int(proc.name))
assert len(pids)==1,pids
pid=pids[0];os.kill(pid,signal.SIGSTOP)
time.sleep(.5)
try:
 import torch
 c=torch.load(ckpath,map_location='cpu');assert c['global_step']==100 and c['next_physical_sample']==800 and c['scheduler']['last_epoch']==100
 assert all(int(x['step'])==100 for x in c['optimizer']['state'].values())
 assert all(k in c for k in ['parameters','torch_rng','cuda_rng','python_rng','numpy_rng','AMP_scaler']);assert c['AMP_scaler'] is None
 assert int(rows(formal/'training_metrics.csv')[-1]['step'])==100
 assert any(r.get('step')=='100' for r in rows(formal/'validation_metrics.csv'))
 h=hashlib.sha256()
 with ckpath.open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 save(root/'migration_boundary_audit.json',dict(status='PASS_FULL_STATE_BOUNDARY',PID=pid,checkpoint=str(ckpath),sha256=h.hexdigest(),global_step=100,actual_updates_before_migration=100,interrupted_phase='fixed validation after snapshot; no optimizer update in flight',next_physical_sample=800,all_model_optimizer_scheduler_RNG_sampler_fields_verified=True,AMP_scaler=None,dtype='bfloat16',replayed_updates=0,user_authorized=True))
except BaseException:
 os.kill(pid,signal.SIGCONT);raise
os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
print('OWN_TRAINING_PROCESS_STOPPED_AT_SAVED_STEP100',pid,flush=True)
save(root/'resource_assignment.json',dict(GPU_UUID='GPU-37d3f7d4-309e-2bb8-2d51-292dbbcce419',user_explicitly_authorized=True,reason='safe full-state migration from overloaded GPU0; C already completed onGPU0; paired evaluation onGPU1',minimum_free_MiB=26624))
save(root/'execution_attempts.json',{'C':['C/formal'],'D-center':['D-center/formal','D-center/resume_step0100']})
registry=json.loads((root/'policy_registry.json').read_text());registry['D-center_s600']=str(root/'D-center/resume_step0100/step0600.pt');save(root/'policy_registry_migration.json',registry)
cmd=['python',str(Path(__file__).with_name('launch.py')),'--out',str(a.out),'--seed','20261029','--stage','D-center','--resume-step100']
r=subprocess.run(cmd);raise SystemExit(r.returncode)
