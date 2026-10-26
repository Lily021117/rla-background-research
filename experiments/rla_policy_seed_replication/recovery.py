"""User-authorized complete-state GPU0 recovery; immutable historical attempts."""
import argparse,json,csv,hashlib,subprocess,os
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();o=a.out
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
specs=[(20261029,'D-center',100),(20261030,'C',300),(20261030,'D-center',0)];configs=[];audits=[]
for seed,arm,step in specs:
 root=o/f'seed_{seed}';source=root/arm/'formal'/f'step{step:04}.pt' if step else None
 if source:
  ck=torch.load(source,map_location='cpu');assert ck['global_step']==step and ck['next_physical_sample']==step*8 and ck['scheduler']['last_epoch']==step
  assert ck['configuration']==json.loads((root/'run_plan.json').read_text())
  assert all(int(x['step'])==step for x in ck['optimizer']['state'].values())
  assert all(k in ck for k in ['parameters','torch_rng','cuda_rng','python_rng','numpy_rng','AMP_scaler']);assert ck['AMP_scaler'] is None
  rs=list(csv.DictReader((root/arm/'formal/training_metrics.csv').open()));last=int(rs[-1]['step'])
  audits.append(dict(seed=seed,method=arm,checkpoint=str(source),sha256=sha(source),restored_step=step,last_recorded_successful_update=last,replayed_updates=last-step,full_state_verified=True));del ck
 attempt=f'recovery_gpu0_s{step:04}' if step else 'formal';label=arm+'_gpu0_recovery'
 config=dict(seed=seed,stage=arm,attempt=attempt,label=label,checkpoint=str(source) if source else None,user_authorized=True,configuration_unchanged=True)
 path=root/(label+'_config.json');save(path,config);configs.append(path)
for seed in [20261029,20261030]:
 root=o/f'seed_{seed}'
 attempts={'C':['C/formal'],'D-center':['D-center/formal','D-center/recovery_gpu0_s0100']} if seed==20261029 else {'C':['C/formal','C/recovery_gpu0_s0300'],'D-center':['D-center/formal']}
 save(root/'effective_execution_attempts.json',attempts)
 save(root/'policy_registry_gpu0_recovery.json',{arm+'_s600':str(root/paths[-1]/'step0600.pt') for arm,paths in attempts.items()})
 save(root/'evaluate_gpu0_config.json',dict(seed=seed,stage='evaluate',attempt='formal',label='evaluate',checkpoint=None,user_authorized=True))
save(o/'hardware_recovery_audit.json',dict(reason='GPU1 new uncorrectable SRAM parity/SM increment; cuBLAS backward failure',GPU1_before=dict(volatile_parity=0,aggregate_parity=1,aggregate_SM=1),GPU1_after=dict(volatile_parity=1,aggregate_parity=2,aggregate_SM=2),GPU1_disabled_for_remaining_task=True,GPU0_ECC_unchanged=True,user_authorized_GPU0_serial=True,failed_backward_attempts=1,unpersisted_successful_updates_to_replay=206,interrupted_GPU1_migration_optimizer_updates=0,restores=audits))
here=Path(__file__).parent
for path in configs:
 c=json.loads(path.read_text());print('RECOVERY_STAGE',c['seed'],c['stage'],flush=True)
 r=subprocess.run(['python',str(here/'launch.py'),'--out',str(o),'--seed',str(c['seed']),'--stage',c['stage'],'--managed-worker','--recovery-config',str(path)])
 if r.returncode:raise SystemExit(r.returncode)
for seed in [20261029,20261030]:
 r=subprocess.run(['python',str(here/'launch.py'),'--out',str(o),'--seed',str(seed),'--stage','evaluate','--recovery-config',str(o/f'seed_{seed}/evaluate_gpu0_config.json')])
 if r.returncode:raise SystemExit(r.returncode)
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
with (o/'analysis.log').open('x') as f:r=subprocess.run(['/data/lyc/projects/vla-adapter-ref/.venv-cpu/bin/python',str(here/'analyze.py'),'--out',str(o)],env=env,stdout=f,stderr=subprocess.STDOUT)
raise SystemExit(r.returncode)
