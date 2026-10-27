"""One authorized full-state recovery, then two frozen evaluations; no new arms."""
import os,sys,json,subprocess,fcntl
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_sampling_stability.d_completion import read,save,sha,analyze,publish,REPO,HERE
import torch
out=Path(sys.argv[1]);lock=(out/'gpu0_recovery.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
gpu='GPU-205a3f5a-d91b-6385-c588-4e4cf3505f7e'
assert int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))>=26624
assert int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True))==0
root=out/'seed_20261102';ckpath=root/'D-center/formal/step0300.pt';ck=torch.load(ckpath,map_location='cpu')
assert ck['global_step']==300 and ck['next_physical_sample']==2400 and ck['scheduler']['last_epoch']==300
assert ck['configuration']==read(root/'run_plan.json') and ck['activation_checkpointing'] and ck['AMP_scaler'] is None
assert all(int(v['step'])==300 for v in ck['optimizer']['state'].values())
assert all(k in ck for k in ['python_rng','numpy_rng','torch_rng','cuda_rng'])
del ck
save(out/'gpu0_recovery_audit.json',dict(user_authorized=True,checkpoint=str(ckpath),sha256=sha(ckpath),restored_step=300,failed_attempt_logged_updates=400,replayed_updates=100,remaining_actual_updates=300,full_model_optimizer_scheduler_RNG_sampler=True,AMP='bfloat16 no scaler',GPU_UUID=gpu,configuration_unchanged=True,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()))
jobs=[(20261102,'D-center','gpu0_resume300',str(ckpath)),(20261101,'evaluate','gpu0_evaluate',None),(20261102,'evaluate','gpu0_evaluate',None)]
for seed,stage,attempt,checkpoint in jobs:
 r=out/f'seed_{seed}'
 if stage=='evaluate':
  final=r/'D-center'/('gpu0_resume300' if seed==20261102 else 'formal')/'step0600.pt'
  save(r/'policy_registry_gpu0_recovery.json',{'D-center_s600':str(final)})
 cfg=r/(attempt+'_config.json');save(cfg,dict(seed=seed,stage=stage,attempt=attempt,label=attempt,user_authorized=True,checkpoint=checkpoint))
 print('GPU0_START',seed,stage,flush=True)
 subprocess.run([sys.executable,str(HERE.parent/'rla_policy_seed_replication/launch.py'),'--out',str(out),'--seed',str(seed),'--stage',stage,'--recovery-config',str(cfg)],cwd=REPO,check=True)
analyze(out);publish(out)
print('GPU0_RECOVERY_AND_ALL_DELIVERIES_COMPLETE',flush=True)
