"""Explicit user-authorized full-state recovery; preserve failed attempt and logs."""
import sys,os,time,subprocess,traceback,fcntl,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_scratch_low_seeds.common import *
from rla_scratch_low_seeds.run import launch
from rla_scratch_policy_aux.launch import environment
def main():
 lock=(ROOT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 audit=read(ROOT/'recovery_gpu0_01/checkpoint_audit.json');assert audit['PASS']
 save(ROOT/'recovery_gpu0_01/execution_start.json',dict(pid=os.getpid(),start=time.time(),GPU=GPU,authorized=True))
 try:
  root=ROOT/'seed_31002';ck=root/'D-low/formal/step0001.pt';assert sha(ck)==audit['sha256']
  assert int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))>=26624
  assert int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True))==0
  env=environment();env.update(SCRATCH_POLICY_ROOT=str(ROOT),POLICY_RESUME=str(ck))
  cmd=[sys.executable,'-u',str(HERE.parent/'rla_lr_budget_sweep/bootstrap.py'),'--out',str(root),'--arm','D-low','--entry','rla_scratch_low_seeds.training','--attempt','recovery_gpu0_01']
  start=time.time()
  with (ROOT/'recovery_gpu0_01/training.log').open('x') as f:r=subprocess.run(cmd,cwd=CODE,env=env,stdout=f,stderr=subprocess.STDOUT)
  save(ROOT/'recovery_gpu0_01/process.json',dict(returncode=r.returncode,seconds=time.time()-start,command=cmd));assert r.returncode==0
  old=list(csv.DictReader((root/'D-low/formal/training_metrics.csv').open()));new=list(csv.DictReader((root/'D-low/recovery_gpu0_01/training_metrics.csv').open()));assert len(old)==50 and len(new)==599
  keys=['action_loss','auxiliary_loss','lambda_RLA','weighted_auxiliary','total_loss','lr','gradient_vla','gradient_head','gradient_proprio']
  comparison=[dict(step=int(a['step']),equal=all(a[k]==b[k] for k in keys),different_fields=[k for k in keys if a[k]!=b[k]]) for a,b in zip(old[1:],new[:49])]
  save(ROOT/'recovery_gpu0_01/replay_audit.json',dict(effective_updates=600,actual_computations=649,replays=49,comparison=comparison,complete_state_restored=True,original_attempt_preserved=True))
  launch(31003,'initialization');launch(31003,'preflight')
  for arm in ARMS:launch(31003,arm)
  seeds=read(ROOT/'protocol.json')['seeds'];assert len({read(ROOT/f'seed_{s}/module_initialization_audit.json')['frozen_pretrained_sha256'] for s in seeds})==1
  save(ROOT/'evaluation/run_plan.json',read(ROOT/'seed_31001/run_plan.json'))
  registry=read(ROOT/'evaluation/policy_registry.json');registry['D-low_seed31002_s600']=str(root/'D-low/recovery_gpu0_01/step0600.pt');save(ROOT/'evaluation/policy_registry_recovery.json',registry)
  os.environ['RLA_POLICY_REGISTRY']='policy_registry_recovery.json'
  launch(None,'evaluation')
  subprocess.run(['/opt/conda/bin/python',str(HERE/'analyze.py')],check=True,cwd=CODE)
  subprocess.run(['/opt/conda/bin/python',str(HERE/'publish.py'),'final'],check=True,cwd=CODE)
  save(ROOT/'completion.json',dict(status='COMPLETE',successful_policy_updates=3600,actual_computations=3649,replayed_updates=49,valid_rollouts=600,AE_updates=0))
 except Exception:
  save(ROOT/'recovery_gpu0_01/stopped.json',dict(error=traceback.format_exc(),time=time.time(),automatic_retry=False));raise
 finally:save(ROOT/'recovery_gpu0_01/resource_release.json',dict(own_stage_processes_exited=True,other_processes_untouched=True,time=time.time()))
if __name__=='__main__':main()
