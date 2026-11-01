"""Serial authorized GPU0 pipeline; stop on any infrastructure/numerical failure."""
import os,sys,subprocess,time,traceback,fcntl
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_drawer_baseline_gap.common import *
from rla_scratch_policy_aux.launch import environment
PY=P.parent/'vla-adapter-ref/.venv-cpu/bin/python'
def launch(stage,seed=None):
 assert int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))>=26624
 assert int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True))==0
 env=environment(stage=='extract');env.update(DRAWER_GAP_ROOT=str(ROOT),NEW_TASK_RUN=str(ROOT),NEW_TASK_RESULT_ROOT=str(ROOT/'evaluation'))
 if stage=='extract':
  root=ROOT;cmd=[str(P/'.venv-sidecar/bin/python'),'-u',str(HERE/'extract.py'),'--project',str(P),'--dataset',str(DATA),'--out',str(ROOT/'sidecars')]
 else:
  root=ROOT/f'seed_{seed}' if seed else ROOT/'evaluation';entry='initialize' if stage=='initialization' else 'evaluation' if stage=='evaluation' else 'training'
  if stage=='initialization':env.update(RLA_BOOTSTRAP_PLAN='planned_run_plan.json',RLA_BOOTSTRAP_SEED=str(seed),RLA_FULL_SEED_RECIPE='1')
  cmd=[str(PY),'-u',str(HERE.parent/'rla_lr_budget_sweep/bootstrap.py'),'--out',str(root),'--arm',stage,'--attempt','formal','--entry','rla_drawer_baseline_gap.'+entry]
 start=time.time()
 with (root/(stage+'.log')).open('x') as f:r=subprocess.run(cmd,cwd=CODE,env=env,stdout=f,stderr=subprocess.STDOUT)
 save(root/(stage+'_process.json'),dict(returncode=r.returncode,seconds=time.time()-start,command=cmd,GPU=GPU));assert r.returncode==0,(stage,seed,r.returncode)
def main():
 lock=(ROOT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert read(ROOT/'protocol_commit.json')['remote_verified']
 save(ROOT/'execution_start.json',dict(pid=os.getpid(),unix=time.time(),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip(),protocol_sha256=sha(ROOT/'protocol.json')))
 try:
  launch('extract')
  assert read(ROOT/'sidecars/sidecar_alignment_audit.json')['status']=='PASS'
  for seed in SEEDS:
   launch('initialization',seed)
   launch('preflight-standard',seed);launch('preflight-rla',seed)
   for method in METHODS:launch(method,seed)
  assert len({read(ROOT/f'seed_{s}/initialization_audit.json')['frozen_pretrained_sha256'] for s in SEEDS})==1
  save(ROOT/'evaluation/run_plan.json',read(ROOT/f'seed_{SEEDS[0]}/run_plan.json'))
  launch('evaluation')
  subprocess.run([str(PY),str(HERE/'analyze.py')],check=True,cwd=CODE)
  subprocess.run([str(PY),str(HERE/'publish.py'),'final'],check=True,cwd=CODE)
  save(ROOT/'completion.json',dict(status='COMPLETE',successful_policy_updates=10800,valid_trials=len(read(ROOT/'evaluation/per_trial_results.json')),AE_updates=0,DINO_updates=0,finished_unix=time.time()))
 except Exception:
  save(ROOT/'stopped.json',dict(error=traceback.format_exc(),unix=time.time(),automatic_retry=False));raise
 finally:save(ROOT/'resource_release.json',dict(stage_processes_exited=True,other_processes_untouched=True,unix=time.time()))
if __name__=='__main__':main()
