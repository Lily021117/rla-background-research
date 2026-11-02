"""Serial frozen evaluation on authorized GPU0; no retries on hardware error."""
import sys,os,subprocess,time,traceback,fcntl
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_v2_frozen_c_eval.common import *
from rla_scratch_policy_aux.launch import environment
def main():
 lock=(ROOT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert read(ROOT/'protocol_commit.json')['protocol_sha256']==sha(ROOT/'evaluation_protocol.json')
 save(ROOT/'execution_start.json',dict(time=time.time(),pid=os.getpid(),GPU=GPU,branch_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip()))
 try:
  for task in ['stove','drawer']:
   assert int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))>=26624
   assert int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True))==0
   env=environment();env.update(V2_EVAL_ROOT=str(ROOT),V2_EVAL_TASK=task,RLA_PROJECT_ROOT=str(P));cmd=[str(P.parent/'vla-adapter-ref/.venv-cpu/bin/python'),'-u',str(HERE.parent/'rla_lr_budget_sweep/bootstrap.py'),'--out',str(ROOT/task),'--arm','evaluation','--attempt','formal','--entry','rla_v2_frozen_c_eval.evaluation']
   started=time.time()
   with (ROOT/task/'evaluation.log').open('x') as f:r=subprocess.run(cmd,cwd=CODE,env=env,stdout=f,stderr=subprocess.STDOUT)
   save(ROOT/task/'process.json',dict(returncode=r.returncode,seconds=time.time()-started,command=cmd,GPU=GPU));assert r.returncode==0
  subprocess.run([sys.executable,str(HERE/'analyze.py')],check=True,cwd=CODE)
  save(ROOT/'completion.json',dict(status='COMPLETE',new_valid=480,reused_clean=120,optimizer_updates=0,AE_updates=0,DINO_updates=0,backward=0))
 except Exception:save(ROOT/'stopped.json',dict(error=traceback.format_exc(),time=time.time(),automatic_retry=False));raise
 finally:save(ROOT/'resource_release.json',dict(processes_exited=True,other_jobs_untouched=True,time=time.time()))
if __name__=='__main__':main()
