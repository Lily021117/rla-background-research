import os,sys,subprocess,time,traceback,fcntl
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_nuisance_pilot.common import *
from rla_scratch_policy_aux.launch import environment
PY=P.parent/'vla-adapter-ref/.venv-cpu/bin/python'
def launch(stage):
 env=environment(stage=='offline_ae');env.update(NUISANCE_ROOT=str(ROOT),SCRATCH_POLICY_ROOT=str(ROOT));root=ROOT
 if stage in ['offline_policy','evaluate']:
  assert read(ROOT/'scoring_gate.json')['committed_before_scoring'];root=ROOT/'evaluation';cmd=[str(PY),'-u',str(HERE.parent/'rla_lr_budget_sweep/bootstrap.py'),'--out',str(root),'--arm',stage,'--entry','rla_nuisance_pilot.'+stage,'--attempt','formal']
 else:cmd=[str(P/'.venv-sidecar/bin/python' if stage=='offline_ae' else PY),'-u',str(HERE/(stage+'.py'))]
 if stage in ['offline_ae','offline_policy','evaluate']:
  assert int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))>=26624
  assert int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True))==0
 start=time.time()
 with (ROOT/(stage+'.log')).open('x') as f:r=subprocess.run(cmd,cwd=CODE,env=env,stdout=f,stderr=subprocess.STDOUT)
 save(ROOT/(stage+'_process.json'),dict(returncode=r.returncode,seconds=time.time()-start,command=cmd));assert r.returncode==0,stage
def main():
 lock=(ROOT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);save(ROOT/'execution_start.json',dict(pid=os.getpid(),time=time.time(),optimizer_updates=0))
 try:
  launch('verify_states')
  subprocess.run([str(PY),str(HERE/'publish.py'),'ready'],check=True,cwd=CODE)
  launch('render_offline');launch('offline_ae');launch('offline_policy');launch('evaluate');launch('analyze')
  subprocess.run([str(PY),str(HERE/'publish.py'),'final'],check=True,cwd=CODE)
  save(ROOT/'completion.json',dict(status='COMPLETE',valid_new_rollouts=900,reused_clean=150,optimizer_updates=0,AE_updates=0,DINO_updates=0,backward=0,time=time.time()))
 except Exception:save(ROOT/'stopped.json',dict(error=traceback.format_exc(),time=time.time(),automatic_retry=False));raise
 finally:save(ROOT/'resource_release.json',dict(all_own_subprocesses_exited=True,others_untouched=True,time=time.time()))
if __name__=='__main__':main()
