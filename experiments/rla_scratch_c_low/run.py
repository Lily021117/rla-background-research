"""Only the two additional C-low trainings and200 new rollouts; fail closed."""
import sys,os,subprocess,time,traceback,fcntl
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_scratch_c_low.common import *
from rla_scratch_policy_aux.launch import environment
def launch(order):
 assert int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))>=26624
 assert int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True))==0
 root=ROOT/f'seed_{order}' if order else ROOT/'evaluation';arm='C-low' if order else 'evaluation'
 entry='rla_scratch_c_low.training' if order else 'rla_scratch_c_low.evaluation'
 cmd=[sys.executable,'-u',str(HERE.parent/'rla_lr_budget_sweep/bootstrap.py'),'--out',str(root),'--arm',arm,'--entry',entry,'--attempt','formal']
 env=environment();env['SCRATCH_POLICY_ROOT']=str(ROOT);start=time.time()
 with (root/(arm+'.log')).open('x') as f:r=subprocess.run(cmd,cwd=CODE,env=env,stdout=f,stderr=subprocess.STDOUT)
 save(root/(arm+'_process.json'),dict(returncode=r.returncode,seconds=time.time()-start,command=cmd,GPU=GPU));assert r.returncode==0
def main():
 lock=(ROOT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 save(ROOT/'execution_start.json',dict(pid=os.getpid(),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip(),protocol_hash=sha(ROOT/'protocol.json'),start_unix=time.time(),serial_GPU=GPU))
 try:
  for order in ORDERS:launch(order)
  launch(None)
  subprocess.run(['/opt/conda/bin/python',str(HERE/'analyze.py')],check=True,cwd=CODE)
  subprocess.run(['/opt/conda/bin/python',str(HERE/'publish.py'),'final'],check=True,cwd=CODE)
  save(ROOT/'completion.json',dict(status='COMPLETE',new_policy_updates=1200,new_valid_trials=200,AE_updates=0,finished_unix=time.time()))
 except Exception:
  save(ROOT/'stopped.json',dict(error=traceback.format_exc(),time=time.time(),automatic_retry=False));raise
 finally:save(ROOT/'resource_release.json',dict(own_stage_processes_exited=True,other_processes_untouched=True,time=time.time()))
if __name__=='__main__':main()
