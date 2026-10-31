"""Bounded six-policy scoring, automatic analysis/delivery, stop on faults."""
import argparse,fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_scratch_policy_aux.launch import environment
from frozen1800_state_readiness.generated_states import sha,save,read

def main(root):
 code=Path(__file__).resolve().parents[2]
 lock=(root/'evaluation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 ready=read(root/'github_sync_ready.json');assert ready['commit']==ready['remote_commit']==subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()
 frozen=read(root/'frozen_inputs.json');assert frozen['evaluation_protocol_sha256']==sha(root/'evaluation/evaluation_protocol.json')
 save(root/'scoring_gate.json',dict(status='FROZEN_COMMITTED_REMOTE_VERIFIED',commit=ready['commit'],**frozen,time=time.time()))
 env=environment();gpu=env['RLA_GPU_UUID'];env['SCRATCH_POLICY_ROOT']=str(root)
 start=time.time();save(root/'execution_start.json',dict(pid=os.getpid(),time=start,GPU=gpu,protocol_commit=ready['commit']))
 try:
  free=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True));assert free>=26624
  assert int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True))==0
  (root/'resource_start.txt').write_text(subprocess.check_output(['nvidia-smi','-i',gpu,'-q','-d','ECC,MEMORY'],text=True))
  cmd=[sys.executable,'-u',str(code/'experiments/rla_lr_budget_sweep/bootstrap.py'),'--out',str(root/'evaluation'),'--arm','evaluation','--entry','frozen1800_state_readiness.evaluate','--attempt','formal']
  with (root/'evaluation.log').open('x') as f:r=subprocess.run(cmd,cwd=code,env=env,stdout=f,stderr=subprocess.STDOUT)
  save(root/'process.json',dict(returncode=r.returncode,seconds=time.time()-start,command=cmd,GPU=gpu));assert r.returncode==0
  from frozen1800_state_readiness.analyze import main as analyze
  analyze(root)
  save(root/'resource_release.json',dict(own_evaluation_subprocess_exited=True,training_processes_started=0,other_processes_untouched=True,time=time.time()))
  from frozen1800_state_readiness.publish import publish
  publish(root,'complete')
  save(root/'completion.json',dict(status='COMPLETE',optimizer_updates=0,AE_updates=0,backward=0,valid_trials=read(root/'evaluation/evaluation_protocol.json')['trials'],seconds=time.time()-start))
 except Exception:
  save(root/'stopped.json',dict(time=time.time(),error=traceback.format_exc(),automatic_retry=False,automatic_device_switch=False))
  if not (root/'resource_release.json').exists():save(root/'resource_release.json',dict(own_evaluation_subprocess_exited=True,other_processes_untouched=True,time=time.time()))
  raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
