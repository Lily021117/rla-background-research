"""Sequential bounded training, then Clean gate, then all-or-none background evaluation."""
import sys,os,time,subprocess,fcntl,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_scratch_policy_aux.common import *
from rla_scratch_policy_aux.launch import run
def main():
 lock=(ROOT/'orchestrator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 save(ROOT/'execution_start.json',dict(pid=os.getpid(),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip(),GPU=GPU,protocol_hash=sha(ROOT/'protocol.json'),serial_gpu=True,started_unix=time.time()))
 try:
  assert read(ROOT/'target_acceptance.json')['status']=='PASS'
  for order in ORDERS:
   gate=ROOT/f'seed_{order}/preflight/formal/preflight_acceptance.json'
   if not gate.exists():run('preflight',order,'preflight')
   assert read(gate)['status']=='PASS'
   for arm in ARMS:
    # A completed stage is reused only if its acceptance explicitly reports600.
    done=ROOT/f'seed_{order}'/arm/'formal/training_acceptance.json'
    if done.exists():assert read(done)['successful_updates']==600;continue
    run('train',order,arm)
  run('clean')
  results=read(ROOT/'evaluation_clean/per_trial_results.json');assert len(results)==160 and all(r['status']=='COMPLETE' for r in results)
  counts={a:{str(o):sum(r['success'] for r in results if r['model']==f'{a}_order{o}_s600') for o in ORDERS} for a in ARMS}
  passed=any(all(n>=12 for n in v.values()) for v in counts.values());save(ROOT/'clean_gate.json',dict(counts=counts,passed=passed,rule='any single method succeeds>=12/20 in both orders',evaluated_all160=True))
  if passed:run('background')
  subprocess.run(['/opt/conda/bin/python',str(HERE/'summarize.py')],check=True,cwd=CODE)
  subprocess.run(['/opt/conda/bin/python',str(HERE/'publish.py'),'final'],check=True,cwd=CODE)
  save(ROOT/'completion.json',dict(status='COMPLETE',completed_unix=time.time(),policy_updates=4800,AE_updates=0,rollouts=800 if passed else 160))
 except Exception:
  save(ROOT/'stopped.json',dict(error=traceback.format_exc(),time=time.time(),automatic_retry=False));raise
 finally:save(ROOT/'resource_release.json',dict(own_stage_processes_exited=True,supervisor_finished_unix=time.time(),other_processes_untouched=True))
if __name__=='__main__':main()
