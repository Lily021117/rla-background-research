import sys,os,subprocess,time,traceback,fcntl
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_scratch_low_seeds.common import *
from rla_scratch_policy_aux.launch import environment
def launch(seed,stage):
 assert int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))>=26624
 assert int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True))==0
 root=ROOT/f'seed_{seed}' if seed else ROOT/'evaluation'
 entry='gate' if stage=='initialization' else 'evaluation' if stage=='evaluation' else 'training'
 env=environment();env['SCRATCH_POLICY_ROOT']=str(ROOT)
 if stage=='initialization':env.update(RLA_BOOTSTRAP_PLAN='planned_run_plan.json',RLA_BOOTSTRAP_SEED=str(seed),RLA_FULL_SEED_RECIPE='1')
 cmd=[sys.executable,'-u',str(HERE.parent/'rla_lr_budget_sweep/bootstrap.py'),'--out',str(root),'--arm',stage,'--entry','rla_scratch_low_seeds.'+entry,'--attempt','formal']
 start=time.time()
 with (root/(stage+'.log')).open('x') as f:r=subprocess.run(cmd,cwd=CODE,env=env,stdout=f,stderr=subprocess.STDOUT)
 save(root/(stage+'_process.json'),dict(returncode=r.returncode,seconds=time.time()-start,command=cmd,GPU=GPU));assert r.returncode==0
def main():
 lock=(ROOT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 seeds=read(ROOT/'protocol.json')['seeds'];save(ROOT/'execution_start.json',dict(pid=os.getpid(),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip(),protocol_hash=sha(ROOT/'protocol.json'),start_unix=time.time(),serial_GPU=GPU))
 try:
  for s in seeds:
   launch(s,'initialization');launch(s,'preflight')
   for arm in ARMS:launch(s,arm)
  frozen=[read(ROOT/f'seed_{s}/module_initialization_audit.json')['frozen_pretrained_sha256'] for s in seeds];assert len(set(frozen))==1
  save(ROOT/'evaluation/run_plan.json',read(ROOT/f'seed_{seeds[0]}/run_plan.json'))
  launch(None,'evaluation')
  subprocess.run(['/opt/conda/bin/python',str(HERE/'analyze.py')],check=True,cwd=CODE)
  subprocess.run(['/opt/conda/bin/python',str(HERE/'publish.py'),'final'],check=True,cwd=CODE)
  save(ROOT/'completion.json',dict(status='COMPLETE',successful_policy_updates=3600,valid_rollouts=600,AE_updates=0,finished_unix=time.time()))
 except Exception:
  save(ROOT/'stopped.json',dict(error=traceback.format_exc(),time=time.time(),automatic_retry=False));raise
 finally:save(ROOT/'resource_release.json',dict(own_stage_processes_exited=True,other_processes_untouched=True,time=time.time()))
if __name__=='__main__':main()
