"""Bounded three-stage evaluation. No automatic infrastructure retry or training."""
import os,sys,subprocess,time,traceback,fcntl
from common import *
sys.path.insert(0,str(HERE.parent))
from rla_scratch_policy_aux.launch import environment
PY=P.parent/'vla-adapter-ref/.venv-cpu/bin/python'
def make_stage(stage,conditions,subset):
 root=BASE/stage;root.mkdir();save(root/'stage_conditions.json',conditions)
 ss=[s for s in read(BASE/'state_manifest.json') if s['subset']==subset and s['task'] in conditions];save(root/'state_manifest.json',ss)
 save(root/'checkpoint_inventory.json',read(BASE/'checkpoint_inventory.json'))
 protocol=read(BASE/'calibration_protocol.json');save(root/'evaluation_protocol.json',protocol)
 save(root/'protocol_commit.json',dict(protocol_sha256=sha(root/'evaluation_protocol.json'),parent_commit=read(BASE/'protocol_commit.json')))
 for task in conditions:
  (root/task).mkdir();save(root/task/'run_plan.json',read(OLD/task/'run_plan.json'))
 return root
def run(stage,conditions,subset):
 if not conditions:return
 root=make_stage(stage,conditions,subset)
 for task in conditions:
  assert int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))>=26624
  assert int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True))==0
  env=environment();env.update(OBS_CAL_ROOT=str(BASE),OBS_STAGE=stage,OBS_TASK=task,RLA_PROJECT_ROOT=str(P))
  cmd=[str(PY),'-u',str(HERE.parent/'rla_lr_budget_sweep/bootstrap.py'),'--out',str(root/task),'--arm','evaluation','--attempt','formal','--entry','rla_observation_calibration.evaluation']
  start=time.time()
  with (root/task/'evaluation.log').open('x') as log:p=subprocess.run(cmd,env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT)
  save(root/task/'process.json',dict(returncode=p.returncode,seconds=time.time()-start,command=cmd,GPU=GPU));assert p.returncode==0
def select(stage):
 result={}
 for task in TASK_NAMES:
  f=BASE/stage/task/'per_trial_results.json'
  if not f.exists():continue
  rows=read(f);conditions=read(BASE/stage/'stage_conditions.json')[task]
  rate={c:sum(r['success'] for r in rows if r['condition']==c)/30 for c in conditions}
  assert all(sum(r['condition']==c and r['status']=='COMPLETE' for r in rows)==30 for c in conditions)
  candidates=[c for c in conditions if .4<=rate[c]<.6]
  result[task]=dict(rates=rate,candidate=candidates[0] if candidates else None,combo_eligible=stage=='calibration' and len(conditions)==9 and all(v>=.6 for v in rate.values()),below40=any(v<.4 for v in rate.values()))
 return result
def main():
 lock=(BASE/'execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert read(BASE/'protocol_commit.json')['protocol_sha256']==sha(BASE/'calibration_protocol.json')
 accepted=read(BASE/'image_acceptance/manual_review.json')['accepted_by_task']
 save(BASE/'execution_start.json',dict(time=time.time(),pid=os.getpid(),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip()))
 try:
  run('calibration',{t:[c for c in CONDITIONS if c in accepted[t]] for t in TASK_NAMES},'calibration')
  first=select('calibration');save(BASE/'single_factor_selection.json',first)
  combo={t:[COMBO] for t,v in first.items() if v['combo_eligible'] and COMBO in accepted[t]};run('combination',combo,'calibration')
  second=select('combination');save(BASE/'combination_selection.json',second)
  selected={t:[v['candidate']] for t,v in first.items() if v['candidate']}
  selected.update({t:[v['candidate']] for t,v in second.items() if v['candidate']})
  save(BASE/'selected_candidates.json',selected);run('review',selected,'review')
  subprocess.run([str(PY),str(HERE/'analyze.py')],cwd=CODE,check=True)
  save(BASE/'completion.json',dict(status='COMPLETE',optimizer_updates=0,AE_updates=0,DINO_updates=0,backward=0))
 except Exception:save(BASE/'stopped.json',dict(error=traceback.format_exc(),time=time.time(),automatic_retry=False));raise
 finally:save(BASE/'resource_release.json',dict(processes_exited=True,other_jobs_untouched=True,time=time.time()))
if __name__=='__main__':main()
