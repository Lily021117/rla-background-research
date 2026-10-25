"""Completed counts, initialization, retained history and stage summaries."""
import argparse
import collections
import csv
import json
import hashlib
from pathlib import Path
import subprocess
import numpy as np
import torch
from control_preserving_rla.common import read,save,sha,digest

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();o=a.out
def csvsave(name,rows):
 with (o/name).open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
plan=read(o/'run_plan.json');inventory=[];train=[];val=[];accept=[];audits=[];rng_hashes=[]
for arm,cfg in plan['arms'].items():
 original=o/arm/'formal';path=(o/'combined_metrics'/arm) if (o/'combined_metrics'/arm).exists() else original
 final_dir=(o/arm/'resume_gpu1') if arm=='E-D-keep' and (o/arm/'resume_gpu1/training_acceptance.json').exists() else original
 rows=list(csv.DictReader((path/'training_metrics.csv').open()))
 assert len(rows)==600 and [int(r['step']) for r in rows]==list(range(1,601))
 acc=read(final_dir/'training_acceptance.json');assert acc['successful_updates']==600 and acc['strict_reload'];accept.append(dict(model=arm,**acc))
 audit=read(original/'initial_state_audit.json');audits.append(audit);assert audit['parameter_hash']==audit['common_parameter_hash']==read(o/'initialization_acceptance.json')['student_parameter_hash']
 initial=torch.load(original/'step0000.pt',map_location='cpu')
 assert digest(initial['parameters'])==audit['parameter_hash']
 h=hashlib.sha256();h.update(initial['torch_rng'].numpy().tobytes())
 for x in initial['cuda_rng']:h.update(x.numpy().tobytes())
 h.update(repr(initial['python_rng']).encode());h.update(initial['numpy_rng'][0].encode());h.update(initial['numpy_rng'][1].tobytes());h.update(repr(initial['numpy_rng'][2:]).encode())
 rng_hashes.append(h.hexdigest());del initial
 for step in (0,1,100,300,600):
  ck=(final_dir if step==600 else original)/f'step{step:04}.pt';inventory.append(dict(model=arm,step=step,path=str(ck),sha256=sha(ck),bytes=ck.stat().st_size))
 for lo,hi in ((1,100),(101,300),(301,600),(1,600)):
  rs=[r for r in rows if lo<=int(r['step'])<=hi]
  train.append(dict(model=arm,target=cfg['target'],beta=cfg['beta'],from_step=lo,to_step=hi,updates=len(rs),**{k:float(np.mean([float(r[k]) for r in rs])) for k in ('action_loss','auxiliary_loss','weighted_auxiliary','keep_loss','weighted_keep','gradient_vla','gradient_head','gradient_aux_head')}))
 for r in csv.DictReader((path/'validation_metrics.csv').open()):val.append(dict(model=arm,**r))
assert len(set(rng_hashes))==1
save(o/'four_student_initialization_audit.json',dict(status='PASS',parameter_hash=audits[0]['parameter_hash'],actual_step0_RNG_hashes=rng_hashes,exact_matching_parameters_and_RNG=True,checked_from_actual_formal_checkpoints=True))
csvsave('checkpoint_inventory.csv',inventory);csvsave('training_contribution_summary.csv',train)
groups=collections.defaultdict(list)
for r in val:groups[r['model'],r['step'],r['appearance']].append(r)
csvsave('validation_summary.csv',[dict(model=k[0],step=int(k[1]),appearance=k[2],windows=len(rs),**{f:float(np.mean([float(r[f]) for r in rs])) for f in ('action_L1','translation_L1','rotation_L1','gripper_L1','auxiliary_L1')}) for k,rs in groups.items()])
stagegroups=collections.defaultdict(list)
for r in csv.DictReader((o/'stage_metrics_five_arms.csv').open()):stagegroups[r['model'],r['appearance']].append(r)
summaries=[]
for (arm,c),rs in stagegroups.items():
 row=dict(model=arm,appearance=c,trials=len(rs),close=sum(r.get('ever_close')=='True' for r in rs),lift=sum(r.get('lifted')=='True' for r in rs),ever_recomputed_simultaneous=sum(r.get('ever_simultaneous')=='True' for r in rs),postlift_open=sum(bool(r.get('first_open')) for r in rs),finger_contact_loss=sum(bool(r.get('first_finger_contact_loss')) for r in rs))
 for key in ('descent_horizontal_path_m','mean_descent_horizontal_increment_m','descent_xy_error_change_m','final_xy_m','minimum_height_when_centered_m'):
  values=[float(r[key]) for r in rs if r.get(key) not in ('',None)];row[key+'_n']=len(values);row[key+'_mean']=float(np.mean(values)) if values else None;row[key+'_median']=float(np.median(values)) if values else None
 summaries.append(row)
csvsave('stage_condition_summary.csv',summaries)
protected=[dict(path=x['path'],unchanged=sha(Path(x['path']))==x['sha256']) for x in read(o/'protected_inputs.json')];assert all(r['unchanged'] for r in protected)
trials=read(o/'per_trial_results.json');assert len(trials)==240 and all(r['status']=='COMPLETE' for r in trials)
disagreement=list(csv.DictReader((o/'teacher_demonstration_disagreement.csv').open()));groups=collections.defaultdict(list)
for r in disagreement:groups[int(r['horizon']),int(r['dimension'])].append(r)
csvsave('teacher_disagreement_summary.csv',[dict(horizon=k[0],dimension=k[1],windows=sum(int(r['windows']) for r in rs),teacher_demo_L1=sum(int(r['windows'])*float(r['teacher_demo_L1']) for r in rs)/sum(int(r['windows']) for r in rs)) for k,rs in groups.items()])
save(o/'delivery_verification.json',dict(status='PASS_EXECUTION_NOT_METHOD',policy_successful_updates={m:600 for m in plan['arms']},actual_computed_updates={m:(730 if m=='E-D-keep' else 600) for m in plan['arms']},optimizer_update_replays=130,failed_update_attempts=1,parent_C_policy_updates_shared_cost=600,AE_updates=0,teacher_updates=0,teacher_train_cache=read(o/'teacher_cache_metadata.json'),valid_new_rollouts=240,invalid_new_rollouts=0,closed_loop_retries=0,sealed_reads=0,protected_inputs=protected,initialization=read(o/'initialization_acceptance.json'),gate=read(o/'gate_acceptance.json'),training_acceptance=accept,recovery=read(o/'recovery_comparison.json'),resume_precheck=read(o/'resume_precheck_acceptance.json'),source_lineage=read(o/'source_lineage.json'),active_task_gpu_processes_expected=0,raw_gpu_process_snapshot=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv'],text=True),limitations=['one training seed','20 reused development states','teacher is an imperfect learned policy, not action truth','historical C comparisons include auxiliary readout reset, fresh optimizer and extra600 updates','E-D-keep restored step300 after external-memory-pressure OOM at step431;130 successful computations replayed on authorized GPU1; historical ECC retained','recomputed contact/goal timing never replaces recorded success']))
print('FINALIZATION_PASS',flush=True)
