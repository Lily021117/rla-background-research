"""All candidates, frozen ranking and paired state-clustered tuning statistics."""
import os,sys
if sys.executable=='/usr/bin/python3':
 os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1'
 os.environ['TMPDIR']='/data/lyc/tmp/rla_background_invariance/cache/tmp'
 os.execv('/data/lyc/projects/vla-adapter-ref/.venv-cpu/bin/python',['/data/lyc/projects/vla-adapter-ref/.venv-cpu/bin/python',__file__]+sys.argv[1:])
import argparse,csv,json,collections,hashlib
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();o=a.out
def read(p):return json.loads(Path(p).read_text())
def save(n,x):
 with (o/n).open('x') as f:json.dump(x,f,indent=2)
def csvsave(n,rows):
 with (o/n).open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
plan=read(o/'run_plan.json');proto=read(o/'evaluation_protocol.json');rows=read(o/'per_trial_results.json');assert len(rows)==960 and all(r['status']=='COMPLETE' for r in rows)
conditions=proto['conditions'];seeds=proto['states'];assert len(seeds)==20
index={(r['model'],r['seed'],r['condition']):r for r in rows};assert len(index)==960
for seed in seeds:
 for cond in conditions:
  for key in ('state_sha256','xml_sha256','language'):assert len({index[m,seed,cond][key] for m in proto['models']})==1
scores=[];summary=[];stages=[]
for model in proto['models']:
 arm,step=model.rsplit('_s',1);cfg=plan['arms'][arm];counts=[]
 for cond in conditions:
  rs=[index[model,s,cond] for s in seeds];n=sum(r['success'] for r in rs);counts.append(n)
  summary.append(dict(model=model,method=cfg['method'],multiplier=cfg['multiplier'],step=int(step),appearance=cond,successes=n,trials=20,success_rate=n/20))
  desc=[r for r in rs if r['first_lift_step'] is not None]
  stages.append(dict(model=model,appearance=cond,trials=20,success=n,issued_close=sum(r['issued_close_command'] for r in rs),lifted=len(desc),postlift_fail=sum(r['stage']=='post_lift_or_placement_failure' for r in rs),approach_fail=sum(r['stage']=='approach_failure' for r in rs),near_target_fail=sum(r['stage']=='near_target_grasp_or_contact_failure' for r in rs),conditional_descent_n=len(desc),descent_horizontal_path_m=float(np.mean([r['descent_horizontal_path_m'] for r in desc])) if desc else None,final_xy_m=float(np.mean([r['final_xy_m'] for r in rs])),postlift_contact_height_xy=sum(r['postlift_contact_height_xy'] for r in rs)))
 scores.append(dict(model=model,method=cfg['method'],multiplier=cfg['multiplier'],step=int(step),clean_success=counts[0],seen_success=counts[1],unseen_success=counts[2],total_success=sum(counts),mean_success_rate=sum(counts)/60))
selected={method:sorted([s for s in scores if s['method']==method],key=lambda x:(-x['total_success'],-x['clean_success'],x['step'],x['multiplier']))[0] for method in ('C','D-center')}
csvsave('all_candidate_scores.csv',scores);csvsave('success_rates.csv',summary);csvsave('stage_summary.csv',stages)
save('selected_configurations.json',dict(selection=plan['selection'],selected=selected,requires_original_1200_schedule_even_if_stopping600=True,development_used_for_selection=True,independent_confirmation=False,next_matching_policy_seeds=[20261026,20261027],fixed_AE_seed_stability_not_tested=True))
rng=np.random.default_rng(20261025);draw=rng.integers(0,20,(10000,20));stats=[];paired=[]
def compare(left,right,label):
 arr=np.array([[int(index[left,s,c]['success'])-int(index[right,s,c]['success']) for c in conditions] for s in seeds],float)
 for j,c in enumerate(conditions):
  d=arr[:,j];stats.append(dict(comparison=label,left=left,right=right,appearance=c,difference=float(d.mean()),CI95=np.quantile(d[draw].mean(1),[.025,.975]).tolist(),wins=int((d>0).sum()),losses=int((d<0).sum())))
  paired.extend(dict(comparison=label,seed=s,appearance=c,left_success=bool(index[left,s,c]['success']),right_success=bool(index[right,s,c]['success']),difference=int(d[i])) for i,s in enumerate(seeds))
 d=arr.mean(1);stats.append(dict(comparison=label,left=left,right=right,appearance='equal_mean_state_clustered',difference=float(d.mean()),CI95=np.quantile(d[draw].mean(1),[.025,.975]).tolist()))
 for j in (1,2):
  d=arr[:,j]-arr[:,0];stats.append(dict(comparison=label,left=left,right=right,appearance=conditions[j]+' interaction_vs_clean',difference=float(d.mean()),CI95=np.quantile(d[draw].mean(1),[.025,.975]).tolist()))
for m in (.1,.3,1,3):
 token=str(m).replace('.','p')
 for step in (600,1200):compare(f'D-center_m{token}_s{step}',f'C_m{token}_s{step}',f'matched_m{m}_s{step}')
compare(selected['D-center']['model'],selected['C']['model'],'separately_selected_tuning_estimates')
historical=[r for r in read(Path(plan['historical_source'])/'policy_run/per_trial_results.json') if r['model']=='C'];assert len(historical)==60
for r in historical:
 ref=index[proto['models'][0],r['seed'],r['condition']]
 for k in ('state_sha256','xml_sha256','language'):assert r[k]==ref[k]
 index['historical_C',r['seed'],r['condition']]=r
for method in selected:compare(selected[method]['model'],'historical_C',method+'_selected_minus_historical_C')
save('paired_statistics.json',dict(physical_state_clusters=20,draws=10000,bootstrap_seed=20261025,method='same sampled physical states jointly across appearances',selection_bias_warning='all intervals descriptive of reused tuning states; not independent validation, no adjustment for winner selection',single_training_seed=True,comparisons=stats));csvsave('paired_results.csv',paired)
curve=[];validation=[];inventory=[];init=[];rngs=[];actual={};counts=[]
for arm,cfg in plan['arms'].items():
 root=o/arm/'formal';train=list(csv.DictReader((root/'training_metrics.csv').open()));assert [int(r['step']) for r in train]==list(range(1,1201))
 acc=read(root/'training_acceptance.json');assert acc['successful_updates']==1200;actual[arm]=acc['actual_computed_updates']
 init.append(read(root/'initial_state_audit.json'))
 for r in train:assert abs(float(r['lr'])-plan['shared_protocol']['lr']*cfg['multiplier'])<1e-12 and float(r['weighted_keep'])==0
 for lo,hi in ((1,100),(501,600),(601,700),(1101,1200)):
  rs=train[lo-1:hi];curve.append(dict(model=arm,from_step=lo,to_step=hi,**{k:float(np.mean([float(r[k]) for r in rs])) for k in ['action_loss','auxiliary_loss','weighted_auxiliary','gradient_vla','gradient_head','gradient_aux_head']+[f'dim{d}_L1' for d in range(7)]}))
 vals=collections.defaultdict(list)
 for r in csv.DictReader((root/'validation_metrics.csv').open()):vals[int(r['step']),r['appearance']].append(r)
 for (step,c),rs in vals.items():validation.append(dict(model=arm,step=step,appearance=c,windows=len(rs),**{k:float(np.mean([float(r[k]) for r in rs])) for k in ['action_L1','translation_L1','rotation_L1','gripper_L1','auxiliary_L1']+[f'dim{d}_L1' for d in range(7)]+[f'h{h}_L1' for h in range(8)]}))
 for step in (0,600,1200):
  file=root/f'step{step:04}.pt';ck=torch.load(file,map_location='cpu');assert ck['global_step']==step and ck['next_physical_sample']==step*8
  if step:
   assert ck['scheduler']['last_epoch']==step and len(ck['optimizer']['state'])>0
   assert all(int(s['step'])==step for s in ck['optimizer']['state'].values());counts.append(dict(arm=arm,step=step,optimizer_states=len(ck['optimizer']['state'])))
  else:
   h=hashlib.sha256();h.update(ck['torch_rng'].numpy().tobytes())
   for x in ck['cuda_rng']:h.update(x.numpy().tobytes())
   h.update(repr(ck['python_rng']).encode());h.update(ck['numpy_rng'][1].tobytes());h.update(repr(ck['numpy_rng'][2:]).encode());rngs.append(h.hexdigest())
  inventory.append(dict(model=arm,step=step,path=str(file),sha256=sha(file),bytes=file.stat().st_size));del ck
assert len({x['parameter_hash'] for x in init})==1 and all(x['parameter_hash']==x['common_parameter_hash'] for x in init) and len(set(rngs))==1
csvsave('checkpoint_inventory.csv',inventory);csvsave('training_curve_summary.csv',curve);csvsave('validation_summary.csv',validation)
protected=[dict(path=x['path'],unchanged=sha(x['path'])==x['sha256']) for x in read(o/'protected_inputs.json')];assert all(x['unchanged'] for x in protected)
save('delivery_verification.json',dict(status='PASS_SEARCH_EXECUTION_NOT_METHOD',successful_updates={arm:1200 for arm in plan['arms']},actual_computed_updates=actual,total_successful_updates=9600,actual_total=sum(actual.values()),AE_updates=0,teacher_updates=0,valid_rollouts=960,invalid_rollouts=0,sealed_reads=0,protected_inputs=protected,initial_parameters_equal=True,initial_RNG_equal=True,RNG_hashes=rngs,checkpoint_optimizer_audit=counts,selection=selected,resource_release='all training/evaluation child processes exited before this CPU analysis'))
lines=['# Stove C/D-center LR and training-budget search','', 'Execution PASS; selected results are development tuning estimates, not independent evidence.','', '| Method | LR multiplier | Updates | Clean | Seen | Unseen | Mean |','|---|---:|---:|---:|---:|---:|---:|']
for s in scores:lines.append(f"|{s['method']}|{s['multiplier']}|{s['step']}|{s['clean_success']}/20|{s['seen_success']}/20|{s['unseen_success']}/20|{s['mean_success_rate']:.3f}|")
lines+=['','## Frozen selected configurations','',json.dumps(selected,indent=2),'','Selected600 checkpoints must retain this original1200-step schedule, not a re-scaled600-step schedule.','', 'All8 configurations completed1200 effective updates (9600 total);960 valid development rollouts. AE/teacher updates0; sealed reads0. Actual/replay counts are in delivery_verification.json. Original branches, checkpoints and registered input hashes preserved.','', 'One AdamW group: base LR0.0002, weight decay0.01, betas0.9/0.999, eps1e-8. MultiStepLR milestone1200 gamma0.1 after update; no warmup. Historical milestone600 differed only after update600, so m1 first600 update LRs match; no claim of bitwise historical reproducibility from LR alone.','', 'Equal updates do not imply convergence. Boundary optima and a final1200 optimum leave open LR-range/budget limitations. Full curves and validation summaries include all poor candidates; no outcome-driven extension was run.','', 'Physical-state-cluster bootstrap preserves all appearances of a state together. Paired flips/interaction intervals are in paired_statistics.json. The same20 reused development states and materials were used for selection: winner intervals are optimistic tuning descriptions, not independent validation. Floor/ceiling effects and one policy seed limit interpretation; fixed AE seed stability was not tested.','', 'Phase metrics retain live official success. Lift3cm and near-target10cm are descriptive proxies, not replacement success labels. Conditional descent statistics have stage-entry selection and do not alone establish causal control quality.','', 'Next proposed matched policy seeds:20261026 and20261027, each method using its frozen selected LR/budget and the original1200 scheduler definition, common initial policy/data/settings. They are not run this round. Independent states/background evaluation is still needed; full-method seed stability would additionally require matched AE seeds.','', 'Improving D-center training is distinct from outperforming tuned C. No automatic additional LR, budget, seed, task or loss search.']
with (o/'report.md').open('x') as f:f.write('\n'.join(lines)+'\n')
print('ANALYSIS_COMPLETE',selected,flush=True)
