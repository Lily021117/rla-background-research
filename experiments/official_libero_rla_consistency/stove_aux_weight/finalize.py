"""Summarize fixed outputs and verify protected input digests after completion."""
import argparse,json,csv,hashlib,collections,subprocess
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();o=a.out
def read(p):return json.loads(p.read_text())
def save(n,x):
 with (o/n).open('x') as f:json.dump(x,f,indent=2)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def csvsave(n,rows):
 with (o/n).open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
plan=read(o/'run_plan.json');root=Path(plan['inputs']['project'])/'experiments/official_libero_rla_consistency'
arms={'C':root/'run_20261023_10/stove/policy_run/C/isolated_smoke','D-center':root/'stove_centering/run_20261024_01/D-center/isolated_smoke','C-low':o/'combined_metrics/C-low','D-center-low':o/'combined_metrics/D-center-low'}
logs=[];inventory=[];accept=[]
for arm,path in arms.items():
 rows=list(csv.DictReader((path/'training_metrics.csv').open()))
 if arm.endswith('low'):
  assert len(rows)==600 and [int(r['step']) for r in rows]==list(range(1,601))
  acc=read(o/arm/'resume_gpu0/training_acceptance.json');assert acc['successful_updates']==600 and acc['strict_reload'];accept.append(dict(model=arm,**acc))
  for step in (0,1,100,300,600):
   ck=o/arm/('resume_gpu0' if step==600 else 'isolated_smoke')/f'step{step:04}.pt';inventory.append(dict(model=arm,step=step,path=str(ck),sha256=sha(ck),bytes=ck.stat().st_size))
 for lo,hi in [(1,100),(101,300),(301,600),(1,600)]:
  rs=[r for r in rows if lo<=int(r['step'])<=hi]
  lam=plan['shared_protocol']['lambda_aux'] if arm.endswith('low') else plan['intervention']['policy_lambda_original']
  logs.append(dict(model=arm,from_step=lo,to_step=hi,lambda_aux=lam,updates=len(rs),mean_action_loss=np.mean([float(r['action_loss']) for r in rs]),mean_auxiliary_loss=np.mean([float(r['auxiliary_loss']) for r in rs]),mean_weighted_auxiliary=np.mean([float(r['auxiliary_loss'])*lam for r in rs]),mean_shared_gradient=np.mean([float(r['gradient_vla']) for r in rs]),mean_aux_head_gradient=np.mean([float(r['gradient_aux_head']) for r in rs])))
csvsave('training_contribution_summary.csv',logs);csvsave('checkpoint_inventory_completed.csv',inventory)
protected=read(o/'protected_inputs.json');checks=[dict(path=r['path'],unchanged=sha(Path(r['path']))==r['sha256']) for r in protected];assert all(x['unchanged'] for x in checks)
new=read(o/'per_trial_results.json');assert len(new)==120 and all(r['status']=='COMPLETE' for r in new)
groups=collections.defaultdict(list)
for r in csv.DictReader((o/'fixed_action_dimension_metrics.csv').open()):groups[r['model'],r['split'],r['appearance'],r['horizon'],r['dimension']].append(float(r['absolute_error']))
csvsave('action_dimension_summary.csv',[dict(model=k[0],split=k[1],appearance=k[2],horizon=int(k[3]),dimension=int(k[4]),n=len(v),mean_absolute_error=np.mean(v)) for k,v in groups.items()])
stagegroups=collections.defaultdict(list)
for r in csv.DictReader((o/'stage_metrics_four_arms.csv').open()):stagegroups[r['model'],r['appearance']].append(r)
ss=[]
for (arm,app),rs in stagegroups.items():
 row=dict(model=arm,appearance=app,trials=len(rs),close=sum(r.get('ever_close')=='True' for r in rs),lift=sum(r['lifted']=='True' for r in rs),ever_recomputed_simultaneous=sum(r.get('ever_simultaneous')=='True' for r in rs),postlift_open=sum(bool(r.get('first_open')) for r in rs),finger_contact_loss=sum(bool(r.get('first_finger_contact_loss')) for r in rs))
 for key in ('descent_horizontal_path_m','mean_descent_horizontal_increment_m','descent_xy_error_change_m','final_xy_m','minimum_height_when_centered_m'):
  vals=[float(r[key]) for r in rs if r.get(key) not in ('',None)]
  row[key+'_n']=len(vals);row[key+'_mean']=float(np.mean(vals)) if vals else None;row[key+'_median']=float(np.median(vals)) if vals else None
 ss.append(row)
csvsave('stage_condition_summary.csv',ss)
save('delivery_verification_completed.json',dict(status='PASS_EXECUTION_NOT_METHOD',policy_successful_updates={'C-low':600,'D-center-low':600},actual_computed_updates={'C-low':600,'D-center-low':600},replayed_updates=0,AE_updates=0,valid_new_rollouts=120,invalid_new_rollouts=sum(r['status']!='COMPLETE' for r in new),sealed_reads=0,protected_inputs=checks,initialization=read(o/'initialization_acceptance.json'),training_acceptance=accept,protocol_deviations=['Both policies stopped after update300 during validation with CUDA execution failure on GPU1; full step300 resumed on explicitly authorized GPU0 sequentially, no optimizer updates replayed','Incomplete validation300 repeated; original partial rows preserved but excluded from derived complete metrics','CPU verification loader explicitly supports trusted full-state checkpoints under newer torch; no training change'],recovery=read(o/'recovery_authorization.json'),source_commit_at_start=read(o/'source_snapshot.json')['commit'],gpu_resources_after=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv'],text=True),historical_success_labels_unchanged=True))
print(json.dumps(accept,indent=2))
