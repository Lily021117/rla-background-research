"""Freeze one inv0.3 experiment; reuse read-only stove data and feature cache."""
import argparse,json,csv,hashlib,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();o=a.out
R=Path('/data/lyc/projects/rla-wm');S=R/'experiments/official_libero_rla_consistency/run_20261023_10/stove';H=R/'experiments/official_libero_rla_consistency/stove_centering/run_20261024_01';A=S/'ae_run'
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
ae=o/'ae_run';ae.mkdir(exist_ok=False)
old=read(A/'training_plan.json');new=dict(old);new['lambda_cons']=old['lambda_cons']*.3
new['lambda_selection']='preassigned0.3 times historical actual coefficient; no recalibration'
assert old['updates']==600 and old['lr']==1e-4 and old['seed']==20261023
save(ae/'training_plan.json',new);save(o/'historical_AE_training_plan.json',old)
for n in ['data_manifest.json','official_initialization.json','source_provenance.json']:save(ae/n,read(A/n))
(ae/'features').symlink_to(A/'features',target_is_directory=True)
(ae/'development_protocol').symlink_to(A/'development_protocol',target_is_directory=True)
active=read(sorted((A/'model_registry_versions').glob('*.json'))[-1]);save(ae/'active_models.json',{'D':str(ae/'AE_D_inv03')})
es=read(ae/'data_manifest.json')['episodes'];assert sum(e['split']=='train' for e in es)==20 and sum(e['split']=='validation' for e in es)==5
assert len({e['episode'] for e in es})==25
plan=read(H/'run_plan.json');original=read(S/'policy_run/run_plan.json')
for k in ['common_step0','common_step0_sha256','inputs']:assert plan[k]==original[k]
for k in ['lr','lambda_aux','optimizer','seed','microbatch_appearance','accumulation','backend','dtype','weight_decay','scheduler']:assert plan['shared_protocol'][k]==original['shared_protocol'][k]
assert plan['shared_protocol']['lr']==.0002 and plan['shared_protocol']['lambda_aux']==.22408536585365854
plan.update(experiment='stove_AE_inv03_only',previous=str(ae),arms=['D-inv03'],closed_loop_trials=60,historical_source=str(S),historical_center_source=str(H),AE_lambda_inv=new['lambda_cons'],target_center_definition={'D-inv03':'(z_inv03 - mu_inv03_train) / historical_s_C'})
plan['shared_protocol']['center_only_difference']=False
plan['shared_protocol']['normalization']='new AE own training mean; unchanged historical scalar s_C; paired complete windows and horizons1..8'
save(o/'run_plan.json',plan);save(o/'historical_policy_plan.json',read(H/'run_plan.json'))
protocol=read(H/'evaluation_protocol.json');protocol.update(models=['D-inv03_s600'],trials=60,primary_comparison='new D-inv03 minus historical D-center and C; lowLR separately')
save(o/'evaluation_protocol.json',protocol);save(o/'policy_registry.json',{'D-inv03_s600':str(o/'D-inv03/formal/step0600.pt')})
save(o/'preflight6/isolated_smoke/preflight_acceptance.json',read(H/'preflight6/isolated_smoke/preflight_acceptance.json'))
save(o/'reused_preflight.json',dict(source=str(H/'preflight6/isolated_smoke/preflight_acceptance.json'),reason='same official model, initial weights, batch, dtype, checkpointing, action interface; new target reader checked separately',optimizer_updates=0))
init=read(A/'official_initialization.json');protected={}
for key in ['config','encoder','decoder']:
 path=init[key];assert sha(path)==init[key+'_sha256'];protected[path]=init[key+'_sha256']
assert sha(plan['common_step0'])==plan['common_step0_sha256']
paths=[plan['common_step0'],plan['inputs']['data_manifest'],plan['inputs']['action_proprio_statistics'],plan['inputs']['B20_sampling_order'],A/'training_plan.json',H/'normalization.json',H/'run_plan.json',H/'per_trial_results.json',S/'policy_run/per_trial_results.json',A/'data_manifest.json']
for path in paths:protected[str(path)]=sha(path)
save(o/'protected_inputs.json',protected)
save(o/'experiment_protocol.json',dict(frozen=True,arm='D-inv03',AE_updates=600,policy_updates=600,valid_rollouts=60,lambda_inv_old=old['lambda_cons'],lambda_inv_new=new['lambda_cons'],lambda_aux=plan['shared_protocol']['lambda_aux'],policy_lr=.0002,policy_scheduler='optimizer.step then scheduler.step; milestone600; all600 updates LR2e-4; checkpoint LR2e-5 after last scheduler call',historical_AE_active=active,expected_AE_order=str(Path(active['D'])/'sampling_order.json'),only_intended_change='AE consistency coefficient and derived target/own training center',historical_C_target_center='own C mean',historical_D_target_center='own D mean',common_scale_source=str(H/'normalization.json'),no_keep=True,no_teacher=True,no_new_seed=True,no_sealed_reads=True,sidecar_storage_alias='D directory denotes this run D-inv03 only, never historical D',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
with (o/'policy_LR_curve.csv').open('x') as f:
 w=csv.writer(f);w.writerow(['update','LR_used','scheduler_last_epoch_after','LR_after_scheduler'])
 for step in range(1,601):w.writerow([step,.0002,step,.00002 if step==600 else .0002])
print('FROZEN',new['lambda_cons'],plan['shared_protocol']['lambda_aux'],len(es))
