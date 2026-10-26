"""Freeze paired seeds without reading sealed data or altering any AE target."""
import argparse,json,hashlib,csv,subprocess,collections
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--initialization-scope',choices=['fixed_action_path_new_auxiliary','all_random_modules_new_seed'],required=True);a=p.parse_args();o=a.out
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
historical=read(H/'run_plan.json');original=read(S/'policy_run/run_plan.json');manifest=read(historical['inputs']['data_manifest']);train=[e for e in manifest['episodes'] if e['split']=='train'];dev=[e for e in manifest['episodes'] if e['split']=='validation']
assert len(train)==20 and len(dev)==5 and not ({e['episode'] for e in train}&{e['episode'] for e in dev})
for k in ['common_step0','common_step0_sha256','inputs']:assert historical[k]==original[k]
assert sha(historical['common_step0'])==historical['common_step0_sha256']
assert historical['shared_protocol']['lr']==2e-4 and historical['shared_protocol']['lambda_aux']==.22408536585365854
windows=[(e['episode'],t) for e in train for t in range(e['valid_chunks'])];assert len(windows)==2555
oldorder=[];rng=np.random.RandomState(20261023)
while len(oldorder)<4800:oldorder.extend(windows[i] for i in rng.permutation(len(windows)))
assert [list(x) for x in oldorder[:4800]]==read(historical['inputs']['B20_sampling_order'])
seeds=[20261029,20261030];protected={}
for path in [historical['common_step0'],historical['inputs']['data_manifest'],historical['inputs']['action_proprio_statistics'],historical['inputs']['B20_sampling_order'],H/'normalization.json',H/'run_plan.json',H/'per_trial_results.json',S/'policy_run/per_trial_results.json',A/'sidecar_provenance.json',A/'sidecar_alignment_audit.json']:
 protected[str(path)]=sha(path)
provenance=read(A/'sidecar_provenance.json');assert len(provenance)==100
for x in provenance:
 assert x['model'] in ['C','D'];assert Path(x['file']).is_file();protected[x['file']]=x['sha256']
 for k in ['mask_file']:protected[x[k]]=sha(x[k])
for m in ['C','D']:
 for c in ['clean','background']:
  for n in ['stats.json','raw_target_statistics.json','libero_spatial/keys.json']:protected[str(A/'sidecars'/m/c/n)]=sha(A/'sidecars'/m/c/n)
for seed in seeds:
 root=o/f'seed_{seed}';root.mkdir(exist_ok=False);rng=np.random.RandomState(seed);order=[]
 while len(order)<4800:order.extend(windows[i] for i in rng.permutation(len(windows)))
 order=order[:4800];save(root/'sampling_order.json',order)
 plan=read(H/'run_plan.json');plan.update(experiment='fixed_AE_policy_seed_replication',previous=str(A),common_step0=str(root/'common_seed_step0.pt'),common_step0_sha256='PENDING_ZERO_UPDATE_INITIALIZATION',original_common_step0=historical['common_step0'],original_common_step0_sha256=historical['common_step0_sha256'],initialization_scope=a.initialization_scope,policy_seed=seed,target_source_by_arm={'C':'C','D-center':'D'},arms=['C','D-center'],closed_loop_trials=120,historical_source=str(S),historical_center_source=str(H))
 plan['inputs']['B20_sampling_order']=str(root/'sampling_order.json');plan['inputs']['B20_sampling_sha256']=sha(root/'sampling_order.json');plan['shared_protocol'].update(seed=seed,sampler='same historical repeated RandomState permutation algorithm, new preassigned seed; no replacement within each full pass',common_initial_checkpoint='common_seed_step0.pt',center_only_difference=False,normalization='C=(z_C-mu_C)/s_C; D-center=(z_D-mu_D)/s_C; frozen train-only statistics',C_D_only_target_source_differs=True)
 save(root/'planned_run_plan.json',plan);save(root/'normalization.json',read(H/'normalization.json'));save(root/'preflight6/isolated_smoke/preflight_acceptance.json',read(H/'preflight6/isolated_smoke/preflight_acceptance.json'))
 ep=read(H/'evaluation_protocol.json');ep.update(models=['C_s600','D-center_s600'],trials=120,role='reused development for fixed-AE policy seed replication, not independent test',policy_training_seed=seed);save(root/'evaluation_protocol.json',ep);save(root/'policy_registry.json',{m+'_s600':str(root/m/'formal/step0600.pt') for m in ['C','D-center']})
 save(root/'sampling_audit.json',dict(seed=seed,physical_samples=4800,appearance_samples=9600,independent_train_episodes=20,eligible_windows=2555,equivalent_passes=4800/2555,episode_exposures=dict(collections.Counter(e for e,t in order)),same_C_D_order=True,normalization_fixed=True))
 with (root/'LR_curve.csv').open('x') as f:
  w=csv.writer(f);w.writerow(['update','actual_LR','LR_after_scheduler'])
  for step in range(1,601):w.writerow([step,.0002,.00002 if step==600 else .0002])
save(o/'protected_inputs.json',protected)
save(o/'replication_protocol.json',dict(frozen=True,seeds=seeds,historical_seed=20261023,initialization_scope=a.initialization_scope,methods=['C','D-center'],AE_updates=0,policy_updates_per_arm=600,total_policy_updates=2400,valid_rollouts=240,task='stove',train_episodes=20,development_episodes=5,physical_development_states=20,appearances=['clean','seen_background','unseen_background'],AE_source=str(A),normalization_source=str(H/'normalization.json'),lambda_aux=.22408536585365854,LR=.0002,scheduler='MultiStepLR[600] gamma0.1 AFTER optimizer; all600 updates2e-4',seed_order=seeds,arm_order=['C','D-center'],sampling='matched within seed, distinct between seeds',init_mode_confirmed_before_updates=True,sealed_reads=0,keep=False,teacher=False,automatic_extra_runs=False,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),historical_comparison_limits='fixed AE and reused development; only three policy seeds; initialization scope explicitly reported'))
print('FROZEN',seeds,len(windows),a.initialization_scope,flush=True)
