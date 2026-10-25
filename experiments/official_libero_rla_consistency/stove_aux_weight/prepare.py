"""Freeze the authorized one-factor experiment before GPU work."""
import argparse,json,hashlib,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
root=Path('/data/lyc/projects/rla-wm/experiments/official_libero_rla_consistency')
old=root/'stove_centering/run_20261024_01';out=a.out;out.mkdir(parents=True,exist_ok=False)
def read(p):return json.loads(p.read_text())
def save(n,x):
 with (out/n).open('x') as f:json.dump(x,f,indent=2)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
 return h.hexdigest()
plan=read(old/'run_plan.json');original=plan['shared_protocol']['lambda_aux']
plan.update(experiment='stove_policy_aux_weight_0p1',arms=['C-low','D-center-low'],historical_centering=str(old))
plan['shared_protocol'].update(lambda_aux=original*0.1,center_only_difference=False,normalization='C-low: (z_C-mu_C)/s_C; D-center-low: (z_D-mu_D)/s_C',C_D_only_target_source_differs=False,policy_updates_this_run=1200)
plan['target_center_definition']={'C-low':'(z_C-mu_C)/s_C','D-center-low':'(z_D-mu_D)/s_C','scale':'frozen s_C scalar; no clipping or rescaling'}
plan['intervention']={'policy_lambda_original':original,'policy_lambda_low':original*0.1,'AE_updated':False,'AE_lambda_inv':0.009127798272072366,'automatic_calibration':False,'new_architecture':False,'action_postprocessing_changed':False}
save('run_plan.json',plan);save('normalization.json',read(old/'normalization.json'))
ep=read(old/'evaluation_protocol.json');ep.update(models=plan['arms'],primary_comparison='D-center-low minus D-center; C-low minus C; D-center-low minus C-low and C',historical_C='reuse only after protocol checks',historical_D_center='reuse only after protocol checks')
save('evaluation_protocol.json',ep)
save('policy_registry.json',{arm:str(out/arm/'isolated_smoke/step0600.pt') for arm in plan['arms']})
paths=[old/'run_plan.json',old/'normalization.json',old/'evaluation_protocol.json',Path(plan['common_step0']),Path(plan['inputs']['data_manifest']),Path(plan['inputs']['action_proprio_statistics']),Path(plan['inputs']['B20_sampling_order']),Path(ep['source'])]
for arm in ('C','D'):
 paths.extend(p for p in (Path(plan['previous'])/'sidecars'/arm).rglob('*.json'))
paths.extend([root/'run_20261023_10/stove/policy_run/C/isolated_smoke/step0600.pt',old/'D-center/isolated_smoke/step0600.pt',root/'run_20261023_10/stove/policy_run/per_trial_results.json',old/'per_trial_results.json'])
save('protected_inputs.json',[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in paths])
assert sha(Path(plan['common_step0']))==plan['common_step0_sha256']
save('configuration_diff.json',dict(new_policy_weight=original*0.1,old_policy_weight=original,only_intervention='policy auxiliary coefficient multiplied by 0.1 within each target family',training_updates_each=600,AE_updates=0,other_settings='unchanged, including target means/scalar, RNG, sampler, backend, AdamW, microbatch8 x accumulation2',comparison_confound='C and D targets and centers differ by established method; only within-family weight intervention is single-variable'))
save('source_snapshot.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),new_sources={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}))
print(out,flush=True)
