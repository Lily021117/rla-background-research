"""Freeze the authorized four-arm post-training experiment and protected inputs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
R=Path('/data/lyc/projects/rla-wm/experiments/official_libero_rla_consistency')
old=R/'stove_centering/run_20261024_01';out=a.out;out.mkdir(parents=True,exist_ok=False)
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(n,x):
 with (out/n).open('x') as f:json.dump(x,f,indent=2)
plan=copy.deepcopy(read(old/'run_plan.json'))
teacher=R/'run_20261023_10/stove/policy_run/C/isolated_smoke/step0600.pt'
arms={'E-C':{'target':'C','beta':0.},'E-D':{'target':'D','beta':0.},
      'E-C-keep':{'target':'C','beta':1.},'E-D-keep':{'target':'D','beta':1.}}
plan.update(experiment='control_preserving_rla_posttraining',arms=arms,teacher_checkpoint=str(teacher),teacher_sha256=sha(teacher),updates=600,
            common_student_step0=str(out/'common_student_step0.pt'),parent_teacher_updates=600,
            train_only_teacher_cache=str(out/'teacher_train_actions.npz'),
            checkpoints=[0,1,100,300,600],validation=[0,100,300,600],closed_loop_trials=240,
            seed=20261023,auxiliary_reinitialization_seed=20261025,sealed_read=False)
plan['shared_protocol'].update(lambda_aux=0.22408536585365854,policy_updates_this_run=2400,
    initialization='trained original C parameters; reinitialize only layer_norm2_rla and fc2_rla',
    common_initial_checkpoint='common_student_step0.pt',successful_updates=600,
    normalization='C: (z_C-mu_C)/s_C; D: (z_D-mu_D)/s_C',
    keep_loss='masked mean L1(student(training appearance), frozen eval C(same-state clean)); beta0 or1',
    numerical_loss_coordinates='official normalized action space; valid 8x7 elements; float32 reduction for keep',
    teacher='train observations only; no future frames or sidecars; teacher outputs are not ground truth')
plan['intervention']={'strategy':'new post-training four-arm factorial comparison, not a replacement for historical C/D',
   'AE_updates':0,'teacher_updates':0,'beta':1.,'lambda_aux':0.22408536585365854,
   'reinitialized_parameters':['model.layer_norm2_rla.weight','model.layer_norm2_rla.bias','model.fc2_rla.weight','model.fc2_rla.bias'],
   'unchanged_action_path':'all other C policy parameters, including RLA token trunk and gates',
   'teacher_input':'same physical training state, clean appearance only',
   'no_keep_arms_compute_keep_diagnostic_with_zero_weight':True}
manifest=read(Path(plan['inputs']['data_manifest']));train_manifest=copy.deepcopy(manifest)
train_manifest['episodes']=[e for e in manifest['episodes'] if e['split']=='train']
assert len(train_manifest['episodes'])==20
assert len([e for e in manifest['episodes'] if e['split']=='validation'])==5
save('train_only_manifest.json',train_manifest)
plan['train_only_manifest']=str(out/'train_only_manifest.json')
save('run_plan.json',plan);save('normalization.json',read(old/'normalization.json'))
ep=read(old/'evaluation_protocol.json');ep.update(models=list(arms),trials=240,primary_comparisons=['E-D-keep minus E-D','E-D-keep minus E-C-keep','each arm minus original C'],teacher='historical C, evaluation protocol must match',automatic_infrastructure_retries=0)
save('evaluation_protocol.json',ep);save('policy_registry.json',{m:str(out/m/'formal/step0600.pt') for m in arms})
paths=[teacher,old/'run_plan.json',old/'normalization.json',old/'evaluation_protocol.json',Path(plan['inputs']['data_manifest']),Path(plan['inputs']['action_proprio_statistics']),Path(plan['inputs']['B20_sampling_order']),Path(ep['source']),R/'run_20261023_10/stove/policy_run/per_trial_results.json']
paths.extend(p for p in Path(plan['previous']).glob('sidecars/*/*/*.json'))
save('protected_inputs.json',[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in dict.fromkeys(paths)])
save('source_lineage.json',dict(parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),official_commit='0fc6e97212442ebccffa06f1d42f5481e5f22950',upstream='mlzxy/rla-wm rebuttal/exp2_vla_adapter',historical_definitions_retained=True))
print(out,flush=True)
