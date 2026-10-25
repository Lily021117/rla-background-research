"""Export reviewable per-arm configs and the new implementation hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
plan=json.loads((a.out/'run_plan.json').read_text())
for arm,cfg in plan['arms'].items():
 value=dict(arm=arm,initial_policy=plan['common_student_step0'],teacher=plan['teacher_checkpoint'],teacher_sha256=plan['teacher_sha256'],
    parent_shared_C_updates=600,additional_successful_updates=600,target=cfg['target'],beta=cfg['beta'],lambda_aux=plan['shared_protocol']['lambda_aux'],
    target_coordinates='(z_C-mu_C)/s_C' if cfg['target']=='C' else '(z_D-mu_D)/s_C',normalization=str(a.out/'normalization.json'),
    teacher_cache=plan['train_only_teacher_cache'],teacher_appearance='same-state clean, train only',teacher_mode='eval stop-gradient',
    seed=plan['seed'],sampling_order=plan['inputs']['B20_sampling_order'],data=plan['inputs']['data_manifest'],
    protocol=plan['shared_protocol'],reinitialized_readout=plan['intervention']['reinitialized_parameters'])
 with (a.out/('training_config_'+arm+'.json')).open('x') as f:json.dump(value,f,indent=2)
files={str(p.relative_to(Path(__file__).parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
with (a.out/'implementation_snapshot.json').open('x') as f:json.dump(dict(parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources=files),f,indent=2)
print('CONFIGS_FROZEN',len(plan['arms']))
