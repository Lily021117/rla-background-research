"""Clarify inherited descriptive fields without modifying frozen runtime plans."""
import argparse
import copy
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
raw=json.loads((a.out/'run_plan.json').read_text());clean=copy.deepcopy(raw);ignored={}
for key in ('common_step0','common_step0_sha256','target_center_definition'):
 if key in clean:ignored[key]=clean.pop(key)
for key in ('B20_step0','no_trained_policy_initialization'):
 if key in clean['inputs']:ignored['inputs.'+key]=clean['inputs'].pop(key)
for key in ('center_only_difference','C_D_only_target_source_differs','RLA_auxiliary_baseline'):
 if key in clean['shared_protocol']:ignored['shared_protocol.'+key]=clean['shared_protocol'].pop(key)
clean['target_center_definition']={m:('(z_C-mu_C)/s_C' if cfg['target']=='C' else '(z_D-mu_D)/s_C') for m,cfg in clean['arms'].items()}
audit=dict(status='PASS_RUNTIME_CONFIGURATION_METADATA_CLARIFIED',runtime_plan_unchanged=True,model_data_loss_or_training_change=False,
  actual_initialization='trained original C600 parameters; reset only independent auxiliary readout; new optimizer',
  authoritative_initialization_fields=['teacher_checkpoint','teacher_sha256','common_student_step0'],
  actual_model_loading_evidence='interface.initialize uses only teacher_checkpoint or common_student_step0; formal initial_state_audit verifies actual parameter hash',
  target_evidence='training.py selects arms[arm].target; LowWeightStore uses mu_C for C, mu_D for D, shared s_C',
  inherited_descriptive_fields_not_used_to_execute=ignored,
  caution='Original runtime JSON/checkpoints retain inherited provenance text; resolved files remove that ambiguity and do not replace the execution record.')
for name,value in [('resolved_run_plan.json',clean),('configuration_semantics_audit.json',audit)]:
 with (a.out/name).open('x') as f:json.dump(value,f,indent=2)
for m in clean['arms']:
 cfg=json.loads((a.out/('training_config_'+m+'.json')).read_text());cfg['protocol']=clean['shared_protocol']
 with (a.out/('resolved_training_config_'+m+'.json')).open('x') as f:json.dump(cfg,f,indent=2)
print('RESOLVED_METADATA_EXPORTED; runtime plan/checkpoints unchanged')
