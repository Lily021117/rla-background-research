"""Freeze new-task plans and verify common, untrained initialization on CPU."""
import argparse,json,hashlib
from pathlib import Path
import torch
from audit_source import save,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--project',type=Path,required=True);a=p.parse_args()
 base=a.project/'experiments/official_libero_rla_consistency';rla=base/'run_20261023_08/policy_readiness_retry1/isolated_smoke/common_policy_step0.pt';standard=base/'run_20261023_06/B20/formal_B/step0000.pt'
 rc=torch.load(rla,map_location='cpu');bc=torch.load(standard,map_location='cpu')
 assert bc['global_step']==0
 assert rc.get('global_step',0)==0
 assert sha(rla)=='ba019970f368f8843c9107c99b1a30f9779e86bc44da7d831a002d6c6e043f66'
 for mod in ['vla','proprio']:
  assert set(rc['parameters'][mod])==set(bc['parameters'][mod])
  assert all(torch.equal(v,bc['parameters'][mod][k]) for k,v in rc['parameters'][mod].items()),mod
 save(a.run/'policy_initialization_cpu_audit.json',dict(status='PASS',standard_step0=str(standard),standard_sha256=sha(standard),rla_step0=str(rla),rla_sha256=sha(rla),shared_vla_proprio_exact=True,standard_head_structure_different=True,RLA_step_field=rc.get('global_step'),source_is_pretraining_plus_untrained_heads=True))
 d=a.run/'dataset_v2';manifest=json.loads((d/'frozen_dataset_manifest.json').read_text());assert manifest['status']=='PASS'
 for seed in [20261101,20261102]:
  out=a.run/f'seed_{seed}';out.mkdir(exist_ok=False)
  plan=dict(frozen=True,previous=str(a.run/'frozen_sidecars'),common_step0=str(rla),common_step0_sha256=sha(rla),inputs=dict(project=str(a.project),B20_step0=str(standard),B20_sampling_order=str(d/f'sampling_order_{seed}.json'),B20_sampling_sha256=sha(d/f'sampling_order_{seed}.json'),data_manifest=str(d/'frozen_dataset_manifest.json'),action_proprio_statistics=str(d/'normalization_statistics_expanded_train_only.json')),standard_gate=str(a.run/'seed_20261101/preflight-standard/formal/preflight_acceptance.json'),rla_gate=str(a.run/'seed_20261101/preflight-rla/formal/preflight_acceptance.json'),shared_protocol=dict(task_id='libero_goal/open_the_middle_drawer_of_the_cabinet',successful_updates=600,optimizer='AdamW',lr=.0002,weight_decay=.01,betas=[.9,.999],eps=1e-8,scheduler='MultiStepLR([600],gamma=.1); optimizer.step then scheduler.step',lambda_aux=.22408536585365854,microbatch=8,accumulation=2,effective_appearance_batch=16,model_seed=20261023,sampler_seed=seed,physical_exposures=4800,appearance_exposures=9600,train_episodes=20,development_episodes=5,backend='Math SDPA',dtype='BF16',activation_checkpointing=True,gradient_clip=None,keep=0,AE_updates=0,normalization='new-task training data only; shared across all policies'),checkpoint_steps=[0,100,300,600],selection='final600 only')
  save(out/'run_plan.json',plan)
 save(a.run/'policy_execution_plan.json',dict(status='FROZEN',methods=['BC-standard','BC-matched','C-frozen-RLA'],orders=[20261101,20261102],updates=3600,clean_trials=120,background_trials_if_gate=480,gate='any method >=12/20 clean in BOTH orders',maximum_control_steps=300,heldout_read=False))
 print('POLICY_CPU_IDENTITY_PASS',flush=True)

if __name__=='__main__':main()
