import sys,subprocess,collections
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_drawer_baseline_gap.common import *
def main():
 assert read(DATA/'data_acceptance_audit.json')['status']=='PASS'
 manifest=read(DATA/'frozen_dataset_manifest.json');assert manifest['status']=='PASS'
 assert manifest['task_id']=='libero_goal/open_the_middle_drawer_of_the_cabinet'
 train=[e for e in manifest['episodes'] if e['split']=='train'];dev=[e for e in manifest['episodes'] if e['split']=='validation']
 assert len(train)==20 and len(dev)==5
 assert not {e['source_demo'] for e in train}&{e['source_demo'] for e in dev}
 enc=AE/'C/step2000.pt';expected='8bfa58c22b906e200f511a8bb3876e7b5bb98f5a15ad424e59ec93508c1e8278';assert sha(enc)==expected
 save(ROOT/'encoder_inventory.json',dict(path=str(enc),sha256=expected,global_step=2000,identity='scratch common-untrained-constructor AE-C formal final',trained_tasks=['cookie','drawer_bowl','stove'],drawer_open_training=False,role='frozen representation transfer',AE_updates=0,DINO_updates=0,configuration=str(AE/'precalibration_plan.json'),configuration_sha256=sha(AE/'precalibration_plan.json')))
 protected={str(DATA/n):sha(DATA/n) for n in ['data_acceptance_audit.json','frozen_dataset_manifest.json','normalization_statistics_expanded_train_only.json','development_protocol/evaluation_protocol.json','background_protocol.json']}
 states=read(DATA/'development_protocol/evaluation_protocol.json');assert states['status']=='FROZEN_STATIC_CHECKED' and len(states['new_development_states'])==20
 for s in states['new_development_states']:
  p=Path(s['state_file']);v=np.load(p);assert hashlib.sha256(v.tobytes()).hexdigest()==s['state_sha256'];protected[str(p)]=sha(p)
  for c in s['conditions']:
   p=Path(s['state_file']).parent/(c['condition']+'.xml');assert sha(p)==c['XML_sha256'];protected[str(p)]=sha(p)
 source=Path(train[0]['source_hdf5']);assert sha(source)=='20252c7cf98cd7437061f7f200ae7b6cb6219fabbd53b4536dfaa8abda6ab737';protected[str(source)]=sha(source)
 save(ROOT/'protected_inputs.json',protected)
 save(ROOT/'data_source_audit.json',dict(status='PASS_REUSE',acceptance=read(DATA/'data_acceptance_audit.json'),source=str(source),task=manifest['task_id'],train=[e['source_demo'] for e in train],development=[e['source_demo'] for e in dev],old_root_BLOCKED_report='superseded by accepted dataset_v2; wrong middle_joint identifier fixed historically',horizons=list(range(1,9)),no_new_render_or_sealed_reads=True))
 base=read(SOURCE/'seed_20261101/run_plan.json')
 for seed in SEEDS:
  root=ROOT/f'seed_{seed}';lengths={e['episode']:e['valid_chunks'] for e in train};rg=np.random.RandomState(seed);pending={e:[] for e in lengths};order=[]
  for _ in range(720):
   for j in rg.permutation(20):
    e=list(lengths)[j]
    if not pending[e]:pending[e]=list(map(int,rg.permutation(lengths[e])))
    order.append([e,pending[e].pop()])
  assert len(order)==14400 and set(collections.Counter(e for e,t in order).values())=={720}
  save(root/'sampling_order.json',order)
  plan=read(SOURCE/'seed_20261101/run_plan.json');plan.update(experiment='drawer_baseline_gap',previous=str(ROOT/'sidecars'),common_step0=str(root/'rla_step0.pt'),common_step0_sha256='PENDING_INITIALIZATION',policy_seed=seed,checkpoint_steps=[0,300,600,900,1200,1500,1800],selection='final1800 only',standard_gate=str(root/'preflight-standard/formal/preflight_acceptance.json'),rla_gate=str(root/'preflight-rla/formal/preflight_acceptance.json'))
  plan['inputs'].update(B20_step0=str(root/'standard_step0.pt'),B20_sampling_order=str(root/'sampling_order.json'),B20_sampling_sha256=sha(root/'sampling_order.json'))
  plan['shared_protocol'].update(successful_updates=1800,lambda_aux=LAMBDA,model_seed=seed,sampler_seed=seed,physical_exposures=14400,appearance_exposures=28800,initialization='strict pretrained backbone + full seed constructor initialization; compatible standard/RLA head keys copied',lr_schedule='updates1..600 2e-4;601..1800 2e-5; MultiStepLR600 AFTER optimizer')
  save(root/'planned_run_plan.json',plan)
 save(ROOT/'policy_execution_plan.json',dict(methods=METHODS,orders=SEEDS,maximum_control_steps=300,updates=10800,clean_trials=120,background_trials_per_passed_method=240,maximum_valid_trials=600,gate='per method: all three seeds Clean>=16/20'))
 save(ROOT/'protocol.json',dict(task=manifest['task_id'],methods=METHODS,seeds=SEEDS,full_policy_seeds=True,updates_per_model=1800,total_policy_updates=10800,AE_updates=0,DINO_updates=0,lambda_aux=LAMBDA,LR={'updates1_600':.0002,'updates601_1800':.00002},optimizer='AdamW betas0.9/0.999 eps1e-8 weight_decay0.01; no clipping',scheduler='MultiStepLR milestone600 gamma0.1 after optimizer; no additional decay',physical_batch=8,appearance_batch=16,microbatch=8,accumulation=2,dtype='BF16',backend='Math SDPA',activation_checkpointing=True,checkpoints=[0,300,600,900,1200,1500,1800],validation=[0,600,1200,1800],closed_loop_checkpoint=1800,train_episodes=20,development_episodes=5,train_windows=sum(e['valid_chunks'] for e in train),sampling='independent RandomState(seed), episode-balanced complete window cycles; no development selection',normalization='action/proprio existing drawer train-only statistics; latent new AE-C per-position mean + global raw std over all train windows/horizons/both appearances',state_protocol=str(DATA/'development_protocol/evaluation_protocol.json'),clean_gate='separately for each method, all three seeds >=16/20',background_gap_candidate='all three seed background means < respective Clean AND equal-seed mean decrease >=0.10',statistics='state-cluster paired bootstrap preserves all backgrounds and seeds; does not cover training-seed/AE/task uncertainty',stop='numerical/CUDA/ECC failure stops and preserves; no silent retry, no valid failure rerun; no extension',architecture_confounding=True,initialization_matching='same pretrained VLA, LoRA, action queries, proprio and compatible head keys; RLA extra modules seed+1 in fork_rng; common saved RNG, shapes consume RNG differently during forward so exact trajectories not claimed',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip(),sealed_reads=0))
 save(ROOT/'historical_result_archive.json',dict(source='experiments/rla_frozen1800_independent_states/run_20261031_02/delivery/report.md',commit='fa7a90f21df2ed07a33a8c9b8845335c8bc5f7d3',conclusion='D background advantage did not replicate on 50 newly generated stove states',background_D_minus_C_pp=-.6666666666667,used_for_selection=False,history_unchanged=True))
 print('PROTOCOL_FROZEN',flush=True)
if __name__=='__main__':main()
