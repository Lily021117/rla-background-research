import sys,json,subprocess,collections,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_scratch_low_seeds.common import *
from rla_sampling_stability.experiment import balanced
def main():
 assert read(PRIOR/'github_sync_final.json')['status']=='PUSHED_VERIFIED'
 assert read(PRIOR/'delivery/delivery_verification.json')['new_valid_rollouts']==200
 # Inspect experiment metadata only, never trajectory / sealed state contents.
 used=set();sources=[]
 names={'protocol.json','replication_protocol.json','planned_run_plan.json','run_plan.json','initialization_gate.json'}
 def collect(v):
  if isinstance(v,dict):
   for k,x in v.items():
    if 'seed' in k.lower() and isinstance(x,int):used.add(x)
    elif 'seed' in k.lower() and isinstance(x,list):used.update(n for n in x if isinstance(n,int))
    collect(x)
  elif isinstance(v,list):
   for x in v:collect(x)
 for parent,dirs,files in os.walk(P/'experiments'):
  dirs[:]=[d for d in dirs if d not in ['code','.git','closed_loop','targets','sidecars','videos','__pycache__','data'] and 'sealed' not in d.lower()]
  for name in set(files)&names:
   p=Path(parent)/name
   if p.stat().st_size>5_000_000:continue
   try:collect(read(p));sources.append(str(p))
   except (ValueError,OSError):pass
 seeds=[]
 for proposed in [31001,31002,31003]:
  s=proposed
  while s in used or s in seeds:s+=1
  seeds.append(s)
 save(ROOT/'seed_registry_audit.json',dict(proposed=[31001,31002,31003],selected=seeds,previously_registered=sorted(used),metadata_sources=sources,no_policy_results_used=True))
 protected={};inventory=read(BASE/'encoder_inventory.json')
 for a,v in inventory.items():
  assert int(v['step'])==2000 and sha(v['path'])==v['sha256'];protected[v['path']]=v['sha256']
 for r in read(BASE/'target_manifest.json'):
  assert sha(r['path'])==r['sha256'];protected[r['path']]=r['sha256']
 for p in [BASE/'targets/C/statistics.json',BASE/'targets/D/statistics.json',BASE/'target_acceptance.json',BASE/'encoder_inventory.json',PRIOR/'delivery/per_trial_results.json',PRIOR/'delivery/delivery_verification.json']:
  protected[str(p)]=sha(p)
 original=read(BASE/'seed_20261101/run_plan.json');manifest=read(original['inputs']['data_manifest'])
 lengths={e['episode']:e['valid_chunks'] for e in manifest['episodes'] if e['split']=='train'}
 assert len(lengths)==20 and sum(lengths.values())==2555
 for key in ['data_manifest','action_proprio_statistics']:
  p=original['inputs'][key];protected[p]=sha(p)
 protected[original['common_step0']]=sha(original['common_step0'])
 for s in seeds:
  root=ROOT/f'seed_{s}';order=balanced(lengths,s);save(root/'sampling_order.json',order)
  trace=root/'expected_sample_ids.jsonl'
  with trace.open('x') as f:
   for k in range(600):f.write(json.dumps(dict(step=k+1,actual_sample_ids=[[e,t,c] for e,t in order[k*8:(k+1)*8] for c in ['clean','background']]))+'\n')
  plan=read(BASE/'seed_20261101/run_plan.json')
  plan.update(experiment='scratch_AE_fixed_low_full_policy_seed_replication',arms=ARMS,policy_seed=s,initialization_scope='all_random_modules_new_seed',original_common_step0=original['common_step0'],original_common_step0_sha256=original['common_step0_sha256'],common_step0=str(root/'common_seed_step0.pt'),common_step0_sha256='PENDING_ZERO_UPDATE_BUILD',paired_C_sample_trace=str(trace),target_source_by_arm={'C-low':'C','D-low':'D'},closed_loop_trials=200)
  plan['inputs'].update(B20_sampling_order=str(root/'sampling_order.json'),B20_sampling_sha256=sha(root/'sampling_order.json'))
  plan['shared_protocol'].update(seed=s,lambda_aux=LAMBDA,initialization='strict pretrained backbone; constructor-random modules rebuilt by seed, constant initializers preserved',sampler='unchanged episode-balanced algorithm; independent RandomState(seed)')
  save(root/'planned_run_plan.json',plan)
  save(root/'sampling_audit.json',dict(seed=s,physical_exposures=4800,appearance_exposures=9600,episodes=20,valid_windows=2555,episode_counts=dict(collections.Counter(e for e,t in order)),sampling_sha256=sha(root/'sampling_order.json'),expected_trace_sha256=sha(trace)))
 ep=read(PRIOR/'evaluation/evaluation_protocol.json');ep.pop('C_low_completion',None)
 registry={f'{a}_seed{s}_s600':str(ROOT/f'seed_{s}/{a}/formal/step0600.pt') for s in seeds for a in ARMS}
 ep.update(models=list(registry),trials=600,low_seed_replication=True,gate='all six models all five appearances unconditionally',role='fixed AE policy training seed replication on reused exploratory development states')
 save(ROOT/'evaluation/evaluation_protocol.json',ep);save(ROOT/'evaluation/policy_registry.json',registry)
 frozen=read(ep['frozen_state_protocol']);protected[ep['frozen_state_protocol']]=sha(ep['frozen_state_protocol'])
 for st in frozen['new_development_states']:
  protected[st['state_file']]=sha(st['state_file'])
  for c in st['conditions']:
   p=Path(st['state_file']).parent/(c['condition']+'.xml');assert sha(p)==c['XML_sha256'];protected[str(p)]=c['XML_sha256']
 save(ROOT/'protected_inputs.json',protected)
 save(ROOT/'protocol.json',dict(seeds=seeds,arms=ARMS,updates_each=600,total_updates=3600,valid_clean=120,valid_background=480,total_valid_rollouts=600,AE_updates=0,lambda_low=LAMBDA,initialization='one complete new constructor initialization per seed, exact paired checkpoint and RNG; frozen pretrained parameters identical between seeds',sampler='unchanged episode-balanced algorithm; independent sampler RNG with the same seed value',augmentation='unchanged deterministic official derived image preprocessing; train stochasticity seeded',scheduler='historical MultiStepLR milestone600 after optimizer; actual LR2e-4 at all600 updates',microbatch=8,accumulation=2,effective_appearance_batch=16,backend='BF16 Math SDPA activation checkpointing',checkpoints=[0,1,100,300,600],validation=[0,100,300,600],diagnostics=[0,300,600],target='own appearance, own existing per-position training mean and global scale; unchanged',primary='per-seed Clean and paired D-low minus C-low',statistics='three new full policy seeds only; sample SD ddof1; state-cluster bootstrap resamples all appearances, models and seeds together, not training-seed uncertainty',historical_orders='separate reference only; not pooled',stop='no AE/extra seeds/hyperparameter changes; hardware failure stops, no silent retry',sealed_read=False,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip()))
 print('FROZEN',seeds,flush=True)
if __name__=='__main__':main()
