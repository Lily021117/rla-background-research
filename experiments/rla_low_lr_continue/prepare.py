import sys,json,csv,collections,inspect,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_low_lr_continue.common import *
import torch,numpy as np
from rla_sampling_stability.experiment import balanced
def extended(lengths,seed):
 rng=np.random.RandomState(seed);episodes=list(lengths);pending={e:[] for e in episodes};order=[]
 for cycle in range(720):
  for j in rng.permutation(len(episodes)):
   e=episodes[j]
   if not pending[e]:pending[e]=list(map(int,rng.permutation(lengths[e])))
   order.append([e,pending[e].pop()])
 assert order[:4800]==balanced(lengths,seed)
 assert len(order)==14400 and set(collections.Counter(e for e,t in order).values())=={720}
 return order
def main():
 assert read(SOURCE/'completion.json')['successful_policy_updates']==3600
 inventory=list(csv.DictReader((SOURCE/'delivery/checkpoint_inventory.csv').open()));selected=[x for x in inventory if int(x['step'])==600];assert len(selected)==6
 protected={str(SOURCE/'delivery/checkpoint_inventory.csv'):sha(SOURCE/'delivery/checkpoint_inventory.csv'),str(SOURCE/'delivery/per_trial_results.json'):sha(SOURCE/'delivery/per_trial_results.json')}
 # Reuse frozen data/AE/sidecar provenance, verify all hashes before launch.
 for p,h in read(SOURCE/'protected_inputs.json').items():assert sha(p)==h;protected[p]=h
 audits=[];prefixes=[];registry={}
 for seed in SEEDS:
  sourceplan=read(SOURCE/f'seed_{seed}/run_plan.json');manifest=read(sourceplan['inputs']['data_manifest']);lengths={e['episode']:e['valid_chunks'] for e in manifest['episodes'] if e['split']=='train'}
  assert len(lengths)==20 and sum(lengths.values())==2555
  order=extended(lengths,seed);oldorder=read(sourceplan['inputs']['B20_sampling_order']);assert order[:4800]==oldorder
  seedroot=ROOT/f'seed_{seed}';save(seedroot/'sampling_order_1800.json',order)
  tracefile=seedroot/'expected_sample_ids.jsonl'
  with tracefile.open('x') as f:
   for k in range(1800):f.write(json.dumps(dict(step=k+1,actual_sample_ids=[[e,t,c] for e,t in order[k*8:(k+1)*8] for c in ['clean','background']]))+'\n')
  expected=[json.loads(x) for x in tracefile.read_text().splitlines()]
  for arm in ARMS:
   record=next(x for x in selected if int(x['seed'])==seed and x['method']==arm);path=Path(record['path']);assert sha(path)==record['sha256'];protected[str(path)]=record['sha256']
   ck=torch.load(path,map_location='cpu');assert ck['configuration']==sourceplan and ck['global_step']==600 and ck['next_physical_sample']==4800 and ck['auxiliary_schedule_counter']==600
   assert ck['arm']==arm and ck['AMP_scaler'] is None and ck['dtype']=='bfloat16'
   for key in ['torch_rng','cuda_rng','python_rng','numpy_rng']:assert key in ck
   assert ck['scheduler']['last_epoch']==600 and dict(ck['scheduler']['milestones'])=={600:1}
   assert ck['scheduler']['gamma']==.1 and all(v==2e-5 for v in ck['scheduler']['_last_lr'])
   steps=sorted({int(v['step']) for v in ck['optimizer']['state'].values()});assert steps==[600]
   groups=ck['optimizer']['param_groups'];assert all(g['lr']==2e-5 and g['initial_lr']==2e-4 and g['weight_decay']==.01 for g in groups)
   assert all(torch.isfinite(v).all() for m in ck['parameters'].values() for v in m.values())
   rows=list(csv.DictReader((path.parent/'training_metrics.csv').open()));trace=[json.loads(x) for x in (path.parent/'actual_sample_ids.jsonl').read_text().splitlines()]
   if len(rows)==599:
    assert (seed,arm)==(31002,'D-low')
    prior=SOURCE/f'seed_{seed}/{arm}/formal';rows=[next(csv.DictReader((prior/'training_metrics.csv').open()))]+rows;trace=[json.loads((prior/'actual_sample_ids.jsonl').read_text().splitlines()[0])]+trace
   assert len(rows)==len(trace)==600 and all(float(x['lr'])==2e-4 for x in rows)
   assert all(a['actual_sample_ids']==b['actual_sample_ids'] and a['step']==b['step'] for a,b in zip(trace,expected[:600]))
   root=seedroot/arm;plan=json.loads(json.dumps(sourceplan))
   plan.update(experiment='fixed_AE_half_weight_step600_to1800_low_LR_continuation',common_step0=str(path),common_step0_sha256=record['sha256'],resume_checkpoint=str(path),source_run_plan=str(SOURCE/f'seed_{seed}/run_plan.json'),source_acceptance=str(path.parent/'training_acceptance.json'),source_step600_directory=str(path.parent),paired_C_sample_trace=str(tracefile),continuation_start=600,continuation_end=1800)
   plan['inputs'].update(B20_sampling_order=str(seedroot/'sampling_order_1800.json'),B20_sampling_sha256=sha(seedroot/'sampling_order_1800.json'))
   plan['shared_protocol'].update(successful_updates=1800,new_successful_updates=1200,lr=2e-5,physical_window_exposures=14400,appearance_exposures=28800,common_initial_checkpoint='inventory-resolved full step600 (legacy common_step0 field)',scheduler='restore MultiStepLR last_epoch600; milestone600 already applied; next steps601..1800 no additional decay')
   save(root/'run_plan.json',plan)
   audit=dict(seed=seed,arm=arm,path=str(path),sha256=record['sha256'],global_step=600,optimizer_steps=steps,scheduler_last_epoch=600,optimizer_groups=groups,first600_LR=2e-4,next_LR=2e-5,all_RNG_present=True,sampling_cursor=4800,bfloat16_no_scaler=True,configuration_matches_source=True,finite_parameters=True,prior_replays=49 if (seed,arm)==(31002,'D-low') else 0)
   save(root/'checkpoint_audit.json',audit);audits.append(audit)
   registry[f'{arm}_seed{seed}_s1800']=str(root/arm/'formal/step1800.pt')
   del ck
  prefixes.append(dict(seed=seed,full_length=14400,historical_prefix_length=4800,prefix_matches_both_actual_traces=True,independent_sampler_RNG=True,new_sequence_sha256=sha(seedroot/'sampling_order_1800.json'),expected_trace_sha256=sha(tracefile),same_episode_order=True,continued_pending_windows_not_restarted=True))
 ep=read(SOURCE/'evaluation/evaluation_protocol.json');ep.update(models=list(registry),trials=600,low_lr_continuation=True,checkpoint_selection='step1800 only; historical600 reused, never rerun')
 save(ROOT/'evaluation/evaluation_protocol.json',ep);save(ROOT/'evaluation/policy_registry.json',registry)
 save(ROOT/'evaluation/run_plan.json',read(ROOT/'seed_31001/C-low/run_plan.json'))
 save(ROOT/'source_checkpoint_audit.json',audits);save(ROOT/'sampling_prefix_audit.json',prefixes);save(ROOT/'protected_inputs.json',protected)
 save(ROOT/'protocol.json',dict(source_commit='9f2a33876542fda15b4c872a1757ab94c57f73a4',seeds=SEEDS,arms=ARMS,global_start=600,global_end=1800,new_updates_each=1200,total_new_updates=7200,LR=2e-5,lambda_RLA=LAMBDA,scheduler='restore original MultiStepLR state after600; no reset, no repeated milestone, no new decay',checkpoints=[900,1200,1500,1800],validation=[900,1200,1500,1800],step600_validation='reuse source, no full re-evaluation; only fixed-batch resume equality check',new_valid_clean=120,new_valid_background=480,new_valid_trials=600,AE_updates=0,sealed_read=False,primary=['C-low1800 minus C-low600','D-low1800 minus D-low600','D-low1800 minus C-low1800'],statistical_unit='physical state, all appearances/methods/seeds jointly resampled; sample SD/range over3 seeds separately',interpretation='more updates PLUS prescribed low-LR phase, NOT pure step-count ablation',stop='only1800 endpoint, no additional budget/seed/loss/AE; hardware ECC increments stop',sampler_source_sha256=sha(HERE.parent/'rla_sampling_stability/experiment.py'),GPU=GPU))
 table(ROOT/'LR_curve.csv',[dict(seed=s,method=a,global_update=k,actual_LR=2e-5,lambda_RLA=LAMBDA) for s in SEEDS for a in ARMS for k in range(601,1801)])
 print('CONTINUATION_FROZEN_ALL_SIX_PASS',flush=True)
if __name__=='__main__':main()
