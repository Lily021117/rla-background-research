"""User-requested bounded200 revision; preserve interrupted long-run artifacts."""
import argparse,json,csv,hashlib,subprocess
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();o=a.out;o.mkdir(parents=True,exist_ok=False)
def read(p):return json.loads(Path(p).read_text())
def save(n,x):
 with (o/n).open('x') as f:json.dump(x,f,indent=2)
plan=read(a.source/'run_plan.json');original=a.source/'C_m0p1/formal';logged=list(csv.DictReader((original/'training_metrics.csv').open()));assert int(logged[-1]['step'])==416
ck=torch.load(original/'step0400.pt',map_location='cpu');assert ck['global_step']==400 and ck['next_physical_sample']==3200 and ck['scheduler']['last_epoch']==400
assert all(int(s['step'])==400 for s in ck['optimizer']['state'].values());del ck
save('interruption_audit.json',dict(reason='user shortened experiment; explicitly stopped only recorded task PIDs2010972/2010978/2010979',confirmed_successful_updates=416,last_complete_recovery_step=400,unpersisted_logged_updates=16,inflight_attempt_at_stop='possible417; no confirmed successful-update log, not counted',replays=0,original_paths_preserved=str(a.source),no_rollback_or_further_training_C_m0p1=True,new_closed_loop=0))
plan.update(experiment='short_lr_and_two_stage_curves',updates=200,closed_loop_trials=0,checkpoints=[0,50,100,200],validation=[0,50,100,200],supersedes_long_run=True,long_run_source=str(a.source))
plan['shared_protocol'].update(successful_updates=200,physical_window_exposures=1600,appearance_exposures=3200,policy_updates_this_run=1400,scheduler='MultiStepLR milestones=[1200], gamma0.1 after update; no warmup',selection='offline screening only; no closed-loop hyperparameter selection this round')
plan.pop('selection',None);plan['diagnostic_nodes']=[0,100,200];plan['fixed_eval_manifest']=str(o/'fixed_sample_manifest.json')
manifest=read(plan['inputs']['data_manifest']);samples=[]
for e in manifest['episodes']:
 name=e.get('episode',e.get('episode_id',e.get('source_episode')))
 assert name is not None,e.keys()
 n=e['valid_chunks'];k=5 if e['split']=='train' else 20
 for t in sorted(set(round(i*(n-1)/(k-1)) for i in range(k))):
  for c in ('clean','background'):samples.append(dict(episode=name,timestep=t,appearance=c,split=e['split']))
save('fixed_sample_manifest.json',dict(rule='5 evenly spaced valid anchors per train episode;20 per development; both appearances; no model outputs used',samples=samples,gradient_samples=[dict(episode=e,timestep=t,appearance=c) for e,t in read(plan['inputs']['B20_sampling_order'])[:4] for c in ('clean','background')],stage_labels='no validated approach/grasp/transport/descent frame labels; only historical pre/postlift labels may be referenced'))
save('run_plan.json',plan);save('normalization.json',read(a.source/'normalization.json'));save('gate_acceptance.json',read(a.source/'gate_acceptance.json'));save('historical_optimizer_audit.json',read(a.source/'historical_optimizer_audit.json'));save('protected_inputs.json',read(a.source/'protected_inputs.json'))
save('checkpoint_registry.json',{arm:{str(step):str((original if arm=='C_m0p1' else o/arm/'formal')/f'step{step:04}.pt') for step in (0,50,100,200) if arm!='C_m0p1' or (original/f'step{step:04}.pt').exists()} for arm in plan['arms']})
save('short_protocol.json',dict(target200=True,scheduler_milestone1200=True,already_over_budget={'C_m0p1':416},new_authorized_updates=1400,expected_confirmed_total=1816,normal_total_without_prior_excess=1600,gradient_nodes=[0,100,200],gradient_updates=0,rollouts=0,sealed_reads=0,missing_C_m0p1_step50='no checkpoint; do not regenerate'))
for arm,cfg in plan['arms'].items():save('training_config_'+arm+'.json',dict(**cfg,updates=200,scheduler=plan['shared_protocol']['scheduler'],initialization=plan['common_step0'],lambda_aux=plan['shared_protocol']['lambda_aux'],seed=plan['seed'],batch16_micro8_accum2=True))
print('SHORT_PROTOCOL_FROZEN',len(samples),flush=True)
