import sys,csv,collections
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_nuisance_pilot.common import *
def main():
 out=ROOT/'delivery';out.mkdir(exist_ok=False);new=read(ROOT/'evaluation/per_trial_results.json');clean=read(ROOT/'clean_reuse.json')['rows'];assert len(new)==900 and all(r['status']=='COMPLETE' for r in new)
 rows=clean+new;assert len({(r['model'],r['seed'],r['condition']) for r in rows})==1050
 states=read(ROOT/'evaluation/evaluation_protocol.json')['states'];models=list(read(ROOT/'evaluation/policy_registry.json'));conditions=['clean']+[c['id'] for c in read(ROOT/'conditions.json')];lookup={(r['model'],r['seed'],r['condition']):int(r['success']) for r in rows};seeds=[31001,31002,31003];summary=[];pairs=[];rng=np.random.default_rng(31001);boot=rng.integers(0,25,size=(20000,25));results={}
 for seed in seeds:
  for c in conditions:
   C=np.array([lookup[f'C-low_seed{seed}_s1800',s,c] for s in states]);D=np.array([lookup[f'D-low_seed{seed}_s1800',s,c] for s in states]);delta=D-C
   pairs.append(dict(policy_seed=seed,condition=c,D_minus_C=float(delta.mean()),C_success_D_failure=int(((C==1)&(D==0)).sum()),D_success_C_failure=int(((D==1)&(C==0)).sum()),CI_low=float(np.quantile(delta[boot].mean(1),.025)),CI_high=float(np.quantile(delta[boot].mean(1),.975))))
   for method,arr in [('C-low',C),('D-low',D)]:
    ref=np.array([lookup[f'{method}_seed{seed}_s1800',s,'clean'] for s in states]);summary.append(dict(method=method,policy_seed=seed,condition=c,successes=int(arr.sum()),denominator=25,rate=float(arr.mean()),minus_clean=float((arr-ref).mean()),clean_success_to_failure=int(((ref==1)&(arr==0)).sum()),clean_failure_to_success=int(((ref==0)&(arr==1)).sum())))
 for kind,conds in [('clean',['clean']),('background',[f'background_{i}' for i in range(1,4)]),('lighting',[f'lighting_{i}' for i in range(1,4)])]:
  values={m:np.array([[[lookup[f'{m}_seed{seed}_s1800',s,c] for c in conds] for s in states] for seed in seeds]) for m in ['C-low','D-low']};delta=(values['D-low']-values['C-low']).mean((0,2));rates={m:v.mean((1,2)) for m,v in values.items()}
  results[kind]=dict(C_seed_rates=rates['C-low'].tolist(),D_seed_rates=rates['D-low'].tolist(),C_mean=float(rates['C-low'].mean()),D_mean=float(rates['D-low'].mean()),D_minus_C_by_seed=(rates['D-low']-rates['C-low']).tolist(),D_minus_C=float(delta.mean()),paired_state_CI=np.quantile(delta[boot].mean(1),[.025,.975]).tolist(),C_seed_SD=float(rates['C-low'].std(ddof=1)),D_seed_SD=float(rates['D-low'].std(ddof=1)),C_seed_range=[float(rates['C-low'].min()),float(rates['C-low'].max())],D_seed_range=[float(rates['D-low'].min()),float(rates['D-low'].max())])
  if kind!='clean':
   for m in ['C-low','D-low']:
    cr=np.array([np.mean([lookup[f'{m}_seed{seed}_s1800',s,'clean'] for s in states]) for seed in seeds]);results[kind][m+'_minus_clean_by_seed']=(rates[m]-cr).tolist()
 table(out/'success_by_seed_condition.csv',summary);table(out/'paired_flips.csv',pairs);save(out/'main_statistics.json',results)
 offline=list(csv.DictReader((ROOT/'offline_metrics.csv').open()));groups=collections.defaultdict(list)
 for r in offline:groups[r['model'],r['episode'],r['appearance']].append(r)
 ids=['model','episode','appearance','t','horizon','current_action','executed_chunk_member'];metrics=[k for k in offline[0] if k not in ids];reduced=[]
 for (model,episode,appearance),rs in groups.items():
  for subset,chosen in [('full_chunk',rs),('current_action',[r for r in rs if r['horizon']=='1'])]:
   reduced.append(dict(model=model,episode=episode,appearance=appearance,subset=subset,anchors=20,**{k:float(np.mean([float(r[k]) for r in chosen if r[k] not in ['',None]])) if any(r[k] not in ['',None] for r in chosen) else None for k in metrics}))
 table(out/'offline_by_demo.csv',reduced)
 table(out/'offline_metrics.csv',offline);save(out/'per_trial_summary.json',[{k:v for k,v in r.items() if k not in ['video','trajectory']} for r in rows])
 primary=results['background'];secondary=results['lighting'];lines=['# Frozen C/D new appearance exploration','', 'Completed 900 new valid trials; reused 150 matching Clean trials. No training or backward.', '', '## Main and secondary results','', '| Category | C mean | D mean | D-C | Paired-state 95% CI |','|---|---:|---:|---:|---|']
 for kind,x in results.items():lines.append(f"| {kind} | {100*x['C_mean']:.2f}% | {100*x['D_mean']:.2f}% | {100*x['D_minus_C']:.2f} pp | [{100*x['paired_state_CI'][0]:.2f}, {100*x['paired_state_CI'][1]:.2f}] pp |")
 lines+=['','## Full per-seed outcomes','','| Model | Seed | Condition | Success |','|---|---:|---|---:|']+[f"| {r['method']} | {r['policy_seed']} | {r['condition']} | {r['successes']}/25 |" for r in summary]
 lines+=['','## Interpretation boundaries','','Background is primary; lighting is secondary cross-intervention transfer, not an AE training intervention. Prior stove and drawer negative conclusions remain unchanged. These 25 states were previously scored: new-appearance exploration, not a new sealed test. All conditions/seeds retained. State bootstrap does not include policy-seed, AE, nuisance-sampling or task uncertainty. Per-demo offline summaries and all per-anchor/horizon values are supplied; raw latent MSE is not a cross-AE quality ranking. Lower action drift is not improvement unless absolute demonstration error is retained. No gradients or interventions were used to identify causes.','', 'A/B: see primary/secondary effect sizes, per-seed direction and Clean cost; no isolated seed or condition establishes efficacy. C: label, predicted-latent and action measurements are separated in offline_by_demo.csv. D: no automatic next experiment; independent replication is justified only as a candidate if aggregate gains are not driven by a single seed/condition and Clean cost is explicit. Otherwise retain the existing negative archive rather than tune to this pilot.']
 panel=[]
 for kind in ['background','lighting']:
  for method in ['C-low','D-low']:
   rs=[r for r in reduced if r['model'].startswith(method) and r['appearance'].startswith(kind) and r['subset']=='full_chunk']
   panel.append(dict(kind=kind,method=method,**{k:float(np.mean([r[k] for r in rs])) for k in ['AE_standardized_MSE_vs_clean','policy_standardized_latent_MSE_vs_clean','policy_latent_L1_own_target','translation_normalized_L1_vs_clean','rotation_normalized_L1_vs_clean','gripper_normalized_L1_vs_clean','translation_normalized_L1_demonstration','rotation_normalized_L1_demonstration','gripper_normalized_L1_demonstration']}))
 table(out/'offline_stage_summary.csv',panel)
 lines+=['','## Same-state measurement chain','','| Type / method | AE normalized MSE | Policy normalized latent MSE | Action drift xyz / rotation / gripper | Absolute action L1 xyz / rotation / gripper |','|---|---:|---:|---|---|']
 for r in panel:
  drift=' / '.join(f"{r[k+'_normalized_L1_vs_clean']:.5f}" for k in ['translation','rotation','gripper']);err=' / '.join(f"{r[k+'_normalized_L1_demonstration']:.5f}" for k in ['translation','rotation','gripper'])
  lines.append(f"| {r['kind']} / {r['method']} | {r['AE_standardized_MSE_vs_clean']:.5f} | {r['policy_standardized_latent_MSE_vs_clean']:.5f} | {drift} | {err} |")
 c,d=panel[:2];label_gain=d['AE_standardized_MSE_vs_clean']<c['AE_standardized_MSE_vs_clean'];policy_gain=d['policy_standardized_latent_MSE_vs_clean']<c['policy_standardized_latent_MSE_vs_clean']
 lines+=['',f'Background normalized label distance decreases in D: {label_gain}; predicted-latent distance decreases in D: {policy_gain}. These are finite-sample associations, not causal mediation. Movement axes and gripper errors must be read separately; no mixed physical-unit error is used.']
 if primary['D_minus_C']>0 and all(x>0 for x in primary['D_minus_C_by_seed']) and primary['paired_state_CI'][0]>0:recommendation='One priority: independent matched training replication, explicitly retaining Clean cost and all nuisance conditions. Current fixed-AE development evidence is only a candidate.'
 elif label_gain and not policy_gain:recommendation='One priority: consider a separately specified diagnostic of how auxiliary supervision is used; label-distance improvement did not propagate to predicted-latent distance here. This is not permission for a new method or training run.'
 else:recommendation='One priority: retain/archive the current implementation and this exploratory result; no consistent chain or independently established control benefit justifies automatic further tuning.'
 lines+=['','## Unique next-step recommendation','',recommendation]
 (out/'report.md').write_text('\n'.join(lines)+'\n');save(out/'delivery_verification.json',dict(status='COMPLETE',valid_new_rollouts=900,actual_attempts=900,invalid_trials=0,reused_clean=150,optimizer_updates=0,AE_updates=0,DINO_updates=0,backward=0,offline_policy=read(ROOT/'offline_policy_acceptance.json'),offline_AE=read(ROOT/'offline_ae_acceptance.json')))
if __name__=='__main__':main()
