"""Fixed paired-state analysis; preserve seeds, do not impute unrun backgrounds."""
import sys,csv,collections
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_drawer_baseline_gap.common import *
def table(p,rows):
 with p.open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 out=ROOT/'delivery';out.mkdir(exist_ok=False)
 rows=read(ROOT/'evaluation/per_trial_results.json');assert all(r['status']=='COMPLETE' for r in rows)
 assert len({(r['model'],r['seed'],r['condition']) for r in rows})==len(rows)
 states=sorted({r['seed'] for r in rows});assert len(states)==20
 conditions=['clean']+[f'new_background_{i}' for i in range(1,5)]
 lookup={(r['model'],r['seed'],r['condition']):int(r['success']) for r in rows}
 summaries=[];method_results={};rg=np.random.default_rng(20261101);boot=rg.integers(0,20,size=(20000,20))
 for method in METHODS:
  clean=[];background=[]
  for seed in SEEDS:
   key=f'{seed}_{method}';cl=np.array([lookup[key,s,'clean'] for s in states]);clean.append(cl)
   for cond in conditions:
    present=[lookup.get((key,s,cond)) for s in states]
    if all(x is None for x in present):continue
    assert None not in present
    ar=np.array(present);delta=ar-cl
    summaries.append(dict(method=method,policy_seed=seed,appearance=cond,successes=int(ar.sum()),denominator=20,success_rate=float(ar.mean()),minus_clean=float(delta.mean()),clean_success_to_failure=int(((cl==1)&(ar==0)).sum()),clean_failure_to_success=int(((cl==0)&(ar==1)).sum()),state_bootstrap_low=float(np.quantile(delta[boot].mean(1),.025)),state_bootstrap_high=float(np.quantile(delta[boot].mean(1),.975))))
   if all((key,states[0],c) in lookup for c in conditions[1:]):background.append(np.array([[lookup[key,s,c] for c in conditions[1:]] for s in states]))
  clean=np.array(clean);passed=bool(np.all(clean.sum(1)>=16));assert passed==(len(background)==3)
  result=dict(clean_seed_rates=clean.mean(1).tolist(),clean_mean=float(clean.mean()),clean_SD=float(clean.mean(1).std(ddof=1)),gate_passed=passed)
  if passed:
   bg=np.array(background);delta=bg.mean(2)-clean;state_delta=delta.mean(0)
   result.update(background_seed_means=bg.mean((1,2)).tolist(),background_mean=float(bg.mean()),background_SD=float(bg.mean((1,2)).std(ddof=1)),background_minus_clean_by_seed=delta.mean(1).tolist(),background_minus_clean_mean=float(delta.mean()),paired_state_CI=np.quantile(state_delta[boot].mean(1),[.025,.975]).tolist(),gap_candidate=bool(np.all(delta.mean(1)<0) and delta.mean()<=-.10))
  method_results[method]=result
 table(out/'success_by_seed_appearance.csv',summaries);save(out/'method_summary.json',method_results)
 save(out/'per_trial_summary.json',[{k:v for k,v in r.items() if k not in ['trajectory','video']} for r in rows])
 inventory=[];curves=[];valid=[]
 for seed in SEEDS:
  for method in METHODS:
   folder=ROOT/f'seed_{seed}/{method}/formal';accept=read(folder/'training_acceptance.json');assert accept['successful_updates']==1800
   metrics=list(csv.DictReader((folder/'training_metrics.csv').open()));assert [int(r['step']) for r in metrics]==list(range(1,1801))
   assert all(abs(float(r['lr'])-(.0002 if int(r['step'])<=600 else .00002))<1e-12 for r in metrics)
   curves.extend(dict(policy_seed=seed,method=method,**r) for r in metrics)
   valid.extend(dict(policy_seed=seed,method=method,**r) for r in csv.DictReader((folder/'validation_metrics.csv').open()))
   for step in [600,1800]:
    p=folder/f'step{step:04}.pt';inventory.append(dict(policy_seed=seed,method=method,step=step,path=str(p),sha256=sha(p)))
 table(out/'training_metrics.csv',curves)
 groups=collections.defaultdict(list)
 for r in valid:groups[(r['policy_seed'],r['method'],int(r['step']),r['appearance'])].append(r)
 reduced=[]
 for (seed,m,step,c),rs in groups.items():
  reduced.append(dict(policy_seed=seed,method=m,step=step,appearance=c,n_windows=len(rs),**{k:float(np.mean([float(r[k]) for r in rs])) for k in ['action_L1','translation_L1','rotation_L1','gripper_L1','auxiliary_L1']}))
 table(out/'validation_summary.csv',reduced);save(out/'checkpoint_inventory.json',inventory)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,3,figsize=(15,8))
 for i,m in enumerate(METHODS):
  for j,seed in enumerate(SEEDS):
   rs=[r for r in curves if r['method']==m and r['policy_seed']==seed];ax=axes[i,j]
   for key in ['action_loss','weighted_auxiliary']:ax.plot([int(r['step']) for r in rs],[float(r[key]) for r in rs],label=key,linewidth=.6)
   ax.set_title(f'{m} / {seed}');ax.axvline(600,color='gray',linestyle='--');ax.legend()
 fig.tight_layout();fig.savefig(out/'training_curves.png');plt.close(fig)
 bc=method_results['BC-standard'];c=method_results['C-low']
 if c['gate_passed']:decision='C qualifies; propose matched D in a later round only.' if c['gap_candidate'] else 'C qualifies without the prespecified overall gap: stop method search on this task/intervention combination.'
 elif bc['gate_passed']:decision='BC qualifies but C does not: prioritize frozen representation transfer / RLA policy adaptation; do not proceed to consistency.'
 else:decision='Neither method establishes sufficiently stable capability; background robustness is not assessed.'
 lines=['# Drawer capability and appearance-gap development study','',f'Completed 10800 successful policy updates and {len(rows)} valid rollouts. AE/DINO updates: 0.', '', '## Per-seed results','', '| Method | Seed | Appearance | Success |','|---|---:|---|---:|']
 lines += [f"| {r['method']} | {r['policy_seed']} | {r['appearance']} | {r['successes']}/20 |" for r in summaries]
 lines += ['', '## Decision','',decision,'','## Interpretation','', 'C uses frozen scratch-trained AE-C final2000 from three bowl pick/place tasks, not drawer-trained AE. BC/C architecture differs: this is not a pure auxiliary-loss causal estimate. Both use the same 20 training demos, paired appearances and seed-specific balanced indices. 1800 is a fixed limited budget, not demonstrated convergence. State-cluster intervals retain every seed/appearance but do not quantify training-seed, AE or task uncertainty. Reused development states/backgrounds are not independent tests. No D was trained. The earlier stove new-state negative result remains archived unchanged.']
 (out/'report.md').write_text('\n'.join(lines)+'\n')
 save(out/'delivery_verification.json',dict(status='COMPLETE',policy_successful_updates=10800,actual_update_computations=10800,replayed_updates=0,valid_rollouts=len(rows),actual_attempts=len(rows),invalid_trials=0,AE_updates=0,DINO_updates=0,sealed_reads=0,method_results=method_results,history_preserved=all(sha(p)==h for p,h in read(ROOT/'protected_inputs.json').items())))
if __name__=='__main__':main()
