"""Complete development result table and physical-state clustered comparisons."""
import argparse,json,csv
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();rows=json.loads((a.run/'per_trial_results.json').read_text());assert all(r['status']=='COMPLETE' for r in rows)
 methods=['BC-standard','BC-matched','C-frozen-RLA'];orders=[20261101,20261102];states=sorted({r['seed'] for r in rows});conditions=['clean']+sorted({r['condition'] for r in rows}-{'clean'});assert len(states)==20
 values={(r['model'],r['seed'],r['condition']):int(r['success']) for r in rows};summary=[];rng=np.random.RandomState(20261101);draws=rng.randint(0,20,(10000,20));comparisons=[]
 for order in orders:
  for method in methods:
   key=f'{order}_{method}'
   for c in conditions:
    v=np.array([values[key,s,c] for s in states]);ci=np.quantile(v[draws].mean(1),[.025,.975]);summary.append(dict(order=order,method=method,condition=c,successes=int(v.sum()),trials=20,success_rate=float(v.mean()),CI_low=float(ci[0]),CI_high=float(ci[1])))
  for base in ['BC-standard','BC-matched']:
   for c in conditions:
    x=np.array([values[f'{order}_C-frozen-RLA',s,c] for s in states]);y=np.array([values[f'{order}_{base}',s,c] for s in states]);delta=x-y;ci=np.quantile(delta[draws].mean(1),[.025,.975]);comparisons.append(dict(order=order,comparison='C-frozen-RLA minus '+base,condition=c,difference=float(delta.mean()),C_win=int((delta==1).sum()),C_loss=int((delta==-1).sum()),CI=ci.tolist()))
 paired=[]
 for order in orders:
  for m in methods:
   for c in conditions[1:]:
    x=np.array([values[f'{order}_{m}',s,c] for s in states]);y=np.array([values[f'{order}_{m}',s,'clean'] for s in states]);paired.append(dict(order=order,method=m,condition=c,clean_success_background_failure=int(((y==1)&(x==0)).sum()),reverse=int(((y==0)&(x==1)).sum()),difference=float((x-y).mean())))
 for filename,data in [('success_rate_summary.csv',summary),('appearance_flips.csv',paired)]:
  if data:
   with (a.run/filename).open('x') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 with (a.run/'paired_statistics.json').open('x') as f:json.dump(dict(comparisons=comparisons,bootstrap='10000 physical-state resamples; common indices preserve appearances/orders',limitation='does not include task or full training-seed uncertainty'),f,indent=2)
 gate=json.loads((a.run/'clean_gate.json').read_text());lines=['# New middle-drawer baseline screening','',f'Policy successful updates: 3600. AE/DINO updates: 0. Valid Clean trials: 120. Background trials: {len(rows)-120}.',f'Clean capability gate passed: {gate["passed"]}.','', '|Order|Method|Appearance|Success|','|---|---|---|---|']
 lines += [f'|{r["order"]}|{r["method"]}|{r["condition"]}|{r["successes"]}/20|' for r in summary]
 lines += ['', 'Development exploration, not independent test evidence. Two orders share model initialization, and are not full training seeds. Official AE task-pretraining exclusion is not established. C-frozen-RLA uses the original frozen step90000 AE, not task-finetuned AE-C.','', 'BC-matched versus C-frozen-RLA controls architecture. BC-standard differs in action-head architecture. No selected-best checkpoint, extra training, or hyperparameter search.', '', 'If Clean gate failed, appearance robustness is unanswered; stop background scoring. If it passed, inspect all frozen backgrounds and both orders, not the worst background alone.']
 with (a.run/'report.md').open('x') as f:f.write('\n'.join(lines)+'\n')
 with (a.run/'delivery_verification.json').open('x') as f:json.dump(dict(status='COMPLETED_EXPLORATORY',policy_updates=3600,AE_updates=0,DINO_updates=0,valid_trials=len(rows),invalid_trials=0,clean_gate=gate,heldout_read=False),f,indent=2)

if __name__=='__main__':main()
