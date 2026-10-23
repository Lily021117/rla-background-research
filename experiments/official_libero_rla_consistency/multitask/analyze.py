"""Three fixed tasks, paired state clusters; no model or state selection."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--historical',type=Path,required=True);a=p.parse_args()
conditions=['clean','seen_background','unseen_background'];models=['B','C','D'];rng=np.random.RandomState(20261023);indices=rng.randint(0,20,(20000,20));arrays={};summary=[];pairs=[];interactions=[];failures=[]
def interval(v):return np.quantile(v[indices].mean(1),[.025,.975]).tolist()
def save(name,obj):
 with (a.run/name).open('x') as f:json.dump(obj,f,indent=2)
def csvsave(name,rows):
 with (a.run/name).open('x') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for task,root in [('cookie_historical',a.historical),('drawer',a.run/'drawer/policy_run'),('stove',a.run/'stove/policy_run')]:
 rows=json.loads((root/'per_trial_results.json').read_text());assert len(rows)==180 and all(r['status']=='COMPLETE' for r in rows);states=sorted({r['seed'] for r in rows});assert len(states)==20;x=np.full((20,3,3),np.nan);seen=set()
 for r in rows:
  c='unseen_background' if r['condition'].startswith('unseen') else r['condition'];m='B' if r['model']=='B20' else r['model'];key=(r['seed'],c,m);assert key not in seen;seen.add(key);x[states.index(r['seed']),conditions.index(c),models.index(m)]=int(r['success'])
  failures.append(dict(task=task,seed=r['seed'],appearance=c,model=m,success=r['success'],stage_proxy=r.get('stage','unrecorded'),steps=r['steps'],evidence='existing structured target/EE trajectory proxy; not verified contact truth'))
 assert np.isfinite(x).all();arrays[task]=x
 for j,c in enumerate(conditions):
  for k,m in enumerate(models):
   n=20;s=int(x[:,j,k].sum());z=1.959963984540054;v=s/n;mid=(v+z*z/(2*n))/(1+z*z/n);half=z*math.sqrt(v*(1-v)/n+z*z/(4*n*n))/(1+z*z/n)
   summary.append(dict(task=task,appearance=c,model=m,successes=s,trials=n,SR=v,Wilson95=[mid-half,mid+half]))
 for hi,lo in [('D','C'),('C','B'),('D','B')]:
  h,l=models.index(hi),models.index(lo)
  for j,c in enumerate(conditions):
   v=x[:,j,h]-x[:,j,l];gain=int((v==1).sum());loss=int((v==-1).sum());pairs.append(dict(task=task,comparison=hi+'-'+lo,appearance=c,difference=float(v.mean()),CI95=interval(v),gain=gain,loss=loss,exact_discordant_p=binomtest(gain,gain+loss,.5).pvalue if gain+loss else 1.))
  v=(x[:,:,h]-x[:,:,l]).mean(1);pairs.append(dict(task=task,comparison=hi+'-'+lo,appearance='three_appearance_mean',difference=float(v.mean()),CI95=interval(v),gain=None,loss=None,exact_discordant_p=None))
  for j,c in enumerate(conditions[1:],1):
   v=(x[:,j,h]-x[:,0,h])-(x[:,j,l]-x[:,0,l]);interactions.append(dict(task=task,comparison=hi+'-'+lo,background=c,value=float(v.mean()),CI95=interval(v)))
equal=[]
for j,c in enumerate(conditions):
 vals=[x[:,j,2]-x[:,j,1] for x in arrays.values()];boot=np.mean([v[indices].mean(1) for v in vals],axis=0)
 equal.append(dict(appearance=c,D_minus_C=float(np.mean([v.mean() for v in vals])),fixed_three_task_state_cluster_CI95=np.quantile(boot,[.025,.975]).tolist(),per_task=[float(v.mean()) for v in vals]))
save('success_rate_summary.json',summary);save('paired_statistics.json',pairs);save('interaction_statistics.json',interactions);save('three_task_equal_summary.json',dict(results=equal,limits=['Three fixed tasks only: conditional state-cluster intervals do not establish generalization across tasks','Single matched AE/policy training seed','Discrete success flips and empirical zero lower bounds do not establish stable positive gains','Historical cookie task has one unseen palette, new tasks have frozen three-palette mix','New reset development states are not independent final test; no sealed contents read']))
csvsave('task_stratified_metrics.csv',summary);csvsave('paired_comparisons.csv',pairs);csvsave('failure_stage_analysis.csv',failures)
print(json.dumps(equal,indent=2))
