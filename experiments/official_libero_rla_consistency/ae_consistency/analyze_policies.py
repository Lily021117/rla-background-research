"""State-clustered paired analysis; never selects checkpoints or states."""
import argparse,json,csv,math,collections,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();root=a.run
rows=json.loads((root/'per_trial_results.json').read_text());assert len(rows)==180 and all(r['status']=='COMPLETE' for r in rows)
conditions=['clean','seen_background','unseen_warm_teal'];models=['B20','C','D'];states=list(range(2026102300,2026102320));x=np.zeros((20,3,3),dtype=float);found=set()
for r in rows:
 k=(r['seed'],r['condition'],r['model']);assert k not in found;found.add(k);x[states.index(r['seed']),conditions.index(r['condition']),models.index(r['model'])]=float(r['success'])
rng=np.random.RandomState(20261023);idx=rng.randint(0,20,(20000,20))
def ci(v):return np.quantile(v[idx].mean(axis=1),[.025,.975]).tolist()
def wilson(k,n):
 z=1.959963984540054;ph=k/n;mid=(ph+z*z/(2*n))/(1+z*z/n);half=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/(1+z*z/n);return [mid-half,mid+half]
summary=[]
for j,c in enumerate(conditions):
 for m,model in enumerate(models):
  v=x[:,j,m];k=int(v.sum());summary.append(dict(model=model,condition=c,successes=k,trials=20,success_rate=k/20,Wilson95=wilson(k,20),state_cluster_bootstrap95=ci(v)))
comparisons=[]
for hi,lo in [('C','B20'),('D','C'),('D','B20')]:
 h,l=models.index(hi),models.index(lo)
 for j,c in enumerate(conditions):
  v=x[:,j,h]-x[:,j,l];gain=int((v==1).sum());loss=int((v==-1).sum());comparisons.append(dict(comparison=f'{hi}-{lo}',condition=c,difference=float(v.mean()),state_cluster_bootstrap95=ci(v),success_gain=gain,success_loss=loss,paired_unchanged=20-gain-loss,exact_discordant_binomial_p=binomtest(gain,gain+loss,.5).pvalue if gain+loss else 1.))
 comparisons.append(dict(comparison=f'{hi}-{lo}',condition='mean_across_three_appearances',difference=float((x[:,:,h]-x[:,:,l]).mean()),state_cluster_bootstrap95=ci((x[:,:,h]-x[:,:,l]).mean(axis=1)),unit='20 physical-state clusters, not60 independent appearance trials'))
effects=[]
for m,model in enumerate(models):
 for j,c in enumerate(conditions[1:],1):
  v=x[:,j,m]-x[:,0,m];effects.append(dict(model=model,background=c,background_minus_clean=float(v.mean()),state_cluster_bootstrap95=ci(v)))
interactions=[]
for hi,lo in [('D','C'),('C','B20'),('D','B20')]:
 h,l=models.index(hi),models.index(lo)
 for j,c in enumerate(conditions[1:],1):
  v=(x[:,j,h]-x[:,0,h])-(x[:,j,l]-x[:,0,l]);interactions.append(dict(comparison=f'{hi}-{lo}',background=c,interaction=float(v.mean()),state_cluster_bootstrap95=ci(v),definition='(higher_background-higher_clean)-(lower_background-lower_clean)',positive_interpretation='smaller background drop only when interpreted with absolute Clean/Background SR'))
save=lambda name,obj:(root/name).write_text(json.dumps(obj,indent=2))
save('success_rate_summary.json',dict(status='EXPLORATORY_DEVELOPMENT',physical_states=20,valid_trials=180,invalid_trials=0,summary=summary,paired_comparisons=comparisons,background_effects=effects,limits=['single training seed','one task','20 development states, not final independent test','possible ceiling/floor effects','same initial state across appearances is a cluster, not independent replication']))
save('interaction_statistics.json',dict(bootstrap_samples=20000,seed=20261023,resampling_unit='physical state across all3 models/all3 conditions',interactions=interactions,ceiling_caution='bootstrap can degenerate atall-success; Wilson intervals also provided for absolute SR'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
registry=json.loads((root/'policy_registry.json').read_text());configs={m:Path(v).parent/('execution_configuration.json' if m!='B20' else 'configuration.json') for m,v in registry.items()};config_hashes={m:sha(p) for m,p in configs.items()}
with (root/'evaluation_manifest.csv').open('x') as f:
 fields=['trial','model','seed','condition','task_id','language','checkpoint_sha256','training_configuration_sha256','evaluation_protocol_sha256','state_content_sha256','XML_sha256','video','trajectory'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for i,r in enumerate(rows):w.writerow(dict(trial=i+1,model=r['model'],seed=r['seed'],condition=r['condition'],task_id='pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate',language=r['language'],checkpoint_sha256=r['checkpoint_sha256'],training_configuration_sha256=config_hashes[r['model']],evaluation_protocol_sha256=sha(root/'evaluation_protocol.json'),state_content_sha256=r['state_sha256'],XML_sha256=r['xml_sha256'],video=r['video'],trajectory=r['trajectory']))
with (root/'paired_state_results.csv').open('x') as f:
 keys=['seed']+[f'{m}_{c}' for c in conditions for m in models];w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
 for i,s in enumerate(states):w.writerow(dict(seed=s,**{f'{m}_{c}':int(x[i,j,k]) for j,c in enumerate(conditions) for k,m in enumerate(models)}))
with (root/'failure_stage_summary.csv').open('x') as f:
 w=csv.writer(f);w.writerow(['model','condition','stage_proxy','count'])
 for key,count in sorted(collections.Counter((r['model'],r['condition'],r['stage']) for r in rows).items()):w.writerow([*key,count])
val=[]
for model,source in [(m,Path(registry[m]).parent/'validation_metrics.csv') for m in models]:
 grouped=collections.defaultdict(list)
 with source.open() as f:
  for r in csv.DictReader(f):
   if model=='B20' and r['set']!='development':continue
   grouped[(int(r['step']),r['appearance'],r['episode'])].append(r)
 for (step,c,e),rs in grouped.items():
  vals={k:float(np.mean([float(r[k]) for r in rs])) for k in ('action_L1','translation_L1','rotation_L1','gripper_L1')};val.append(dict(model=model,step=step,appearance=c,episode=e,**vals,auxiliary_L1=float(np.mean([float(r['auxiliary_L1']) for r in rs])) if model!='B20' else None))
with (root/'validation_summary.csv').open('x') as f:w=csv.DictWriter(f,fieldnames=list(val[0]));w.writeheader();w.writerows(val)
print(json.dumps(summary,indent=2))
