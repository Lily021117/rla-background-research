"""Summarize existing AE training/statistics and physical probes; no new inference."""
import argparse,csv,json,hashlib,collections
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();o=a.out
R=Path('/data/lyc/projects/rla-wm/experiments/official_libero_rla_consistency');S=R/'run_20261023_10/stove/ae_run';sources=[]
def read(p):sources.append(str(p));return json.loads(p.read_text())
def rows(p):sources.append(str(p));return list(csv.DictReader(p.open()))
def save(name,x):
 with (o/name).open('x') as f:json.dump(x,f,indent=2)
def csvsave(name,rs):
 with (o/name).open('x') as f:w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rs for k in r)));w.writeheader();w.writerows(rs)
train=[];rec=[];dist=[];pair=[];means=[];plans={}
for task,root,folders in [('stove',S,{'C':S/'AE_C_recovery1','D':S/'AE_D_recovery1'}),('cookie',R/'run_20261023_08',{'C':R/'run_20261023_08/AE_C','D':R/'run_20261023_08/AE_D_recovery_01'})]:
 plans[task]=read(root/'training_plan.json')
 for arm,path in folders.items():
  rs=rows(path/'training_metrics.csv')
  if int(rs[0]['step'])>1:rs=[r for r in rows(root/('AE_'+arm)/'training_metrics.csv') if int(r['step'])<int(rs[0]['step'])]+rs
  assert [int(r['step']) for r in rs]==list(range(1,601))
  train.extend(dict(task=task,model=arm,**r) for r in rs)
 for r in rows(root/'reconstruction_metrics.csv'):rec.append(dict(task=task,**r))
 for r in rows(root/'latent_distribution_metrics.csv'):dist.append(dict(task=task,**r))
 for arm in ('reference','C','D'):
  for r in rows(root/f'latent_pairs_{arm}.csv'):pair.append(dict(task=task,**r))
 for arm in ('C','D'):
  stats=read(root/f'sidecars/{arm}/clean/raw_target_statistics.json');mu=np.asarray(stats['mean']);std=np.asarray(stats['std']);means.append(dict(task=task,model=arm,training_mean_norm=float(np.linalg.norm(mu)),training_mean_scalar=float(mu.mean()),training_dimension_variance_mean=float((std**2).mean()),count=stats['count'],source='train only pre-FP16 sidecar moments'))
groups=collections.defaultdict(list)
for r in rec:groups[r['task'],r['model'],r['appearance'],int(r['camera']),int(r['h'])].append(r)
summary=[dict(task=k[0],model=k[1],appearance=k[2],camera=k[3],horizon=k[4],n=len(rs),**{f:float(np.mean([float(r[f]) for r in rs])) for f in ('reconstruction_L1','reconstruction_MSE','copy_current_L1','copy_current_MSE')}) for k,rs in groups.items()]
csvsave('AE_training_curves.csv',train);csvsave('AE_development_reconstruction.csv',summary);csvsave('AE_latent_distribution.csv',dist);csvsave('AE_paired_distances.csv',pair);csvsave('AE_training_latent_means.csv',means)
probe=R/'stove_diagnosis/run_20261024_02';csvsave('historical_motion_probe_summary.csv',[r for r in rows(probe/'motion_probe_metrics.csv') if r['episode']=='ALL']);save('historical_motion_probe_uncertainty.json',read(probe/'motion_probe_paired_uncertainty.json'))
save('loss_definition_audit.json',dict(AE=dict(source_code='experiments/official_libero_rla_consistency/ae_consistency/train_ae.py:losses; common.py:forward',lambda_inv=plans['stove']['lambda_cons'],C='0.5*(L1_clean+MSE_clean+L1_background+MSE_background)',D='same reconstruction + lambda_inv*mean((z_clean-z_background)^2)',reconstruction_weights=dict(L1=1,MSE=1,appearance_average=.5),reduction='mean over batch, both views, spatial patches and DINO channels; average8 sampled physical transitions/update; one sampled h1..8 per transition',latent='joint two-view latent16x64; consistency mean across batch/query/dim, not arbitrary per-camera tokens',mask='only complete valid anchors and valid sampled horizons; no padded tail in training',optimizer=plans['stove'],new_updates=0),policy=dict(action_weight=1,lambda_aux=0.22408536585365854,action='official mean normalized L1 over all valid Bx8x7; BF16 action labels in official forward',auxiliary='mean L1(predicted Bx8x1024, normalized target) with common lambda',mask='8x7 all valid action elements;8 cumulative targets; only full-valid windows',batch='8 physical windows x2 appearances; micro8, accumulation2',optimizer='AdamW single trainable group322589983 parameters; base LR2e-4 x multiplier; WD.01, betas.9/.999 eps1e-8',gradient_clipping=None,scheduler='MultiStepLR milestone1200 gamma.1; no warmup; no decay reached this round',teacher=False,keep=False),latent_metrics=dict(relative_L2='norm(zclean-zbg)/(norm(zclean)+1e-8)',unit_L2='norm(zclean/norm(zclean)-zbg/norm(zbg))',effective_rank='SVD of sample-centered latent; exp entropy of normalized squared singular values',participation_ratio='sum(eigenvalues)^2 / sum(eigenvalues^2)',different='only different-episode pairs in same fixed development sample',mean_scope='training mean is saved; full development mean vector not retained'),limitations=['AE development measurements exist only at frozen reference and final600, no invented intermediate curve','No new physical probe; source labels are same-recorded-state simulator bowl position and finger qpos, not external real-world ground truth','Cookie comparison uses same metric definitions but different tasks/distributions; correlation is not causality','Policy LR comparison cannot resolve AE lambda_inv choice, AE convergence or AE seed stability']))
save('AE_source_inventory.json',[dict(path=p,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p in sorted(set(sources))])
print('AE_REVIEW_COMPLETE',flush=True)
