"""Matched checkpoint diagnostics; separate process, no optimizer/RNG pollution."""
import os,json,gc,time,hashlib,random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from stove_centering.training import initialize
from stove_centering.store import CenterStore
def execute(g):
 plan,prev,initial=initialize(g);root=g['a'].run.parent;mods=g['mods'];model,head,prop=mods['vla'],mods['head'],mods['proprio'];lam=plan['shared_protocol']['lambda_aux'];fixed=json.loads((root/'fixed_sample_manifest.json').read_text());meta=json.loads((root/'normalization.json').read_text())
 from coverage_gripper.dataset import DerivedEpisodeDataset
 from rla.data import RlaCollator
 from common import csvsave,save,digest_state
 collate=RlaCollator(2048,g['tokenizer'].pad_token_id);ds=DerivedEpisodeDataset(plan['inputs']['data_manifest'],plan['inputs']['action_proprio_statistics'],g['transform'],collate)
 action_stats=json.loads(Path(plan['inputs']['action_proprio_statistics']).read_text())['action'];assert action_stats['mask'][6] is False
 capture={};predict=head.predict_action
 def pred(*args,**kw):
  y=predict(*args,**kw);capture['action']=y;return y
 head.predict_action=pred
 def batch(xs):
  items=[]
  for x in xs:
   e,t,c=x['episode'],x['t'],x['appearance'];it=ds.sample(e,t,c);it['rla']=stores[c].frame(slots[e][c],t);items.append(it)
  b=collate(items);b['labels']=b['labels'].cuda();b['proprio']=b['proprio'].cuda();assert int(model._process_action_masks(b['labels']).sum())==len(xs)*g['NUM_TOKENS'];return b
 def forward(b):
  capture.clear();act,_=g['ft'].run_forward_pass(vla=model,action_head=SimpleNamespace(module=head),proprio_projector=prop,batch=b,action_tokenizer=g['at'],device_id='cuda',use_l1_regression=True,use_proprio=True,use_film=False,num_patches=model.vision_backbone.get_num_patches()*2,use_pro_version=True,cfg=SimpleNamespace(phase='Inference'));z=head.pop_z_hat();capture['z']=z;aux=torch.nn.functional.l1_loss(z,b['rla'].cuda().to(z.dtype));return act,aux
 def zero():
  for m in mods.values():m.zero_grad(set_to_none=True)
 def rng():
  torch.manual_seed(20261023);torch.cuda.manual_seed_all(20261023);np.random.seed(20261023);random.seed(20261023)
 rows=[];energies=[];gradrows=[];closures=[];verification=[]
 shared=[('vla.'+n,p) for n,p in model.named_parameters() if p.requires_grad]
 shared += [('proprio.'+n,p) for n,p in prop.named_parameters() if p.requires_grad]
 shared += [('shared_head.'+n,p) for n,p in head.named_parameters() if p.requires_grad and n.startswith(('model.layer_norm1.','model.fc1.','model.mlp_resnet_blocks.'))]
 started=time.time()
 for arm in ('D-old','D-center'):
  os.environ['CENTER_ARM']=arm;stores={c:CenterStore(str(prev/'sidecars/D'/c),chunk=8,num_tokens=16,token_dim=64,norm='center') for c in ('clean','background')};slots={e:{c:stores[c].slot(np.load(r/'trajectory_labels.npz')['proprio']) for c in stores} for e,(r,_,_) in ds.episodes.items()};mu=np.asarray(meta['mu_C' if arm=='D-old' else 'mu_D'],np.float32).reshape(1024);scale=meta['s_C']
  for step in plan['diagnostic_steps']:
   ck=torch.load(root/arm/'isolated_smoke'/f'step{step:04}.pt',map_location='cpu');assert ck['global_step']==step
   for k,m in mods.items():
    full=m.state_dict();assert set(ck['parameters'][k])=={n for n,p in m.named_parameters() if p.requires_grad};full.update(ck['parameters'][k]);m.load_state_dict(full,strict=True);m.eval()
   before=digest_state(ck);del ck;zero();rng();gc.collect()
   with torch.inference_mode():
    for start in range(0,len(fixed['policy']),4):
     xs=fixed['policy'][start:start+4];b=batch(xs);act,aux=forward(b);predicted=capture['action'].float();target=b['actions'].cuda().float();err=abs(predicted-target);z=capture['z'].float();truth=b['rla'].cuda().float();res=z-truth;assert torch.isfinite(err).all() and torch.isfinite(res).all()
     for i,x in enumerate(xs):
      for h in range(8):
       row=dict(arm=arm,step=step,**x,horizon=h+1,action_L1=float(err[i,h].mean()))
       for d in range(7):row.update({f'dim{d}_L1':float(err[i,h,d]),f'dim{d}_prediction':float(predicted[i,h,d]),f'dim{d}_target':float(target[i,h,d])})
       rows.append(row)
      # Gripper is excluded from Q99 normalization: 0=close,1=open.
      # Official process_action binarizes at .5 then flips to positive=close.
      pc=torch.where(predicted[i,:,6]<.5)[0];tc=torch.where(target[i,:,6]<.5)[0]
      closures.append(dict(arm=arm,step=step,**x,predicted_first_close=int(pc[0]) if len(pc) else None,target_first_close=int(tc[0]) if len(tc) else None,missed_close=bool(len(tc) and not len(pc)),early_close=bool(len(pc) and (not len(tc) or int(pc[0])<int(tc[0]))),delay_steps=int(pc[0]-tc[0]) if len(pc) and len(tc) else None,window_censoring='8 predictions only; no-close outside chunk not inferred'))
     mean=res.mean((0,1),keepdim=True);targetmean=truth.mean((0,1),keepdim=True)
     rawpred=z*scale+torch.as_tensor(mu,device='cuda');rawtarget=truth*scale+torch.as_tensor(mu,device='cuda')
     energies.append(dict(arm=arm,step=step,fixed_batch_start=start,samples=len(xs),split=xs[0]['split'],auxiliary_L1_official_bf16=float(aux),weighted_auxiliary=float((lam*aux).detach()),residual_MSE=float(res.square().mean()),residual_mean_energy=float(mean.square().mean()),residual_demeaned_energy=float((res-mean).square().mean()),target_squared_energy=float(truth.square().mean()),target_mean_energy=float(targetmean.square().mean()),raw_latent_L1=float(abs(rawpred-rawtarget).mean()),raw_latent_MSE=float((rawpred-rawtarget).square().mean()),coordinate='same raw AE-D latent coordinates; mean decomposition over fixed batch and8 horizons per latent dimension'))
     capture.clear();del b,act,aux,predicted,target,err,z,truth,res,rawpred,rawtarget
   for split in ('train','validation'):
    xs=[dict(episode=x['episode'],t=x['t'],appearance=c) for x in fixed['gradient'] if x['split']==split for c in ('clean','background')];assert len(xs)==8;vectors=[];losses=[];raws=[]
    for objective in ('action','weighted_auxiliary'):
     acc={};lv=[];rv=[];zero();rng()
     for start in range(0,8,2):
      b=batch(xs[start:start+2]);act,aux=forward(b);loss=act if objective=='action' else lam*aux;lv.append(float(loss.detach()));rv.append(float(aux.detach()));(loss/4).backward()
      for n,p in shared:
       if p.grad is not None:
        v=p.grad.detach().float().cpu();assert torch.isfinite(v).all()
        if n in acc:acc[n].add_(v)
        else:acc[n]=v.clone()
      zero();capture.clear();del b,act,aux,loss
     vectors.append(acc);losses.append(float(np.mean(lv)));raws.append(float(np.mean(rv)))
    aa,bb=vectors
    for scope,prefix in [('shared_VLM','vla.'),('shared_head_trunk','shared_head.'),('proprio_projector','proprio.'),('all_shared','')]:
     ka={n for n in aa if n.startswith(prefix)};kb={n for n in bb if n.startswith(prefix)}
     na=sum(float(aa[n].double().square().sum()) for n in sorted(ka))**.5;nb=sum(float(bb[n].double().square().sum()) for n in sorted(kb))**.5;dot=sum(float((aa[n].double()*bb[n].double()).sum()) for n in sorted(ka&kb));cos=dot/max(na*nb,1e-30)
     gradrows.append(dict(arm=arm,step=step,split=split,action_loss=losses[0],raw_auxiliary_loss=raws[1],weighted_auxiliary_loss=losses[1],action_gradient_norm=na,weighted_auxiliary_gradient_norm=nb,cosine=cos,angle_degrees=float(np.degrees(np.arccos(np.clip(cos,-1,1)))),scope=scope,mode='eval_with_grad; separate objective forwards, matched RNG, FP32 accumulation'))
    del vectors,aa,bb,acc;gc.collect()
   after=digest_state(dict(parameters=g['trainable_state']()));assert before==after;verification.append(dict(arm=arm,step=step,parameters_unchanged=True,sha256=after));assert g['counts'](g['before'])==g['counts'](g['ecc']());torch.cuda.empty_cache();print('DIAG',arm,step,time.time()-started,flush=True)
 o=g['o'];csvsave(o/'action_by_dimension_horizon.csv',rows);csvsave(o/'auxiliary_residual_energy.csv',energies);csvsave(o/'gradient_contributions.csv',gradrows);csvsave(o/'closure_predictions.csv',closures);save(o/'verification.json',dict(parameter_checks=verification,optimizer_updates=0,backward_calls=160,peak_allocated=torch.cuda.max_memory_allocated(),seconds=time.time()-started))
