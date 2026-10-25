"""Frozen checkpoint evaluations and separate objective gradients, zero updates."""
from pathlib import Path
import csv,hashlib,time
import numpy as np
import torch
from torch.nn import functional as F
from control_preserving_rla.common import read,save,restore_rng,rng_state,load_parameters,digest
from control_preserving_rla.interface import Interface,enable_checkpointing
from rla_lr_budget_sweep.interface import initialize

def execute(g):
 plan,prev,initial=initialize(g);o=g['o'];root=g['a'].run.parent;arm=g['a'].arm;api=Interface(g,plan);enable_checkpointing(g);lam=plan['shared_protocol']['lambda_aux']
 from stove_aux_weight.store import LowWeightStore
 cfg=plan['arms'][arm];stores={c:LowWeightStore(str(prev/'sidecars'/cfg['target']/c),chunk=8,num_tokens=16,token_dim=64,norm='center') for c in ('clean','background')}
 slots={e:{c:stores[c].slot(np.load(p/'trajectory_labels.npz')['proprio']) for c in stores} for e,(p,_,_) in api.ds.episodes.items()}
 registry=read(root/'checkpoint_registry.json')[arm];fixed=read(Path(plan['fixed_eval_manifest']));rows=[];gradrows=[];updates=[];verification=[];backwards=0
 shared=[('vla.'+n,p) for n,p in g['model'].named_parameters() if p.requires_grad]
 shared += [('proprio.'+n,p) for n,p in g['prop'].named_parameters() if p.requires_grad]
 shared += [('shared_head.'+n,p) for n,p in g['head'].named_parameters() if p.requires_grad and n.startswith(('model.layer_norm1.','model.fc1.','model.mlp_resnet_blocks.'))]
 save(o/'gradient_scope.json',dict(scope='same accepted stove diagnostic scope: trainable VLA/VLM including visual/language LoRA and action queries, proprio projector, shared head layer_norm1/fc1/mlp_resnet_blocks; excludes task-specific final readouts',parameters=[dict(name=n,numel=p.numel()) for n,p in shared],parameter_count=sum(p.numel() for n,p in shared),mode='eval with gradients, same8 appearance samples, two separate objective forwards, same RNG',near_zero_norm_threshold=1e-12))
 def group(k,n):
  if k=='vla':return 'backbone_VLM'
  if k=='proprio':return 'proprio_projector'
  if n.startswith(('model.layer_norm2_rla.','model.fc2_rla.')):return 'auxiliary_readout'
  return 'action_head_including_shared_RLA_trunk'
 base=initial['parameters'];previous=base;previous_step=0
 for step,path in sorted((int(s),p) for s,p in registry.items()):
  ck=torch.load(path,map_location='cpu');assert ck['global_step']==step;load_parameters(g,ck);restore_rng(ck)
  for m in g['mods'].values():m.eval();m.zero_grad(set_to_none=True)
  before=digest(g['trainable_state']());buffers={(k,n):b.detach().cpu().clone() for k,m in g['mods'].items() for n,b in m.named_buffers()};original_rng=rng_state()
  ss={}
  for k,ps in ck['parameters'].items():
   for n,p in ps.items():
    category=group(k,n);v=p.float();b=base[k][n].float();last=previous[k][n].float();d=v-b;di=v-last
    stat=ss.setdefault(category,dict(parameters=0,param_sq=0.,initial_sq=0.,delta_sq=0.,interval_sq=0.,max_update=0.));stat['parameters']+=p.numel();stat['param_sq']+=float(v.double().square().sum());stat['initial_sq']+=float(b.double().square().sum());stat['delta_sq']+=float(d.double().square().sum());stat['interval_sq']+=float(di.double().square().sum());stat['max_update']=max(stat['max_update'],float(d.abs().max()))
  for category,s in ss.items():updates.append(dict(model=arm,step=step,previous_available_step=previous_step,group=category,parameters=s['parameters'],parameter_norm=s['param_sq']**.5,initial_parameter_norm=s['initial_sq']**.5,absolute_change_from0=s['delta_sq']**.5,relative_change_from0=(s['delta_sq']/s['initial_sq'])**.5 if s['initial_sq']>1e-24 else None,interval_change=s['interval_sq']**.5,max_absolute_update=s['max_update'],near_zero_initial=s['initial_sq']<=1e-24))
  examples=[(r['episode'],r['timestep'],r['appearance']) for r in fixed['samples']]
  with torch.inference_mode():
   for start in range(0,len(examples),8):
    ex=examples[start:start+8];b=api.batch(ex);a,z,_=api.forward(b,'Inference');target=api.targets(ex,stores,slots);err=(a.float()-b['actions'].cuda().float()).abs();ze=(z.float()-target).abs();assert torch.isfinite(err).all() and torch.isfinite(ze).all()
    for i,(e,t,c) in enumerate(ex):
     for h in range(8):rows.append(dict(model=arm,step=step,split=api.ds.episodes[e][2]['split'],episode=e,timestep=t,appearance=c,horizon=h,action_L1=float(err[i,h].mean()),translation_L1=float(err[i,h,:3].mean()),rotation_L1=float(err[i,h,3:6].mean()),gripper_L1=float(err[i,h,6]),auxiliary_L1=float(ze[i].mean()),weighted_auxiliary=lam*float(ze[i].mean()),**{f'dim{d}_L1':float(err[i,h,d]) for d in range(7)}))
    api.clear();del b,a,z,target,err,ze
  restore_rng(original_rng)
  if step in (0,100,200):
   ex=[(r['episode'],r['timestep'],r['appearance']) for r in fixed['gradient_samples']];vectors=[];missing=[];losses=[]
   assert len(ex)==8 and all(api.ds.episodes[e][2]['split']=='train' for e,t,c in ex)
   for objective in ('action','weighted_auxiliary'):
    restore_rng(original_rng);b=api.batch(ex);a,z,bc=api.forward(b,'Inference');target=api.targets(ex,stores,slots);aux=F.l1_loss(z,target.to(z.dtype));loss=bc if objective=='action' else lam*aux
    assert torch.isfinite(loss);grads=torch.autograd.grad(loss,[p for n,p in shared],allow_unused=True);backwards+=1
    missing.append([n for (n,p),v in zip(shared,grads) if v is None]);vectors.append([torch.zeros(p.numel(),dtype=torch.float32) if v is None else v.detach().float().cpu().flatten() for (n,p),v in zip(shared,grads)]);losses.append(dict(action=float(bc.detach()),raw_aux=float(aux.detach()),weighted_aux=lam*float(aux.detach())))
    api.clear();del b,a,z,bc,target,aux,loss,grads
   aa=sum(float(v.double().square().sum()) for v in vectors[0]);bb=sum(float(v.double().square().sum()) for v in vectors[1]);ab=sum(float((x.double()*y.double()).sum()) for x,y in zip(*vectors));na=aa**.5;nb=bb**.5
   gradrows.append(dict(model=arm,step=step,action_loss=losses[0]['action'],raw_auxiliary=losses[1]['raw_aux'],weighted_auxiliary=losses[1]['weighted_aux'],action_gradient_norm=na,weighted_auxiliary_gradient_norm=nb,norm_ratio=nb/na if na>1e-12 else None,cosine=ab/(na*nb) if min(na,nb)>1e-12 else None,near_zero_denominator=na<=1e-12,action_no_gradient_names=missing[0],aux_no_gradient_names=missing[1]))
   del vectors
  restore_rng(original_rng)
  current_buffers={(k,n):b for k,m in g['mods'].items() for n,b in m.named_buffers()};assert set(current_buffers)==set(buffers)
  assert all(torch.equal(v.cpu(),buffers[key]) for key,v in current_buffers.items()),'diagnostic changed registered buffers'
  assert before==digest(g['trainable_state']());assert all(p.grad is None for m in g['mods'].values() for p in m.parameters())
  verification.append(dict(step=step,parameter_hash=before,unchanged=True,buffers_unchanged=True,optimizer_not_instantiated=True,RNG_restored=True));previous=ck['parameters'];previous_step=step;del ck,buffers
  print('OFFLINE_NODE_COMPLETE',arm,step,flush=True)
 def write(name,rs):
  with (o/name).open('x') as f:w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rs for k in r)));w.writeheader();w.writerows(rs)
 write('fixed_sample_metrics.csv',rows);write('gradient_contributions.csv',gradrows);write('parameter_update_statistics.csv',updates)
 save(o/'diagnostic_verification.json',dict(status='PASS',nodes=verification,optimizer_updates=0,backward_calls=backwards,AE_updates=0,teacher_updates=0,new_rollouts=0,peak_allocated=torch.cuda.max_memory_allocated(),missing_step50=arm=='C_m0p1'))
