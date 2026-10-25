"""Original-step0 official C/D-center training, fixed LR and1200-update schedule."""
import csv
import os
from pathlib import Path
import resource
import time
import numpy as np
import torch
from torch.nn import functional as F
from control_preserving_rla.common import read,save,rng_state,restore_rng,load_parameters,digest
from rla_lr_budget_sweep.interface import initialize
from control_preserving_rla.interface import Interface,enable_checkpointing

def execute(g):
 out=g['a'].run.parent;o=g['o'];arm=g['a'].arm
 assert read(out/'short_protocol.json')['rollouts']==0
 plan,prev,initial=initialize(g);cfg=plan['arms'][arm];lam=plan['shared_protocol']['lambda_aux'];beta=0.
 api=Interface(g,plan);enable_checkpointing(g)
 from stove_aux_weight.store import LowWeightStore
 stores={c:LowWeightStore(str(prev/'sidecars'/cfg['target']/c),chunk=8,num_tokens=16,token_dim=64,norm='center') for c in ('clean','background')}
 slots={e:{c:stores[c].slot(np.load(root/'trajectory_labels.npz')['proprio']) for c in stores} for e,(root,_,_) in api.ds.episodes.items()}
 order=read(Path(plan['inputs']['B20_sampling_order']));assert len(order)==9600
 for e,t in order:assert api.ds.episodes[e][2]['split']=='train' and 0<=t<api.ds.episodes[e][2]['valid_chunks']
 opt=torch.optim.AdamW(g['params'],lr=plan['shared_protocol']['lr']*cfg['multiplier'],weight_decay=.01);scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[1200],gamma=.1)
 restore_rng(initial)
 save(o/'initial_state_audit.json',dict(parameter_hash=digest(g['trainable_state']()),common_parameter_hash=digest(initial['parameters']),source_step0=plan['common_step0'],parent_C_updates=0,optimizer_initially_empty=not opt.state,seed=plan['seed'],beta=beta,lambda_aux=lam,target=cfg['target'],teacher_graph=False))
 save(o/'execution_configuration.json',dict(arm=arm,beta=beta,lambda_aux=lam,target=cfg['target'],target_coordinates='(z_C-mu_C)/s_C' if cfg['target']=='C' else '(z_D-mu_D)/s_C',batch=16,microbatch=8,accumulation=2,checkpointing=True,dtype='bfloat16',keep_reduction_dtype='float32',math=torch.backends.cuda.math_sdp_enabled(),flash=torch.backends.cuda.flash_sdp_enabled(),efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),optimizer='AdamW',lr=plan['shared_protocol']['lr']*cfg['multiplier'],weight_decay=.01,updates=200,parent_C_updates=0,teacher_updates=0,AE_updates=0))
 save(o/'optimizer_parameter_groups.json',[dict(index=i,name='all_trainable',trainable_parameters=sum(p.numel() for p in group['params']),lr=group['lr'],weight_decay=group['weight_decay'],betas=group['betas'],eps=group['eps'],scheduler='MultiStepLR milestone1200 gamma0.1 no warmup') for i,group in enumerate(opt.param_groups)])
 assert digest(g['trainable_state']())==digest(initial['parameters'])
 def snapshot(step):
  ck=dict(parameters=g['trainable_state'](),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),global_step=step,parent_C_updates=0,next_physical_sample=step*8,configuration=plan,arm=arm,AMP_scaler=None,dtype='bfloat16',activation_checkpointing=True,**rng_state())
  with (o/f'step{step:04}.pt').open('xb') as f:torch.save(ck,f)
 fixed=read(Path(plan['fixed_eval_manifest']))['samples'];dev=[(r['episode'],r['timestep'],r['appearance']) for r in fixed]
 extra_fields=[f'dim{d}_L1' for d in range(7)]+[f'h{h}_L1' for h in range(8)]
 vf=(o/'validation_metrics.csv').open('x');vw=csv.DictWriter(vf,fieldnames=['step','episode','timestep','appearance','split','action_L1','translation_L1','rotation_L1','gripper_L1','auxiliary_L1']+extra_fields);vw.writeheader()
 def validate(step):
  rng=rng_state()
  for m in g['mods'].values():m.eval()
  with torch.inference_mode():
   for start in range(0,len(dev),8):
    ex=dev[start:start+8];b=api.batch(ex);a,z,_=api.forward(b,'Inference');target=api.targets(ex,stores,slots)
    err=(a.float()-b['actions'].cuda().float()).abs();ze=(z.float()-target).abs();assert torch.isfinite(err).all() and torch.isfinite(ze).all()
    for i,(e,t,c) in enumerate(ex):vw.writerow(dict(step=step,episode=e,timestep=t,appearance=c,split=api.ds.episodes[e][2]['split'],action_L1=float(err[i].mean()),translation_L1=float(err[i,:,:3].mean()),rotation_L1=float(err[i,:,3:6].mean()),gripper_L1=float(err[i,:,6].mean()),auxiliary_L1=float(ze[i].mean()),**{f'dim{d}_L1':float(err[i,:,d].mean()) for d in range(7)},**{f'h{h}_L1':float(err[i,h].mean()) for h in range(8)}))
    api.clear();del b,a,z,target,err,ze
  vf.flush();os.fsync(vf.fileno())
  for m in g['mods'].values():m.train()
  restore_rng(rng)
 start_step=0
 if os.environ.get('POLICY_RESUME'):
  ck=torch.load(os.environ['POLICY_RESUME'],map_location='cpu');assert ck['arm']==arm and ck['configuration']==plan
  start_step=ck['global_step'];assert ck['next_physical_sample']==start_step*8 and 0<start_step<200
  load_parameters(g,ck);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);restore_rng(ck)
  save(o/'resume_provenance.json',dict(source=os.environ['POLICY_RESUME'],restored_step=start_step,restored=['model','optimizer','scheduler','RNG','sampling_position','successful_update_count'],AMP_scaler=None));del ck
 snapshot(start_step);validate(start_step);del initial
 fields=['step','action_loss','auxiliary_loss','weighted_auxiliary','keep_loss','weighted_keep','total_loss','lr','gradient_vla','gradient_head','gradient_proprio','gradient_aux_head','allocated','reserved','cpu_maxrss_kib','seconds','successful_updates','preclip_total_gradient_norm','clipping_applied']+extra_fields
 f=(o/'training_metrics.csv').open('x');w=csv.DictWriter(f,fieldnames=fields);w.writeheader();started=time.time()
 for step in range(start_step+1,201):
  opt.zero_grad(set_to_none=True);examples=[(e,t,c) for e,t in order[(step-1)*8:step*8] for c in ('clean','background')];values=[];dim_errors=[];h_errors=[]
  for start in range(0,16,8):
   ex=examples[start:start+8];b=api.batch(ex);a,z,bc=api.forward(b)
   target=api.targets(ex,stores,slots);aux=F.l1_loss(z,target.to(z.dtype))
   keep=torch.zeros((),device='cuda');total=(bc+lam*aux).float()
   err=(a.detach().float()-b['actions'].cuda().float()).abs();dim_errors.append(err.mean((0,1)).cpu().tolist());h_errors.append(err.mean((0,2)).cpu().tolist());del err
   assert torch.isfinite(total);values.append([float(x.detach()) for x in (bc,aux,keep,total)])
   (total/2).backward();api.clear();del b,a,z,bc,target,aux,keep,total
  norms={}
  for k,m in g['mods'].items():
   grads=[p.grad.detach().float() for p in m.parameters() if p.grad is not None]
   # Same finite predicate; one host synchronization per module, not per tensor.
   assert bool(torch.stack([torch.isfinite(x).all() for x in grads]).all())
   norms[k]=float(torch.stack([x.norm().square() for x in grads]).sum().sqrt())
  auxnorm=sum(float(p.grad.detach().float().square().sum()) for n,p in g['head'].named_parameters() if p.grad is not None and ('z_' in n or 'rla' in n))**.5
  lr=opt.param_groups[0]['lr'];opt.step();scheduler.step()
  assert bool(torch.stack([torch.isfinite(p).all() for p in g['params']]).all())
  torch.cuda.empty_cache();bc,aux,keep,total=np.mean(values,axis=0)
  w.writerow(dict(step=step,action_loss=bc,auxiliary_loss=aux,weighted_auxiliary=lam*aux,keep_loss=keep,weighted_keep=beta*keep,total_loss=total,lr=lr,gradient_vla=norms['vla'],gradient_head=norms['head'],gradient_proprio=norms['proprio'],gradient_aux_head=auxnorm,allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved(),cpu_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,seconds=time.time()-started,successful_updates=step,preclip_total_gradient_norm=sum(v*v for v in norms.values())**.5,clipping_applied=False,**{f'dim{d}_L1':float(np.mean(dim_errors,axis=0)[d]) for d in range(7)},**{f'h{h}_L1':float(np.mean(h_errors,axis=0)[h]) for h in range(8)}));f.flush()
  if step%25==0:print('POLICY',arm,step,'BC',bc,'aux',aux,'keep',keep,'seconds',time.time()-started,flush=True);assert g['counts'](g['before'])==g['counts'](g['ecc']())
  if step in (1,50,100,200):snapshot(step)
  if step in (50,100,200):validate(step)
 f.close();vf.close();ck=torch.load(o/'step0200.pt',map_location='cpu');assert ck['global_step']==200 and ck['next_physical_sample']==1600
 assert digest(ck['parameters'])==digest(g['trainable_state']());assert ck['scheduler']['last_epoch']==200
 load_parameters(g,ck)
 save(o/'training_acceptance.json',dict(status='PASS_TRAINING_NOT_METHOD',successful_updates=200,parent_C_updates=0,actual_computed_updates=200-start_step,strict_reload=True,seconds=time.time()-started,peak_allocated=torch.cuda.max_memory_allocated(),physical_exposures=1600,appearance_exposures=3200,ECC_increment=False,teacher_updates=0,AE_updates=0))
