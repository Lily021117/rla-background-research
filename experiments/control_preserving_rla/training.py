"""C-initialized official action/RLA policy post-training, optional keep L1."""
import csv
import os
from pathlib import Path
import resource
import time
import numpy as np
import torch
from torch.nn import functional as F
from control_preserving_rla.common import read,save,rng_state,restore_rng,load_parameters,digest
from control_preserving_rla.interface import initialize,Interface,enable_checkpointing,masked_l1

def execute(g):
 out=g['a'].run.parent;o=g['o'];arm=g['a'].arm
 assert read(out/'gate_acceptance.json')['status']=='PASS'
 plan,prev,initial=initialize(g);cfg=plan['arms'][arm];lam=plan['shared_protocol']['lambda_aux'];beta=cfg['beta']
 api=Interface(g,plan);enable_checkpointing(g)
 from stove_aux_weight.store import LowWeightStore
 stores={c:LowWeightStore(str(prev/'sidecars'/cfg['target']/c),chunk=8,num_tokens=16,token_dim=64,norm='center') for c in ('clean','background')}
 slots={e:{c:stores[c].slot(np.load(root/'trajectory_labels.npz')['proprio']) for c in stores} for e,(root,_,_) in api.ds.episodes.items()}
 order=read(Path(plan['inputs']['B20_sampling_order']));assert len(order)==4800
 for e,t in order:assert api.ds.episodes[e][2]['split']=='train' and 0<=t<api.ds.episodes[e][2]['valid_chunks']
 teacher=np.load(plan['train_only_teacher_cache']);assert set(teacher.files)=={e for e,(_,_,m) in api.ds.episodes.items() if m['split']=='train'}
 opt=torch.optim.AdamW(g['params'],lr=2e-4,weight_decay=.01);scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[600],gamma=.1)
 restore_rng(initial)
 save(o/'initial_state_audit.json',dict(parameter_hash=digest(g['trainable_state']()),common_parameter_hash=digest(initial['parameters']),source_step0=plan['common_student_step0'],parent_C_updates=600,optimizer_initially_empty=not opt.state,seed=plan['seed'],beta=beta,lambda_aux=lam,target=cfg['target'],teacher_graph=False))
 save(o/'execution_configuration.json',dict(arm=arm,beta=beta,lambda_aux=lam,target=cfg['target'],target_coordinates='(z_C-mu_C)/s_C' if cfg['target']=='C' else '(z_D-mu_D)/s_C',batch=16,microbatch=8,accumulation=2,checkpointing=True,dtype='bfloat16',keep_reduction_dtype='float32',math=torch.backends.cuda.math_sdp_enabled(),flash=torch.backends.cuda.flash_sdp_enabled(),efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),optimizer='AdamW',lr=2e-4,weight_decay=.01,updates=600,parent_C_updates=600,teacher_updates=0,AE_updates=0))
 def snapshot(step):
  ck=dict(parameters=g['trainable_state'](),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),global_step=step,parent_C_updates=600,next_physical_sample=step*8,configuration=plan,arm=arm,AMP_scaler=None,dtype='bfloat16',activation_checkpointing=True,**rng_state())
  with (o/f'step{step:04}.pt').open('xb') as f:torch.save(ck,f)
 dev=[(e,t,c) for e,(_,_,m) in api.ds.episodes.items() if m['split']=='validation' for t in range(m['valid_chunks']) for c in ('clean','background')]
 vf=(o/'validation_metrics.csv').open('x');vw=csv.DictWriter(vf,fieldnames=['step','episode','timestep','appearance','action_L1','translation_L1','rotation_L1','gripper_L1','auxiliary_L1']);vw.writeheader()
 def validate(step):
  rng=rng_state()
  for m in g['mods'].values():m.eval()
  with torch.inference_mode():
   for start in range(0,len(dev),8):
    ex=dev[start:start+8];b=api.batch(ex);a,z,_=api.forward(b,'Inference');target=api.targets(ex,stores,slots)
    err=(a.float()-b['actions'].cuda().float()).abs();ze=(z.float()-target).abs();assert torch.isfinite(err).all() and torch.isfinite(ze).all()
    for i,(e,t,c) in enumerate(ex):vw.writerow(dict(step=step,episode=e,timestep=t,appearance=c,action_L1=float(err[i].mean()),translation_L1=float(err[i,:,:3].mean()),rotation_L1=float(err[i,:,3:6].mean()),gripper_L1=float(err[i,:,6].mean()),auxiliary_L1=float(ze[i].mean())))
    api.clear();del b,a,z,target,err,ze
  vf.flush();os.fsync(vf.fileno())
  for m in g['mods'].values():m.train()
  restore_rng(rng)
 start_step=0
 if os.environ.get('POLICY_RESUME'):
  ck=torch.load(os.environ['POLICY_RESUME'],map_location='cpu');assert ck['arm']==arm and ck['configuration']==plan
  start_step=ck['global_step'];assert ck['next_physical_sample']==start_step*8 and 0<start_step<600
  load_parameters(g,ck);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);restore_rng(ck)
  save(o/'resume_provenance.json',dict(source=os.environ['POLICY_RESUME'],restored_step=start_step,restored=['model','optimizer','scheduler','RNG','sampling_position','successful_update_count'],AMP_scaler=None));del ck
 snapshot(start_step);validate(start_step);del initial
 fields=['step','action_loss','auxiliary_loss','weighted_auxiliary','keep_loss','weighted_keep','total_loss','lr','gradient_vla','gradient_head','gradient_proprio','gradient_aux_head','allocated','reserved','cpu_maxrss_kib','seconds','successful_updates']
 f=(o/'training_metrics.csv').open('x');w=csv.DictWriter(f,fieldnames=fields);w.writeheader();started=time.time()
 for step in range(start_step+1,601):
  opt.zero_grad(set_to_none=True);examples=[(e,t,c) for e,t in order[(step-1)*8:step*8] for c in ('clean','background')];values=[]
  for start in range(0,16,8):
   ex=examples[start:start+8];b=api.batch(ex);a,z,bc=api.forward(b)
   target=api.targets(ex,stores,slots);aux=F.l1_loss(z,target.to(z.dtype))
   teacher_actions=torch.from_numpy(np.stack([teacher[e][t] for e,t,c in ex])).cuda();mask=torch.ones_like(a,dtype=torch.float32)
   keep=masked_l1(a,teacher_actions,mask);total=(bc+lam*aux).float()+beta*keep
   assert torch.isfinite(total);values.append([float(x.detach()) for x in (bc,aux,keep,total)])
   (total/2).backward();api.clear();del b,a,z,bc,target,aux,teacher_actions,mask,keep,total
  norms={}
  for k,m in g['mods'].items():
   grads=[p.grad.detach().float() for p in m.parameters() if p.grad is not None];assert all(torch.isfinite(x).all() for x in grads)
   norms[k]=float(torch.stack([x.norm().square() for x in grads]).sum().sqrt())
  auxnorm=sum(float(p.grad.detach().float().square().sum()) for n,p in g['head'].named_parameters() if p.grad is not None and ('z_' in n or 'rla' in n))**.5
  lr=opt.param_groups[0]['lr'];opt.step();scheduler.step();assert all(torch.isfinite(p).all() for p in g['params'])
  torch.cuda.empty_cache();bc,aux,keep,total=np.mean(values,axis=0)
  w.writerow(dict(step=step,action_loss=bc,auxiliary_loss=aux,weighted_auxiliary=lam*aux,keep_loss=keep,weighted_keep=beta*keep,total_loss=total,lr=lr,gradient_vla=norms['vla'],gradient_head=norms['head'],gradient_proprio=norms['proprio'],gradient_aux_head=auxnorm,allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved(),cpu_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,seconds=time.time()-started,successful_updates=step));f.flush()
  if step%25==0:print('POLICY',arm,step,'BC',bc,'aux',aux,'keep',keep,'seconds',time.time()-started,flush=True);assert g['counts'](g['before'])==g['counts'](g['ecc']())
  if step in (1,100,300,600):snapshot(step)
  if step in (100,300,600):validate(step)
 f.close();vf.close();ck=torch.load(o/'step0600.pt',map_location='cpu');assert ck['global_step']==600 and ck['next_physical_sample']==4800
 assert digest(ck['parameters'])==digest(g['trainable_state']());assert ck['scheduler']['last_epoch']==600
 load_parameters(g,ck)
 save(o/'training_acceptance.json',dict(status='PASS_TRAINING_NOT_METHOD',successful_updates=600,parent_C_updates=600,actual_computed_updates=600-start_step,strict_reload=True,seconds=time.time()-started,peak_allocated=torch.cuda.max_memory_allocated(),physical_exposures=4800,appearance_exposures=9600,ECC_increment=False,teacher_updates=0,AE_updates=0))
