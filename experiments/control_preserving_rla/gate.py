"""Necessary step0, teacher isolation and complete-batch zero-update checks."""
import copy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from control_preserving_rla.common import read,save,seed,rng_state,restore_rng,digest
from control_preserving_rla.interface import initialize,Interface,reset_auxiliary,enable_checkpointing,masked_l1

def execute(g):
 plan,prev,teacher=initialize(g,teacher=True);out=g['a'].run.parent
 assert read(out/'teacher_cache_metadata.json')['status']=='PASS'
 api=Interface(g,plan,train_only=True)
 order=read(Path(plan['inputs']['B20_sampling_order']));assert len(order)==4800
 examples=[(e,t,c) for e,t in order[:8] for c in ('clean','background')]
 for e,t,c in examples:assert api.ds.episodes[e][2]['split']=='train'
 for m in g['mods'].values():m.eval()
 b=api.batch(examples[:8])
 with torch.inference_mode():before,_,_=api.forward(b,'Inference');before=before.clone()
 reset_auxiliary(g,plan['auxiliary_reinitialization_seed'])
 state=g['trainable_state']();changed=[]
 allowed={'model.layer_norm2_rla.weight','model.layer_norm2_rla.bias','model.fc2_rla.weight','model.fc2_rla.bias'}
 for k,ps in state.items():
  for n,v in ps.items():
   if not torch.equal(v,teacher['parameters'][k][n]):
    changed.append(k+'.'+n);assert k=='head' and n in allowed,(k,n)
 with torch.inference_mode():after,_,_=api.forward(b,'Inference');assert torch.equal(before,after)
 del teacher,b,before,after;api.clear();seed(plan['seed'])
 initial=dict(parameters=state,global_step=0,parent_C_updates=600,configuration=plan,optimizer={},**rng_state())
 with (out/'common_student_step0.pt').open('xb') as f:torch.save(initial,f)
 save(out/'initialization_acceptance.json',dict(status='PASS',source='original C final600; frozen teacher',teacher_sha256=plan['teacher_sha256'],student_parameter_hash=digest(state),changed_parameters=changed,allowed_reset_parameters=sorted(allowed),all_other_C_parameters_exact=True,action_before_after_auxiliary_reset_exact=True,all_four_groups_use_same_checkpoint_and_RNG=True,teacher_updates=0,optimizer_updates=0))
 enable_checkpointing(g)
 from stove_aux_weight.store import LowWeightStore
 slots={};stores={}
 for target in ('C','D'):
  stores[target]={c:LowWeightStore(str(prev/'sidecars'/target/c),chunk=8,num_tokens=16,token_dim=64,norm='center') for c in ('clean','background')}
  slots[target]={e:{c:stores[target][c].slot(np.load(root/'trajectory_labels.npz')['proprio']) for c in ('clean','background')} for e,(root,_,_) in api.ds.episodes.items()}
 cache=np.load(plan['train_only_teacher_cache']);params=g['params'];moments=[torch.zeros_like(p) for p in params for _ in range(2)]
 outputs=[];checks=[]
 for arm,armcfg in plan['arms'].items():
  restore_rng(initial)
  for m in g['mods'].values():m.train();m.zero_grad(set_to_none=True)
  acts=[];zs=[];losses=[]
  for start in range(0,16,8):
   ex=examples[start:start+8];b=api.batch(ex);a,z,bc=api.forward(b)
   target=api.targets(ex,stores[armcfg['target']],slots[armcfg['target']]);aux=F.l1_loss(z,target.to(z.dtype))
   teach=torch.from_numpy(np.stack([cache[e][t] for e,t,c in ex])).cuda();mask=torch.ones_like(a,dtype=torch.float32)
   keep=masked_l1(a,teach,mask);total=(bc+plan['shared_protocol']['lambda_aux']*aux).float()+armcfg['beta']*keep
   assert torch.isfinite(total);(total/2).backward()
   acts.append(a.detach().cpu());zs.append(z.detach().cpu());losses.append(dict(BC=float(bc.detach()),aux=float(aux.detach()),keep=float(keep.detach()),beta=armcfg['beta']))
   api.clear();del b,a,z,bc,target,aux,teach,keep,total
  assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
  outputs.append((torch.cat(acts),torch.cat(zs)))
  checks.append(dict(arm=arm,losses=losses,shared_gradient_norm=sum(float(p.grad.float().square().sum()) for p in g['model'].parameters() if p.grad is not None)**.5))
 for a,z in outputs[1:]:assert torch.equal(a,outputs[0][0]) and torch.equal(z,outputs[0][1])
 # Isolated keep-only backward: no teacher graph, nonzero shared gradient.
 restore_rng(initial)
 for m in g['mods'].values():m.zero_grad(set_to_none=True)
 b=api.batch(examples[:8]);a,z,_=api.forward(b);teach=torch.from_numpy(np.stack([cache[e][t] for e,t,c in examples[:8]])).cuda()
 keep=masked_l1(a,teach,torch.ones_like(a,dtype=torch.float32));keep.backward()
 kn=sum(float(p.grad.float().square().sum()) for p in g['model'].parameters() if p.grad is not None)**.5
 assert np.isfinite(kn) and kn>0 and not teach.requires_grad
 assert digest(g['trainable_state']())==digest(state)
 save(out/'gate_acceptance.json',dict(status='PASS',optimizer_updates=0,full_effective_batch=16,microbatch=8,accumulation=2,matched_four_arm_action_and_aux_outputs_exact=True,auxiliary_readout_reset_preserves_action=True,teacher_cache_train_only=True,keep_only_shared_gradient=kn,checks=checks,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),parameters_unchanged=True,actual_task_batch_audit=api.audit))
 print('GATE_PASS',flush=True)
