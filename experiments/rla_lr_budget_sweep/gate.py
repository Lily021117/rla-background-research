"""Only necessary original-init/target and complete-batch memory checks; zero updates."""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from control_preserving_rla.common import read,save,restore_rng,digest
from control_preserving_rla.interface import Interface,enable_checkpointing
from rla_lr_budget_sweep.interface import initialize
def execute(g):
 plan,prev,initial=initialize(g);out=g['a'].run.parent;api=Interface(g,plan);enable_checkpointing(g)
 before=digest(g['trainable_state']());assert before==digest(initial['parameters'])
 opt=torch.optim.AdamW(g['params'],lr=plan['shared_protocol']['lr'],weight_decay=.01)
 names=[dict(module=k,name=n,numel=p.numel()) for k,m in g['mods'].items() for n,p in m.named_parameters() if p.requires_grad]
 save(out/'optimizer_parameter_inventory.json',dict(groups=[dict(name='all_trainable',count=sum(p.numel() for p in opt.param_groups[0]['params']),lr=opt.param_groups[0]['lr'],weight_decay=opt.param_groups[0]['weight_decay'],betas=opt.param_groups[0]['betas'],eps=opt.param_groups[0]['eps'])],parameters=names))
 order=read(Path(plan['inputs']['B20_sampling_order']));ex=[(e,t,c) for e,t in order[:8] for c in ('clean','background')]
 from stove_aux_weight.store import LowWeightStore
 moments=[torch.zeros_like(p) for p in g['params'] for _ in range(2)];outputs=[];rows=[]
 for target in ('C','D'):
  stores={c:LowWeightStore(str(prev/'sidecars'/target/c),chunk=8,num_tokens=16,token_dim=64,norm='center') for c in ('clean','background')}
  slots={e:{c:stores[c].slot(np.load(root/'trajectory_labels.npz')['proprio']) for c in stores} for e,(root,_,_) in api.ds.episodes.items()}
  restore_rng(initial);opt.zero_grad(set_to_none=True)
  for m in g['mods'].values():m.train()
  actions=[];zs=[]
  for start in (0,8):
   batch=api.batch(ex[start:start+8]);a,z,bc=api.forward(batch);y=api.targets(ex[start:start+8],stores,slots);aux=F.l1_loss(z,y.to(z.dtype));loss=bc+plan['shared_protocol']['lambda_aux']*aux
   assert torch.isfinite(loss);(loss/2).backward();actions.append(a.detach().cpu());zs.append(z.detach().cpu());api.clear();del batch,a,z,y,bc,aux,loss
  assert all(p.grad is None or torch.isfinite(p.grad).all() for p in g['params']);outputs.append((torch.cat(actions),torch.cat(zs)));rows.append(dict(target=target,gradient_norm=sum(float(p.grad.float().square().sum()) for p in g['params'] if p.grad is not None)**.5))
 assert torch.equal(outputs[0][0],outputs[1][0]) and torch.equal(outputs[0][1],outputs[1][1]);assert before==digest(g['trainable_state']())
 save(out/'gate_acceptance.json',dict(status='PASS',source=plan['common_step0'],parameter_hash=before,original_step0_strictly_loaded=True,action_and_aux_output_exact_C_D=True,full_batch16_micro8=True,optimizer_updates=0,teacher=False,keep=False,AE_updates=0,checks=rows,peak_allocated=torch.cuda.max_memory_allocated(),parameters_unchanged=True))
 print('GATE_PASS',flush=True)
