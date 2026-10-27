"""Isolated C-target full effective-batch device acceptance, zero updates."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from stove_centering.training import initialize,restore
from stove_centering.store import CenterStore
from control_preserving_rla.interface import Interface,enable_checkpointing

def execute(g):
 plan,prev,initial=initialize(g);interface=Interface(g,plan);enable_checkpointing(g)
 stores={c:CenterStore(str(prev/'sidecars/C'/c),chunk=8,num_tokens=16,token_dim=64,norm='center') for c in ['clean','background']}
 slots={}
 for e,(root,ch,meta) in interface.ds.episodes.items():
  raw=np.load(root/'trajectory_labels.npz')['proprio'];slots[e]={c:stores[c].slot(raw) for c in stores};assert min(slots[e].values())>=0
 order=json.loads(open(plan['inputs']['B20_sampling_order']).read());examples=[(e,t,c) for e,t in order[:8] for c in ['clean','background']]
 moments=[torch.zeros_like(p) for p in g['params'] for _ in range(2)]
 records=[]
 for repeat in range(2):
  restore(initial)
  for m in g['mods'].values():m.train();m.zero_grad(set_to_none=True)
  for start in [0,8]:
   ex=examples[start:start+8];b=interface.batch(ex);target=interface.targets(ex,stores,slots)
   action,z,loss=interface.forward(b);aux=F.l1_loss(z,target.to(z.dtype));total=loss+plan['shared_protocol']['lambda_aux']*aux
   assert torch.isfinite(total) and torch.isfinite(action).all() and torch.isfinite(z).all()
   (total/2).backward();records.append(dict(repeat=repeat,microbatch=start//8,action_loss=float(loss.detach()),aux_loss=float(aux.detach()),total=float(total.detach())))
   interface.clear();del b,target,action,z,loss,aux,total
  assert all(p.grad is None or torch.isfinite(p.grad).all() for p in g['params'])
  workspace=[torch.zeros_like(p) for p in g['params']];del workspace
  assert g['counts'](g['before'])==g['counts'](g['ecc']())
 assert all(torch.equal(p.detach().cpu(),initial['parameters'][k][n]) for k,m in g['mods'].items() for n,p in m.named_parameters() if p.requires_grad)
 g['save']('device_preflight.json',dict(status='PASS',optimizer_updates=0,effective_batch=16,microbatch=8,accumulation=2,backward_effective_batches=2,parameters_unchanged=True,model_RNG_isolated=True,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),ECC_increment=False,records=records))
 print('GPU_PREFLIGHT_PASS_ZERO_UPDATES',flush=True)
