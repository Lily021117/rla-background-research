"""Full batch with restored optimizer memory, zero updates, isolated process."""
import os
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from control_preserving_rla.common import read,save,load_parameters,restore_rng,digest,sha
from control_preserving_rla.interface import initialize,Interface,enable_checkpointing,masked_l1

def execute(g):
 plan,prev,_=initialize(g);out=g['a'].run.parent;source=Path(os.environ['POLICY_RESUME']);ck=torch.load(source,map_location='cpu')
 assert ck['arm']=='E-D-keep' and ck['global_step']==300 and ck['next_physical_sample']==2400 and ck['configuration']==plan
 assert ck['AMP_scaler'] is None and ck['scheduler']['last_epoch']==300
 load_parameters(g,ck);api=Interface(g,plan,train_only=True);enable_checkpointing(g)
 opt=torch.optim.AdamW(g['params'],lr=2e-4,weight_decay=.01);opt.load_state_dict(ck['optimizer']);restore_rng(ck)
 assert all(float(s['step'])==300 for s in opt.state.values())
 from stove_aux_weight.store import LowWeightStore
 stores={c:LowWeightStore(str(prev/'sidecars/D'/c),chunk=8,num_tokens=16,token_dim=64,norm='center') for c in ('clean','background')}
 slots={e:{c:stores[c].slot(np.load(root/'trajectory_labels.npz')['proprio']) for c in stores} for e,(root,_,_) in api.ds.episodes.items()}
 assert sha(Path(plan['train_only_teacher_cache']))==read(out/'teacher_cache_metadata.json')['cache_sha256']
 cache=np.load(plan['train_only_teacher_cache']);order=read(Path(plan['inputs']['B20_sampling_order']));ex=[(e,t,c) for e,t in order[2400:2408] for c in ('clean','background')]
 for m in g['mods'].values():m.train()
 opt.zero_grad(set_to_none=True);losses=[]
 for start in (0,8):
  examples=ex[start:start+8];b=api.batch(examples);a,z,bc=api.forward(b);target=api.targets(examples,stores,slots)
  aux=F.l1_loss(z,target.to(z.dtype));teach=torch.from_numpy(np.stack([cache[e][t] for e,t,c in examples])).cuda()
  keep=masked_l1(a,teach,torch.ones_like(a,dtype=torch.float32));total=(bc+plan['shared_protocol']['lambda_aux']*aux).float()+keep
  assert torch.isfinite(total);(total/2).backward();losses.append(dict(BC=float(bc.detach()),aux=float(aux.detach()),keep=float(keep.detach())));api.clear()
 assert bool(torch.stack([torch.isfinite(p.grad).all() for p in g['params'] if p.grad is not None]).all())
 assert digest(g['trainable_state']())==digest(ck['parameters'])
 save(out/'resume_precheck_acceptance.json',dict(status='PASS',checkpoint=str(source),sha256=sha(source),restored_step=300,optimizer_steps=300,scheduler_last_epoch=300,next_physical_sample=2400,full_batch=16,microbatch=8,accumulation=2,backend='Math SDPA',dtype='bfloat16',AMP_scaler=None,model_optimizer_scheduler_RNG_sampler_compatible=True,teacher_cache_hash_unchanged=True,parameters_unchanged=True,optimizer_updates=0,peak_allocated=torch.cuda.max_memory_allocated(),losses=losses,GPU=g['GPU'],historical_GPU1_ECC_retained=True))
 print('RESUME_PREFLIGHT_PASS',flush=True)
