"""Isolated center-store gate, no optimizer updates."""
import os,json,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from stove_centering.training import initialize,restore
from stove_centering.store import CenterStore
def execute(g):
 plan,prev,initial=initialize(g);mods=g['mods'];model,head,prop=mods['vla'],mods['head'],mods['proprio']
 from coverage_gripper.dataset import DerivedEpisodeDataset
 from rla.data import RlaCollator
 collate=RlaCollator(2048,g['tokenizer'].pad_token_id);ds=DerivedEpisodeDataset(plan['inputs']['data_manifest'],plan['inputs']['action_proprio_statistics'],g['transform'],collate)
 order=json.loads(Path(plan['inputs']['B20_sampling_order']).read_text());xs=[(e,t,c) for e,t in order[:4] for c in ('clean','background')]
 capture={};predict=head.predict_action
 def pred(*args,**kw):
  out=predict(*args,**kw);capture['action']=out;return out
 head.predict_action=pred
 results=[];action_hash=[];aux_hash=[]
 for arm in ('D-old','D-center'):
  os.environ['CENTER_ARM']=arm;restore(initial)
  for m in mods.values():m.zero_grad(set_to_none=True);m.eval()
  stores={c:CenterStore(str(prev/'sidecars/D'/c),chunk=8,num_tokens=16,token_dim=64,norm='center') for c in ('clean','background')};items=[]
  for e,t,c in xs:
   root=ds.episodes[e][0];slot=stores[c].slot(np.load(root/'trajectory_labels.npz')['proprio']);assert slot>=0;it=ds.sample(e,t,c);it['rla']=stores[c].frame(slot,t);items.append(it)
  b=collate(items);b['labels']=b['labels'].cuda();b['proprio']=b['proprio'].cuda()
  act,_=g['ft'].run_forward_pass(vla=model,action_head=SimpleNamespace(module=head),proprio_projector=prop,batch=b,action_tokenizer=g['at'],device_id='cuda',use_l1_regression=True,use_proprio=True,use_film=False,num_patches=model.vision_backbone.get_num_patches()*2,use_pro_version=True,cfg=SimpleNamespace(phase='Inference'))
  z=head.pop_z_hat();aux=torch.nn.functional.l1_loss(z,b['rla'].cuda().to(z.dtype));loss=act+plan['shared_protocol']['lambda_aux']*aux;assert torch.isfinite(loss);loss.backward()
  grads=[p.grad for m in mods.values() for p in m.parameters() if p.grad is not None];assert all(torch.isfinite(v).all() for v in grads)
  action_hash.append(hashlib.sha256(capture['action'].detach().cpu().view(torch.uint8).numpy().tobytes()).hexdigest());aux_hash.append(hashlib.sha256(z.detach().cpu().view(torch.uint8).numpy().tobytes()).hexdigest())
  results.append(dict(arm=arm,action_loss=float(act),auxiliary_loss=float(aux),finite_gradients=True))
  del loss,act,aux,z,b;capture.clear()
 assert action_hash[0]==action_hash[1] and aux_hash[0]==aux_hash[1]
 for k,m in mods.items():
  for n,p in m.named_parameters():
   if p.requires_grad:assert torch.equal(p.detach().cpu(),initial['parameters'][k][n])
 bias={n:list(p.shape) for n,p in head.named_parameters() if 'fc2_rla' in n and n.endswith('bias')};assert bias
 g['save']('gate.json',dict(status='PASS',outputs_identical=True,action_hashes=action_hash,aux_hashes=aux_hash,aux_output_bias=bias,automatic_center_compensation=False,initial_parameters_unchanged=True,optimizer_updates=0,backward_calls=2,checks=results,peak_allocated=torch.cuda.max_memory_allocated(),same_full_microbatch=8,formal_effective_batch=16,reused_activation_checkpointing_gate=True))
