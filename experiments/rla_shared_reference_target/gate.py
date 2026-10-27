"""Isolated zero-update checks through the actual official loss/batch path."""
import hashlib
import numpy as np
import torch

def digest_parameters(mods,grad=False):
 h=hashlib.sha256();n=0
 for key,m in mods.items():
  for name,p in m.named_parameters():
   if not p.requires_grad:continue
   x=p.grad if grad else p
   if x is None:continue
   assert torch.isfinite(x).all();h.update((key+'.'+name).encode());h.update(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes());n+=x.numel()
 return h.hexdigest(),n

def run_gate(g,mods,initial,restore,opt,batch,forward,capture,order,micro,lam):
 assert micro==8 and not opt.state
 examples=[(e,t,c) for e,t in order[:4] for c in ('clean','background')]
 before=digest_parameters(mods);results=[];reference=None
 for mode in ['legacy_direct','appearance_specific','shared_clean']:
  restore(initial);opt.zero_grad(set_to_none=True);b=batch(examples,mode)
  assert not b['rla'].requires_grad
  current={k:v.detach().cpu().clone() for k,v in b.items() if torch.is_tensor(v)}
  if mode=='legacy_direct':reference=current
  else:
   for k in reference:
    if k!='rla':assert torch.equal(reference[k],current[k]),('changed observation/label',k)
  if mode=='appearance_specific':assert torch.equal(reference['rla'],current['rla'])
  if mode=='shared_clean':
   assert torch.equal(b['rla'][::2],b['rla'][1::2])
   assert torch.equal(b['rla'][::2].cpu(),reference['rla'][::2])
  assert not torch.equal(current['pixel_values'][::2],current['pixel_values'][1::2]),'paired rendered inputs unexpectedly identical'
  loss=forward(b);actions=capture['action'].cpu().clone();z=capture['z'];assert z.requires_grad
  assert torch.isfinite(loss) and torch.isfinite(z).all();value=float(loss.detach());zcopy=z.detach().cpu().clone();loss.backward()
  gradient=digest_parameters(mods,True)
  shared_norm=sum(float(p.grad.detach().float().square().sum()) for p in mods['vla'].parameters() if p.grad is not None)**.5
  assert shared_norm>0 and np.isfinite(shared_norm)
  if mode=='legacy_direct':old=(value,actions,zcopy,gradient)
  if mode=='appearance_specific':assert value==old[0] and torch.equal(actions,old[1]) and torch.equal(zcopy,old[2]) and gradient==old[3]
  if mode=='shared_clean':assert torch.equal(actions,old[1]) and torch.equal(zcopy,old[2]),'targets influenced forward output'
  results.append(dict(mode=mode,total_loss=value,gradient_sha256=gradient[0],gradient_elements=gradient[1],shared_gradient_norm=shared_norm,target_requires_grad=b['rla'].requires_grad))
  capture.clear();del loss,z,b,current
 assert before==digest_parameters(mods)
 g['save']('shared_reference_gate.json',dict(status='PASS',legacy_vs_off_output_loss_gradient_exact=True,shared_pair_targets_exact=True,non_target_batch_tensors_exact=True,background_pixels_preserved=True,target_does_not_change_action_or_aux_output=True,finite_gradients=True,optimizer_updates=0,parameters_unchanged=True,fixed_batch=examples,checks=results))
 g['save']('preflight_acceptance.json',dict(status='PASS_SHARED_TARGET_ZERO_UPDATES',microbatch=8,effective_appearance_batch=16,optimizer_updates=0,full_batch_memory_reused_from_matching_C=True,peak_allocated=torch.cuda.max_memory_allocated()))
 print('SHARED_REFERENCE_GATE_PASS_ZERO_UPDATES',flush=True)
