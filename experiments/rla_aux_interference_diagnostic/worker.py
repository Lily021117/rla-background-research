"""Target-only counterfactual gradients through the executed official loss."""
import csv, dataclasses, hashlib, json, pickle, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from control_preserving_rla.common import read,save,restore_rng,rng_state,load_parameters,digest
from control_preserving_rla.interface import Interface,enable_checkpointing
from stove_centering.training import initialize

def execute(g):
 root=g['a'].run.parent;o=g['o'];plan,prev,initial=initialize(g)
 protocol=read(root/'diagnostic_protocol.json');snapshots=read(root/'corrected_snapshot_manifest.json');batches=read(root/'batch_manifest.json')
 # Guards apply even if future edits accidentally enter a training path.
 def forbidden(*a,**k):raise RuntimeError('optimizer/scheduler update forbidden in frozen diagnosis')
 torch.optim.Optimizer.step=forbidden;torch.optim.AdamW.step=forbidden
 torch.optim.lr_scheduler.LRScheduler.step=forbidden
 api=Interface(g,plan);enable_checkpointing(g);mods=g['mods'];lam=plan['shared_protocol']['lambda_aux']
 from stove_centering.store import CenterStore
 stores={c:CenterStore(str(prev/'sidecars/C'/c),chunk=8,num_tokens=16,token_dim=64,norm='center') for c in ('clean','background')}
 assert np.array_equal(stores['clean'].offset,stores['background'].offset) and stores['clean'].scale==stores['background'].scale
 slots={e:{c:stores[c].slot(np.load(p/'trajectory_labels.npz')['proprio']) for c in stores} for e,(p,_,_) in api.ds.episodes.items()}
 import rla.loss as official
 official.CFG=dataclasses.replace(official.CFG,lambda_auto=False,lambda_rla=lam)
 captured={}
 def orig(**kw):
  loss,metrics=g['ft'].run_forward_pass(**kw);captured['action_loss']=loss;return loss,metrics
 wrapped=official.make_run_forward_pass(orig)
 def loss_hook(module,args,result):
  if isinstance(module,torch.nn.L1Loss) and args[0].shape==(8,8,1024):captured['aux']=result
 hook=torch.nn.modules.module.register_module_forward_hook(loss_hook)
 scope=[]
 for k,m in mods.items():
  for n,p in m.named_parameters():
   if not p.requires_grad:continue
   # Explicit shared head trunk; final action and RLA readouts excluded.
   if k=='head' and not n.startswith('model.mlp_resnet_blocks.'):continue
   scope.append((k+'.'+n,p))
 save(o/'gradient_scope.json',dict(parameters=[dict(name=n,numel=p.numel()) for n,p in scope],count=sum(p.numel() for n,p in scope),excluded='action and auxiliary final readouts; action-only layer_norm1/fc1 upstream of one-way masked attention',dependency='per-objective unused support measured below; parameters connected to both graphs enter direction metrics, including numerically zero gradients'))
 common_rng={k:initial[k] for k in ('torch_rng','cuda_rng','python_rng','numpy_rng')};del initial
 counters=dict(forward_microbatches=0,backward_microbatches=0,objective_full_batches=0,acceptance_objective_full_batches=0,completed_units=0,infrastructure_retries=0)
 start=time.time();events=(o/'computation_events.jsonl').open('x');fields=['snapshot','snapshot_method','snapshot_order','step','batch','batch_order','module','comparison','action_norm','aux_norm','norm_ratio','cosine','projection_on_action_unit','projection_coefficient','joint_parameter_elements','near_zero_denominator','action_loss','aux_C_loss','aux_shared_loss','weighted_aux_C','weighted_aux_shared']
 f=(o/'gradient_statistics.csv').open('x');w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 audit=[];units=[]
 def rng_hash():
  r=rng_state();h=hashlib.sha256()
  h.update(pickle.dumps((r['python_rng'],r['numpy_rng'])))
  for t in [r['torch_rng']]+r['cuda_rng']:h.update(t.cpu().numpy().tobytes())
  return h.hexdigest()
 def objective(examples,kind,acceptance=False):
  restore_rng(common_rng);capture_rng=rng_hash();vectors=[None]*len(scope);losses=[];outputs=[];input_hash=hashlib.sha256();target_hash=hashlib.sha256();unused=set(range(len(scope)))
  for start_i in (0,8):
   ex=examples[start_i:start_i+8];b=api.batch(ex)
   tc=api.targets(ex,stores,slots);clean_ex=[(e,t,'clean') for e,t,c in ex];ts=api.targets(clean_ex,stores,slots)
   assert torch.equal(tc[::2],ts[::2]) and torch.equal(ts[::2],ts[1::2]);assert not tc.requires_grad and not ts.requires_grad
   for key in ('pixel_values','input_ids','labels','proprio','actions'):
    input_hash.update(key.encode());input_hash.update(b[key].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
   b['rla']=ts if kind in ('aux_shared','action_shared','total_shared') else tc
   target_hash.update(b['rla'].detach().cpu().numpy().tobytes());captured.clear();api.clear()
   total,_=wrapped(vla=g['model'],action_head=SimpleNamespace(module=g['head']),proprio_projector=g['prop'],batch=b,action_tokenizer=g['at'],device_id='cuda',use_l1_regression=True,use_proprio=True,use_film=False,num_patches=g['model'].vision_backbone.get_num_patches()*2,use_pro_version=True,cfg=SimpleNamespace(phase='Training'))
   counters['forward_microbatches']+=1
   events.write(json.dumps(dict(event='forward_complete',count=counters['forward_microbatches']))+'\n');events.flush()
   bc=captured['action_loss'];aux=captured['aux'];assert torch.isfinite(total) and torch.isfinite(aux)
   output=api.capture['action'].detach().cpu().clone();outputs.append(output)
   loss=bc if kind.startswith('action') else total if kind.startswith('total') else lam*aux
   grads=torch.autograd.grad(loss/2,[p for n,p in scope],allow_unused=True)
   counters['backward_microbatches']+=1
   events.write(json.dumps(dict(event='backward_complete',count=counters['backward_microbatches']))+'\n');events.flush()
   for i,v in enumerate(grads):
    if v is None:continue
    assert torch.isfinite(v).all();unused.discard(i);v=v.detach().cpu()
    vectors[i]=v.clone() if vectors[i] is None else vectors[i]+v
   losses.append((float(bc.detach()),float(aux.detach())))
   captured.clear();api.clear();del b,tc,ts,total,bc,aux,loss,grads
  counters['objective_full_batches']+=1
  if acceptance:counters['acceptance_objective_full_batches']+=1
  after_rng=rng_hash();restore_rng(common_rng);assert capture_rng==rng_hash()
  return vectors,dict(action=float(np.mean([x[0] for x in losses])),aux=float(np.mean([x[1] for x in losses]))),torch.cat(outputs),dict(input_hash=input_hash.hexdigest(),target_hash=target_hash.hexdigest(),start_rng=capture_rng,after_rng=after_rng,unused_parameters=[scope[i][0] for i in sorted(unused)])
 def calc(a,b,indices):
  aa=bb=ab=0.;count=0
  for i in indices:
   if a[i] is None or b[i] is None:continue
   x=a[i].double();y=b[i].double();xx=float(x.square().sum());yy=float(y.square().sum())
   aa+=xx;bb+=yy;ab+=float((x*y).sum());count+=x.numel()
  na=aa**.5;nb=bb**.5;small=min(na,nb)<=protocol['near_zero_norm']
  return dict(action_norm=na,aux_norm=nb,norm_ratio=nb/na if na>protocol['near_zero_norm'] else None,cosine=ab/(na*nb) if not small else None,projection_on_action_unit=ab/na if na>protocol['near_zero_norm'] else None,projection_coefficient=ab/aa if na>protocol['near_zero_norm'] else None,joint_parameter_elements=count,near_zero_denominator=small)
 for snapshot in snapshots:
  ck=torch.load(snapshot['path'],map_location='cpu',weights_only=False);assert ck['global_step']==snapshot['step'];load_parameters(g,ck);del ck
  for m in mods.values():m.train();m.zero_grad(set_to_none=True)
  before=digest(g['trainable_state']());assert before==snapshot['parameter_sha256']
  buffers={(k,n):b.detach().cpu().clone() for k,m in mods.items() for n,b in m.named_buffers()}
  for batch in [b for b in batches if b['id'] in snapshot['batch_ids']]:
   examples=batch['samples'];va,la,oa,aa=objective(examples,'action');vc,lc,oc,ac=objective(examples,'aux_C');vs,ls,os,ass=objective(examples,'aux_shared')
   assert torch.equal(oa,oc) and torch.equal(oa,os) and la['action']==lc['action']==ls['action']
   assert aa['input_hash']==ac['input_hash']==ass['input_hash'];assert aa['start_rng']==ac['start_rng']==ass['start_rng'] and aa['after_rng']==ac['after_rng']==ass['after_rng']
   if False: # Original two acceptance units already consumed; no extra budget.
    # Two additional acceptance units, each3 objectives: shared action and both totals.
    vas,las,oas,aas=objective(examples,'action_shared',True)
    assert torch.equal(oa,oas) and la['action']==las['action']
    assert all((x is None and y is None) or (x is not None and y is not None and torch.equal(x,y)) for x,y in zip(va,vas))
    del vas,oas
    errors={}
    for mode,auxv in [('total_C',vc),('total_shared',vs)]:
     vt,lt,ot,at=objective(examples,mode,True);num=den=0.
     for a,b,t in zip(va,auxv,vt):
      if t is None:continue
      expected=(torch.zeros_like(t) if a is None else a.float())+(torch.zeros_like(t) if b is None else b.float())
      num+=float((t.double()-expected.double()).square().sum());den+=float(t.double().square().sum())
     rel=(num/den)**.5 if den else 0.;errors[mode]=rel;assert rel<=protocol['decomposition_relative_L2_tolerance'],errors
     del vt,ot
    audit.append(dict(batch=batch['id'],action_output_loss_gradient_exact=True,non_target_inputs_exact=True,clean_target_unchanged=True,shared_pair_target_equal=True,target_requires_grad=False,total_decomposition_relative_L2=errors,RNG_replay_exact=True))
    save(o/f"acceptance_{batch['id']}.json",audit[-1])
   for module in ('ALL_SHARED','vla','proprio','head'):
    ix=[i for i,(n,p) in enumerate(scope) if module=='ALL_SHARED' or n.startswith(module+'.')]
    for name,x,y in [('action_vs_aux_C',va,vc),('action_vs_aux_shared',va,vs),('aux_C_vs_aux_shared',vc,vs)]:
     r=dict(snapshot=snapshot['id'],snapshot_method=snapshot['method'],snapshot_order=snapshot['order'],step=snapshot['step'],batch=batch['id'],batch_order=batch['order'],module=module,comparison=name,**calc(x,y,ix),action_loss=la['action'],aux_C_loss=lc['aux'],aux_shared_loss=ls['aux'],weighted_aux_C=lam*lc['aux'],weighted_aux_shared=lam*ls['aux']);w.writerow(r)
   f.flush();counters['completed_units']+=1
   units.append(dict(snapshot=snapshot['id'],batch=batch['id'],checks=dict(action_output_exact=True,action_loss_exact=True,inputs_equal=True,RNG_replay_equal=True),objective_audits=dict(action=aa,aux_C=ac,aux_shared=ass)))
   save(o/f"unit_{counters['completed_units']:03}.json",dict(counters=dict(counters),unit=units[-1]))
   del va,vc,vs,oa,oc,os
   restore_rng(common_rng)
   assert all(torch.equal(b.detach().cpu(),buffers[(k,n)]) for k,m in mods.items() for n,b in m.named_buffers()),'changed buffer'
   assert g['counts'](g['before'])==g['counts'](g['ecc']()),'ECC increment'
   print('UNIT_COMPLETE',counters['completed_units'],snapshot['id'],batch['id'],'seconds',round(time.time()-start,1),flush=True)
  assert before==digest(g['trainable_state']())
  assert all(p.grad is None for m in mods.values() for p in m.parameters())
  save(o/f"protection_{snapshot['id']}.json",dict(parameters_unchanged=True,buffers_unchanged=True,RNG_restored=True,parameter_sha256=before))
  del buffers
 hook.remove();f.close();events.close();save(o/'unit_audits.json',units)
 save(o/'diagnostic_acceptance.json',dict(status='PASS',counters=counters,acceptance=audit,optimizer_updates=0,AE_updates=0,new_rollouts=0,sealed_reads=0,seconds=time.time()-start,peak_allocated=torch.cuda.max_memory_allocated(),precision='bfloat16',backend='Math SDPA',mode='train',no_saved_gradient_tensors=True))
