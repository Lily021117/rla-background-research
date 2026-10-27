"""Frozen policy outputs on a predeclared train/development sample set; no updates."""
import csv,dataclasses,json,random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from stove_centering.training import initialize

def execute(g):
 plan,prev,initial=initialize(g);root=g['a'].run.parent;del initial
 from coverage_gripper.dataset import DerivedEpisodeDataset
 from rla.data import RlaCollator
 from stove_centering.store import CenterStore
 import rla.loss as lossmod
 lossmod.CFG=dataclasses.replace(lossmod.CFG,lambda_auto=False,lambda_rla=plan['shared_protocol']['lambda_aux'])
 collator=RlaCollator(2048,g['tokenizer'].pad_token_id);ds=DerivedEpisodeDataset(plan['inputs']['data_manifest'],plan['inputs']['action_proprio_statistics'],g['transform'],collator)
 store=CenterStore(str(prev/'sidecars/C/clean'),chunk=8,num_tokens=16,token_dim=64,norm='center')
 slots={e:store.slot(np.load(p/'trajectory_labels.npz')['proprio']) for e,(p,_,_) in ds.episodes.items()}
 selected=json.loads((root.parent/'fixed_representation_samples.json').read_text())
 paths=json.loads((root/'representation_registry.json').read_text());mods=g['mods'];head=g['head'];capture={}
 predict=head.predict_action;pop=head.pop_z_hat
 def pred(*args,**kw):
  a=predict(*args,**kw);capture['action']=a.detach();return a
 def popz():
  z=pop();capture['z']=z.detach();return z
 head.predict_action=pred;head.pop_z_hat=popz
 wrapped=lossmod.make_run_forward_pass(g['ft'].run_forward_pass);rows=[]
 # Official mini-VLM transform randomly pads action-token IDs. These IDs
 # are replaced by action queries; continuous BC labels are not randomized.
 # Diagnose this explicitly, then match preprocessing RNG for paired inputs.
 e,t,_=selected[0];probe=collator([ds.sample(e,t,c) for c in ('clean','background')])
 masks=g['model'].base_model.model._process_action_masks(probe['labels'])
 assert torch.equal(masks[0],masks[1])
 for key in ['actions','proprio']:assert torch.equal(probe[key][0],probe[key][1])
 for key in ['input_ids','labels']:
  assert not torch.any((probe[key][0]!=probe[key][1]) & ~masks[0])
 g['save']('token_padding_audit.json',dict(action_labels_equal=True,action_masks_equal=True,different_token_ids=int((probe['input_ids'][0]!=probe['input_ids'][1]).sum()),differences_only_in_action_query_replacement_positions=True,source='official RLDSBatchTransform random.choices padding; modeling_prismatic._replace_input_embeddings; continuous L1 uses batch actions',diagnostic_only_RNG_matching=True,training_unchanged=True))
 del probe,masks
 common_python_rng=random.getstate()
 for method,path in paths.items():
  random.setstate(common_python_rng)
  ck=torch.load(path,map_location='cpu');assert ck['global_step']==600
  for k,m in mods.items():
   full=m.state_dict();full.update(ck['parameters'][k]);m.load_state_dict(full,strict=True);m.eval().requires_grad_(False)
  del ck
  for e,t,split in selected:
   pair_rng=random.getstate();clean=ds.sample(e,t,'clean');after_clean=random.getstate()
   random.setstate(pair_rng);background=ds.sample(e,t,'background');random.setstate(after_clean)
   ex=[clean,background]
   for item in ex:item['rla']=store.frame(slots[e],t)
   b=collator(ex)
   for key in ['actions','proprio','labels','input_ids','rla']:assert torch.equal(b[key][0],b[key][1]),('pair identity mismatch',e,t,key)
   b['labels']=b['labels'].cuda();b['proprio']=b['proprio'].cuda()
   with torch.inference_mode():
    wrapped(vla=g['model'],action_head=SimpleNamespace(module=head),proprio_projector=g['prop'],batch=b,action_tokenizer=g['at'],device_id='cuda',use_l1_regression=True,use_proprio=True,use_film=False,num_patches=g['model'].vision_backbone.get_num_patches()*2,use_pro_version=True,cfg=SimpleNamespace(phase='Inference'))
   z=capture['z'].float();a=capture['action'].float();target=b['rla'].cuda().float();assert torch.isfinite(z).all()
   cross=z[1]-z[0];den=z[0].norm();action_error=(a-b['actions'].cuda().float()).abs()
   rows.append(dict(method=method,episode=e,timestep=t,split=split,cross_prediction_L1=float(cross.abs().mean()),cross_prediction_MSE=float(cross.square().mean()),cross_prediction_relative_L2=float(cross.norm()/den) if den>1e-8 else None,
    clean_prediction_norm=float(z[0].norm()),background_prediction_norm=float(z[1].norm()),clean_reference_L1=float((z[0]-target[0]).abs().mean()),background_reference_L1=float((z[1]-target[1]).abs().mean()),
    clean_reference_weighted_L1=plan['shared_protocol']['lambda_aux']*float((z[0]-target[0]).abs().mean()),background_reference_weighted_L1=plan['shared_protocol']['lambda_aux']*float((z[1]-target[1]).abs().mean()),
    clean_action_L1=float(action_error[0].mean()),background_action_L1=float(action_error[1].mean())))
   capture.clear();del b,z,a,target
 with (g['o']/'predicted_representation_metrics.csv').open('x') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 g['save']('representation_acceptance.json',dict(status='PASS',rows=len(rows),fixed_pairs=len(selected),optimizer_updates=0,backward=0,teacher=False,paired_actions_proprio_language_labels_and_targets_exact=True,all_metrics_normalized_with_existing_C_statistics=True,not_evidence_of_invariance_from_target_equality_alone=True))
