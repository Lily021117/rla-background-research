"""Frozen C, clean training states only, deterministic eval output cache."""
import csv
from pathlib import Path
import numpy as np
import torch
from control_preserving_rla.common import save,seed,sha,digest
from control_preserving_rla.interface import initialize,Interface

def execute(g):
 plan,_,ck=initialize(g,teacher=True);out=g['a'].run.parent
 api=Interface(g,plan,train_only=True);assert len(api.ds.episodes)==20
 for m in g['mods'].values():m.eval().requires_grad_(False)
 seed(plan['seed']);cache={};records=[];isolated=False
 with torch.inference_mode():
  for e,(_,_,meta) in api.ds.episodes.items():
   assert meta['split']=='train';N=meta['valid_chunks'];predictions=np.empty((N,8,7),np.float32);labels=np.empty_like(predictions)
   for start in range(0,N,8):
    ex=[(e,t,'clean') for t in range(start,min(start+8,N))];b=api.batch(ex);a,z,_=api.forward(b,'Inference')
    assert torch.isfinite(a).all();predictions[start:start+len(ex)]=a.float().cpu().numpy();labels[start:start+len(ex)]=b['actions'].numpy()
    if not isolated:
     expected=a.clone();b['actions']=torch.full_like(b['actions'],.731);again,_,_=api.forward(b,'Inference');assert torch.equal(expected,again)
     isolated=True
    api.clear();del a,z,b
   cache[e]=predictions
   error=np.abs(predictions-labels)
   for h in range(8):
    for d in range(7):records.append(dict(episode=e,split='train',appearance='clean',windows=N,horizon=h+1,dimension=d,teacher_demo_L1=float(error[:,h,d].mean()),teacher_mean=float(predictions[:,h,d].mean()),demo_mean=float(labels[:,h,d].mean())))
   print('TEACHER_CACHE',e,N,flush=True)
 with Path(plan['train_only_teacher_cache']).open('xb') as f:np.savez(f,**cache)
 with (out/'teacher_demonstration_disagreement.csv').open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 current={k:{n:dict(g['mods'][k].named_parameters())[n].detach().cpu() for n in ps} for k,ps in ck['parameters'].items()}
 assert digest(current)==digest(ck['parameters'])
 save(out/'teacher_cache_metadata.json',dict(status='PASS',teacher_sha256=plan['teacher_sha256'],cache_sha256=sha(Path(plan['train_only_teacher_cache'])),episodes=20,windows=sum(len(x) for x in cache.values()),shape_per_window=[8,7],dtype='float32 (BF16 eval predictions retained exactly)',appearance='clean only',split='train only',normalization='unchanged official task training statistics',mode='eval inference_mode',future_images=False,sidecars_read=False,action_label_value_isolation=isolated,teacher_parameters_unchanged=True,teacher_outputs_are_not_ground_truth=True,optimizer_updates=0,backward=0))
