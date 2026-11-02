"""Reuse frozen V2 evaluator with explicit, asserted loop/processor adaptations.
The emitted effective evaluator is retained locally for audit; no policy math is replaced.
"""
import os,inspect,random,hashlib,json
import numpy as np
import torch
from PIL import Image
from rla_observation_calibration.common import *
from rla_observation_calibration.transforms import Hook,intrinsic
from rla_v2_frozen_c_eval import evaluation as prior

def rng_get():return (random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
def rng_set(x):random.setstate(x[0]);np.random.set_state(x[1]);torch.set_rng_state(x[2]);torch.cuda.set_rng_state_all(x[3])

def audit(g,mods,weights,models,stateset,infer,proprio,task,root):
 samples=read(BASE/'image_acceptance/samples.json');sample=next(s for s in samples if s['task']==task);a=np.load(sample['path']);images=[im.transpose(1,2,0).copy() for im in a['clean']];p=a['proprio'];processor=g['imageproc'];hook=Hook(processor);records=[]
 if os.environ['OBS_STAGE']=='calibration':
  for entry in models:
   for k,m in mods.items():
    full=m.state_dict();full.update(weights[entry['model_id']][k]);m.load_state_dict(full,strict=True);m.eval().requires_grad_(False)
   state=rng_get();wrapped=processor.apply_transform;processor.apply_transform=hook.original
   base_pixels=[processor(Image.fromarray(im),return_tensors='pt')['pixel_values'].clone() for im in images]
   old=infer(images,p,mods['head']);rng_set(state);processor.apply_transform=wrapped
   hook.set_context('clean',TASK_NAMES[task],digest(a['state']),0)
   new_pixels=[processor(Image.fromarray(im),return_tensors='pt')['pixel_values'] for im in images]
   assert all(torch.equal(x,y) for x,y in zip(base_pixels,new_pixels))
   hook.set_context('clean',TASK_NAMES[task],digest(a['state']),0);new=infer(images,p,mods['head']);rng_set(state)
   np.testing.assert_array_equal(old,new)
   records.append(dict(model=entry['model_id'],pixel_exact=True,action_exact=True,max_abs_action_delta=float(np.abs(old-new).max()),policy_forwards=2,RNG_restored=True))
 save(root/task/'clean_identity.json',dict(records=records,policy_forwards=sum(r['policy_forwards'] for r in records),skipped_if_not_calibration=True,source_sample=sample,normalization=processor.tvf_normalize_params,geometry_resize=[str(x) for x in processor.tvf_resize_params],geometry_crop=processor.tvf_crop_params))
 # Intrinsics metadata from actual evaluation XML; no camera pose or dynamics change.
 import mujoco
 geometry=[]
 for s in stateset:
  m=mujoco.MjModel.from_xml_path(s['xmls']['clean'])
  for cam in CAMS:
   f=112/np.tan(np.deg2rad(m.cam_fovy[m.camera(cam).id])/2);K=np.array([[-f,0,111.],[0,f,112.],[0,0,1.]])
   for c in read(root/'stage_conditions.json')[task]:
    new,A=intrinsic(K,c);geometry.append(dict(state_sha256=s['state_sha256'],camera=cam,condition=c,K_original=K.tolist(),K_transformed=new.tolist(),pixel_affine=A.tolist(),extrinsics='unchanged; wrist follows online state',convention='original cx=112 mirrored x -> 223-x; negative fx'))
 save(root/task/'camera_transform_metadata.json',geometry)
 return hook

def execute(g):
 stage=os.environ['OBS_STAGE'];task=os.environ['OBS_TASK'];root=BASE/stage
 assert read(BASE/'protocol_commit.json')['protocol_sha256']==sha(BASE/'calibration_protocol.json')
 conditions=read(root/'stage_conditions.json')[task]
 # All geometry and success logic remains the audited V2 code. Only clean XML and input pixels differ.
 src=inspect.getsource(prior.execute)
 changes=[("task=os.environ['V2_EVAL_TASK']","task=os.environ['OBS_TASK']"),
  ("out=ROOT/task/'closed_loop'","hook=audit(g,mods,weights,models,stateset,infer,proprio,task,ROOT)\n out=ROOT/task/'closed_loop'"),
  ("xml=s['xmls'][condition];assert sha(xml)==s['xml_sha256'][condition]","xml=s['xmls']['clean'];assert sha(xml)==s['xml_sha256']['clean']"),
  ("s['xml_sha256'][condition]","s['xml_sha256']['clean']"),
  ("chunk=infer(images,p,head)","hook.set_context(condition,TASK_NAMES[task],s['state_sha256'],step);chunk=infer(images,p,head);assert hook.camera_index==2"),
  ("assert len(rows)==240","assert len(rows)==len(stateset)*len(CONDITIONS)*len(models)"),
  ("dict(valid=240,actual_attempts=240,invalid=0","dict(valid=len(rows),actual_attempts=len(rows),invalid=0")]
 for old,new in changes:
  assert old in src,old;src=src.replace(old,new)
 def technical(t):
  old=prior.ROOT;prior.ROOT=root
  try:prior.technical(t)
  finally:prior.ROOT=old
 scope=dict(prior.__dict__);scope.update(ROOT=root,CONDITIONS=conditions,technical=technical,audit=audit,TASK_NAMES=TASK_NAMES)
 save(root/task/'effective_evaluator_identity.json',dict(source_sha256=hashlib.sha256(inspect.getsource(prior.execute).encode()).hexdigest(),effective_sha256=hashlib.sha256(src.encode()).hexdigest(),changes=changes,video_note='video shows raw Clean online render; actual transformed inputs audited separately; no V2 XML used'))
 (root/task/'effective_evaluator.py').write_text(src)
 exec(compile(src,'frozen_effective_evaluator','exec'),scope);scope['execute'](g)
