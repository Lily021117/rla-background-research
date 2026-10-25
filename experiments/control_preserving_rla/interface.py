"""Reuse accepted official batch/forward; attach no teacher or target to inference inputs."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from control_preserving_rla.common import read,load_parameters

def initialize(g,teacher=False):
 root=g['a'].run.parent;plan=read(root/'run_plan.json')
 path=Path(plan['teacher_checkpoint']) if teacher else Path(plan['common_student_step0'])
 ck=torch.load(path,map_location='cpu');assert ck['global_step']==(600 if teacher else 0)
 load_parameters(g,ck)
 g['params']=[p for m in g['mods'].values() for p in m.parameters() if p.requires_grad]
 return plan,Path(plan['previous']),ck

class Interface:
 def __init__(self,g,plan,train_only=False):
  from coverage_gripper.dataset import DerivedEpisodeDataset
  from rla.data import RlaCollator
  self.g=g;self.plan=plan;self.capture={};self.head=g['head'];self.model=g['model'];self.prop=g['prop']
  self.collator=RlaCollator(2048,g['tokenizer'].pad_token_id)
  self.ds=DerivedEpisodeDataset(plan['train_only_manifest'] if train_only else plan['inputs']['data_manifest'],plan['inputs']['action_proprio_statistics'],g['transform'],self.collator)
  self.original_predict=self.head.predict_action
  def predict(*args,**kwargs):
   a=self.original_predict(*args,**kwargs);self.capture['action']=a;return a
  self.head.predict_action=predict
  self.audit=[]
 def batch(self,examples):
  items=[self.ds.sample(e,t,c) for e,t,c in examples];b=self.collator(items)
  b['labels']=b['labels'].cuda();b['proprio']=b['proprio'].cuda()
  assert int(self.model._process_action_masks(b['labels']).sum())==len(examples)*self.g['NUM_TOKENS']
  assert b['actions'].shape==(len(examples),8,7)
  for i,(e,t,c) in enumerate(examples):
   meta=self.ds.episodes[e][2];assert 0<=t<meta['valid_chunks']
   assert meta['task'].lower() in self.g['tokenizer'].decode(b['input_ids'][i].tolist()).lower()
   if len(self.audit)<16:self.audit.append(dict(episode=e,timestep=t,appearance=c,language=meta['task'],action_shape=list(b['actions'][i].shape),proprio_shape=list(b['proprio'][i].shape),action_mask='all 8x7 valid; no tail padding',future_image_inputs=False))
  return b
 def forward(self,b,phase='Training'):
  self.capture.clear()
  action_loss,_=self.g['ft'].run_forward_pass(vla=self.model,action_head=SimpleNamespace(module=self.head),proprio_projector=self.prop,batch=b,action_tokenizer=self.g['at'],device_id='cuda',use_l1_regression=True,use_proprio=True,use_film=False,num_patches=self.model.vision_backbone.get_num_patches()*2,use_pro_version=True,cfg=SimpleNamespace(phase=phase))
  z=self.head.pop_z_hat();assert z is not None and z.shape==(len(b['actions']),8,1024)
  return self.capture['action'],z,action_loss
 def targets(self,examples,stores,slots):
  z=np.stack([stores[c].frame(slots[e][c],t) for e,t,c in examples]);assert z.shape==(len(examples),8,1024) and np.isfinite(z).all()
  return torch.from_numpy(z).cuda()
 def clear(self):self.capture.clear()

def reset_auxiliary(g,seed):
 with torch.random.fork_rng(devices=[0]):
  torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  g['head'].model.layer_norm2_rla.reset_parameters();g['head'].model.fc2_rla.reset_parameters()

def enable_checkpointing(g):
 from torch.utils.checkpoint import checkpoint
 model,head=g['model'],g['head']
 model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
 for block in head.model.mlp_resnet_blocks:
  original=block.forward
  def cp(*args,_original=original,**kwargs):return checkpoint(_original,*args,use_reentrant=False,preserve_rng_state=True,**kwargs)
  block.forward=cp
 for v in (model.vision_backbone.featurizer,model.vision_backbone.fused_featurizer):
  if hasattr(v,'set_grad_checkpointing'):v.set_grad_checkpointing(True)

def masked_l1(a,b,mask):
 assert a.shape==b.shape==mask.shape
 return ((a.float()-b.float()).abs()*mask).sum()/mask.sum().clamp(min=1)
