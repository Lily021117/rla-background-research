"""Information-preserving32-token h8 target path; actual upstream transforms/head."""
import json,hashlib
from pathlib import Path
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([],'GPU')
import torch
from torch import nn
from torch.utils.data import IterableDataset
from rla.action_head import RlaMLPResNet
from rla.data import RlaBatchTransform,RlaCollator,_normalize_with_rla
from rla.sidecar import set_store
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.vla.datasets.rlds.oxe.transforms import libero_dataset_transform
from prismatic.vla.datasets.rlds.traj_transforms import chunk_act_obs
from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.backbones.llm.prompting import PurePromptBuilder

TARGET_HORIZONS=(8,)
ACTION_CHUNK=8
QUERIES=32
TOKEN_DIM=64
assert len(TARGET_HORIZONS)==1 and TARGET_HORIZONS[0]==ACTION_CHUNK

class SparseHead(L1RegressionActionHead):
 """Same official RLA trunk; explicit target_count1≠action_count8, Q32/D64."""
 def __init__(self,hidden):
  nn.Module.__init__(self)
  self.num_task_tokens=512;self.action_dim=7;self.hidden_dim=hidden
  self.model=RlaMLPResNet(num_blocks=24,input_dim=hidden*7,hidden_dim=hidden,output_dim=7,
   use_pro_version=True,num_rla_tokens=32,rla_steps=1,rla_dim=64,action_tokens=True,mask_rla_to_action=True)
 def pop_z_hat(self):return self.model.pop_z_hat()

class SparseStore:
 def __init__(self,root,group):
  self.root=Path(root);self.episodes=json.loads((self.root/'episodes.json').read_text());self.rows=json.loads((self.root/'samples.json').read_text())
  self.targets=np.load(self.root/f'targets_{group}.npy',mmap_mode='r');assert self.targets.shape==(30,1,32,64)
  norm=np.load(self.root/f'normalization_{group}.npz');self.mean=norm['mean'];self.scale=float(norm['scale']);assert self.scale>0
  self.mapping={(r['slot'],r['current_video_index']):i for i,r in enumerate(self.rows)}
  self.slots={}
  for e in self.episodes:
   state=np.load(e['source'])['proprio'].astype('float32');self.slots[self.key(state)]=e['slot']
 def key(self,x):return hashlib.sha1(np.ascontiguousarray(x,dtype='float32').tobytes()).hexdigest()
 def slot(self,states):
  k=self.key(states);assert k in self.slots,'Unknown raw-replay proprio identity (not a regenerated RLDS hash)'
  return self.slots[k]
 def frame(self,slot,t):
  assert (slot,t) in self.mapping,'Missing sparse target; never fill with zeros'
  return np.ascontiguousarray(((self.targets[self.mapping[slot,t]]-self.mean)/self.scale).reshape(1,2048))

class SparseMatchedDataset(IterableDataset):
 """Real source trajectory → real normalizer/chunker/batch-transform, workers0."""
 def __init__(self,root,group,processor,statistics):
  self.root=Path(root);self.store=SparseStore(root,group);self.processor=processor;self.stats=statistics
  self.action_tokenizer=ActionTokenizer(processor.tokenizer)
  self.transform=RlaBatchTransform(self.action_tokenizer,processor.tokenizer,
   image_transform=processor.image_processor.apply_transform,prompt_builder_fn=PurePromptBuilder,
   use_wrist_image=True,use_proprio=True,use_minivlm=True)
 def __iter__(self):
  assert torch.utils.data.get_worker_info() is None,'num_workers must remain0'
  set_store(self.store)
  metadata={k:{kk:tf.convert_to_tensor(vv) for kk,vv in self.stats[k].items()} for k in ['action','proprio']}
  for e in self.store.episodes:
   s=np.load(e['source']);state=s['proprio'];n=len(state)
   traj=libero_dataset_transform(dict(action=tf.convert_to_tensor(s['actions']),observation=dict(state=tf.convert_to_tensor(state))))
   traj['observation']['proprio']=tf.convert_to_tensor(state)
   traj=_normalize_with_rla(traj,metadata,ACTION_PROPRIO_NORMALIZATION_TYPE)
   traj['observation']={k:traj['observation'][k] for k in ['proprio','rla_slot']}
   traj['observation'].update(image_primary=tf.convert_to_tensor(s['images']),image_wrist=tf.convert_to_tensor(s['wrist_images']),timestep=tf.range(n))
   traj['task']=dict(language_instruction=tf.repeat(e['task'].encode(),n));traj['dataset_name']=tf.repeat(b'libero_spatial_no_noops',n)
   traj['absolute_action_mask']=tf.tile([[False]*6+[True]],[n,1])
   traj=chunk_act_obs(traj,window_size=1,future_action_window_size=7)
   for r in self.store.rows:
    if r['slot']!=e['slot']:continue
    t=r['current_video_index'];assert r['future_video_index']==t+8
    frame=tf.nest.map_structure(lambda x:x[t].numpy(),traj)
    np.testing.assert_array_equal(frame['observation']['image_primary'][0],s['images'][t])
    np.testing.assert_array_equal(frame['observation']['image_wrist'][0],s['wrist_images'][t])
    instance=self.transform(frame)
    np.testing.assert_array_equal(instance['rla'],self.store.frame(e['slot'],t))
    assert instance['actions'].shape==(8,7)
    np.testing.assert_array_equal(instance['actions'],frame['action'])
    yield instance

class SparseCollator(RlaCollator):
 """Preserve the batch axis the legacy singleton proprio squeeze removes."""
 def __call__(self,instances):
  out=super().__call__(instances)
  out['proprio']=out['proprio'].reshape(len(instances),8)
  return out

def make_loader(root,group,processor,statistics):
 ds=SparseMatchedDataset(root,group,processor,statistics)
 collator=SparseCollator(processor.tokenizer.model_max_length,processor.tokenizer.pad_token_id)
 return torch.utils.data.DataLoader(ds,batch_size=1,num_workers=0,collate_fn=collator),ds
