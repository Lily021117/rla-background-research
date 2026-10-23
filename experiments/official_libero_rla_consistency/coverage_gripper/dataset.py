"""Derived source identities routed into official VLA batch components."""
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
import torch
from prismatic.vla.datasets.rlds.traj_transforms import chunk_act_obs
from prismatic.vla.datasets.rlds.utils.data_utils import normalize_action_and_proprio
from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE,NUM_TOKENS,STOP_INDEX,IGNORE_INDEX

class DerivedEpisodeDataset:
    def __init__(self,manifest,statistics,transform,collator):
        self.manifest=json.loads(Path(manifest).read_text());assert self.manifest['status']=='PASS'
        self.stats={k:{kk:tf.convert_to_tensor(vv) for kk,vv in values.items()} for k,values in json.loads(Path(statistics).read_text()).items()}
        self.transform,self.collator=transform,collator;self.episodes={}
        for e in self.manifest['episodes']:
            root=Path(e['root']);lab=np.load(root/'trajectory_labels.npz');act=lab['actions'].astype(np.float32).copy();act[:,6]=1-np.clip(act[:,6],0,1);N=len(act)
            trajectory=dict(action=tf.constant(act),observation=dict(proprio=tf.constant(lab['proprio']),timestep=tf.range(N)),task=dict(language_instruction=tf.constant([e['task']]*N)),dataset_name=tf.constant([self.manifest.get('dataset_version','cookie_dual224_fk_v4')]*N),absolute_action_mask=tf.constant(np.tile([False]*6+[True],(N,1))))
            normalized=normalize_action_and_proprio(trajectory,self.stats,ACTION_PROPRIO_NORMALIZATION_TYPE)
            chunk=chunk_act_obs(normalized,window_size=1,future_action_window_size=7)
            self.episodes[e['episode']]=(root,tf.nest.map_structure(lambda x:x.numpy(),chunk),e)
    def sample(self,episode,timestep,appearance):
        root,ch,e=self.episodes[episode];assert appearance in ('clean','background') and 0<=timestep<e['valid_chunks']
        raw=tf.nest.map_structure(lambda x:x[timestep].copy() if isinstance(x[timestep],np.ndarray) else x[timestep],ch)
        with np.load(root/'frames'/f'{timestep:04}.npz') as f:rgb=f[appearance].transpose(0,2,3,1)
        raw['observation']['image_primary']=rgb[0:1];raw['observation']['image_wrist']=rgb[1:2];item=self.transform(raw)
        prefix=len(item['input_ids'])-NUM_TOKENS;item['input_ids']=torch.cat([item['input_ids'],torch.tensor([STOP_INDEX])]);item['labels']=item['input_ids'].clone();item['labels'][:prefix]=IGNORE_INDEX
        return item
    def batch(self,identities):
        return self.collator([self.sample(*key) for key in identities])
