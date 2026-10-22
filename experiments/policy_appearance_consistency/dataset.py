"""Read only current-state RGB/FK and verified eight-command chunks.

Requires the frozen Step4.4.1 CSV schema. No .pt policy batches or RLA sidecars
are read. A task-specific policy must select one exact source task name.
"""
import csv, json, hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.normalize_util import get_image_range_normalizer


def checksum(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8388608),b''): h.update(b)
    return h.hexdigest()


class PairedDataset(Dataset):
    def __init__(self, root, split, task, normalizer=None):
        assert split in ('train','validation')
        root=Path(root)
        transitions=list(csv.DictReader(open(root/'transition_manifest.csv')))
        samples=list(csv.DictReader(open(root/'sample_manifest.csv')))
        index={(s['episode'],int(s['source_state_id']),s['camera']):s for s in samples}
        self.entries=[]
        for t in transitions:
            if t['split']!=split or t['valid']!='True' or Path(t['source_hdf5']).stem!=task: continue
            s=index[t['episode'],int(t['source_start']),'agentview']
            assert all(s[k]=='True' for k in ('geometry_check','segmentation_equal','depth_equal','finite'))
            assert int(t['source_future'])-int(t['source_start'])==8
            assert int(t['action_start_inclusive'])==int(t['source_start'])
            assert int(t['action_stop_exclusive'])==int(t['source_future'])
            assert checksum(s['array'])==s['array_SHA256']
            with np.load(s['array'],allow_pickle=False) as z:
                state=z['fk_proprio'].astype(np.float32)
            action=np.asarray(json.loads(t['actions']),dtype=np.float32)
            assert state.shape==(8,) and action.shape==(8,7)
            assert np.isfinite(state).all() and np.isfinite(action).all()
            self.entries.append((t,s,state,action))
        assert self.entries, 'No accepted rows for explicit task/split'
        if normalizer is None:
            assert split=='train', 'Validation cannot fit normalization'
            normalizer=LinearNormalizer()
            normalizer.fit({'state':np.stack([x[2] for x in self.entries]),
                            'action':np.concatenate([x[3] for x in self.entries])},mode='limits',last_n_dims=1)
            normalizer['image']=get_image_range_normalizer()
        self.normalizer=normalizer

    def __len__(self): return len(self.entries)

    def __getitem__(self,i):
        t,s,state,action=self.entries[i]
        obs=[]
        for appearance in ('clean','background'):
            assert checksum(s[appearance])==s[appearance+'_SHA256']
            rgb=np.asarray(Image.open(s[appearance]).convert('RGB')).copy()
            image=torch.from_numpy(rgb).permute(2,0,1).float()[None,None]/255.
            obs.append(self.normalizer.normalize({'image':image,'state':torch.from_numpy(state)[None]}))
        target=torch.zeros(12,7);target[:8]=self.normalizer['action'].normalize(torch.from_numpy(action))
        return {'clean':obs[0],'background':obs[1],'target':target,
                'mask':torch.tensor([True]*8+[False]*4),'transition_id':t['transition_id'],
                'episode':t['episode']}
