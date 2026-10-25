"""Exclusive output, hashing and explicit complete checkpoint restoration."""
import hashlib
import json
import random
import numpy as np
import torch

def read(p):return json.loads(p.read_text())
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2)
def restore_rng(ck):
 torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng'])
def rng_state():return dict(torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),numpy_rng=np.random.get_state())
def seed(v):random.seed(v);np.random.seed(v);torch.manual_seed(v);torch.cuda.manual_seed_all(v)
def load_parameters(g,ck):
 for k,m in g['mods'].items():
  names={n for n,p in m.named_parameters() if p.requires_grad}
  assert names==set(ck['parameters'][k]),k
  state=m.state_dict();state.update(ck['parameters'][k]);m.load_state_dict(state,strict=True)
def digest(parameters):
 h=hashlib.sha256()
 for mod,ps in sorted(parameters.items()):
  for name,p in sorted(ps.items()):h.update((mod+name).encode());h.update(p.contiguous().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
