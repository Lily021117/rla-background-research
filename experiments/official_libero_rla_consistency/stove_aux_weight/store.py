"""Frozen target coordinates; only policy loss weight changes this run."""
import os,json
from pathlib import Path
import numpy as np
from rla.sidecar import RlaStore

class LowWeightStore(RlaStore):
 def __init__(self,path,*args,**kwargs):
  super().__init__(path,*args,**kwargs)
  meta=json.loads((Path(os.environ['CENTER_RUN'])/'normalization.json').read_text())
  np.testing.assert_array_equal(self.offset,np.asarray(meta['mu_C'],dtype=np.float32))
  assert self.scale==np.float32(meta['s_C'])
  source=Path(path).parent.name
  assert source in ('C','D')
  if source=='D':self.offset=np.asarray(meta['mu_D'],dtype=np.float32)
  self.target_source=source
