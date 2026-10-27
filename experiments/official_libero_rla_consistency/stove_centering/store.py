"""Only the offset changes; original sidecars and scale remain read-only."""
import os,json
from pathlib import Path
import numpy as np
from rla.sidecar import RlaStore
class CenterStore(RlaStore):
 def __init__(self,*args,**kw):
  super().__init__(*args,**kw)
  meta=json.loads((Path(os.environ['CENTER_RUN'])/'normalization.json').read_text())
  assert np.array_equal(self.offset,np.array(meta['mu_C'],np.float32))
  assert self.scale==np.float32(meta['s_C'])
  if os.environ['CENTER_ARM'] in ('D-center','D-inv03'):self.offset=np.array(meta['mu_D'],np.float32)
  else:assert os.environ['CENTER_ARM'] in ('C','C-shared-target','D-old','preflight','evaluation','representation')
