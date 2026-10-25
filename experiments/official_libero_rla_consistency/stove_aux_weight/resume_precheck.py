"""Isolated full-batch gate on saved step300; no optimizer update."""
import os,torch
from stove_aux_weight import training
def execute(g):
 original=training.initialize
 def load300(g,kind='rla'):
  plan,prev,_=original(g,kind)
  ck=torch.load(os.environ['RLA_RESUME_PRECHECK'],map_location='cpu')
  assert ck['global_step']==300 and ck['next_physical_sample']==2400
  for k,m in g['mods'].items():
   full=m.state_dict();full.update(ck['parameters'][k]);m.load_state_dict(full,strict=True)
  training.restore(ck)
  return plan,prev,ck
 training.initialize=load300
 training.execute(g)
