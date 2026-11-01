import inspect,hashlib
from rla_scratch_policy_aux import evaluation as original
from rla_low_lr_continue.evaluation import initialize
from rla_nuisance_pilot.common import *
def execute(g):
 assert read(ROOT/'scoring_gate.json')['committed_before_scoring']
 source=inspect.getsource(original.execute)
 for old,new in [("protocol['trials'] in (160,640)","protocol['trials']==900 and protocol.get('nuisance_pilot')"),("['successful_updates']==600","['successful_updates']==1800"),('physical_states=20','physical_states=25')]:
  assert source.count(old)==1;source=source.replace(old,new)
 save(g['o']/'source_adaptation.json',dict(original_sha256=hashlib.sha256(inspect.getsource(original.execute).encode()).hexdigest(),effective_sha256=hashlib.sha256(source.encode()).hexdigest(),deployment_unchanged=True))
 scope=dict(original.__dict__);scope['initialize']=initialize
 exec(compile(source,__file__,'exec'),scope);scope['execute'](g)
