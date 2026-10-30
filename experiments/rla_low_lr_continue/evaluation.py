import inspect,hashlib
from rla_scratch_policy_aux import evaluation as original
from rla_low_lr_continue.common import save
from stove_centering.training import initialize as restored_initialize
def initialize(g):
 result=restored_initialize(g);assert result[2]['global_step']==600
 return result
def execute(g):
 source=inspect.getsource(original.execute);patched=source
 for a,b in [("protocol['trials'] in (160,640)","protocol['trials']==600 and protocol.get('low_lr_continuation')"),("['successful_updates']==600","['successful_updates']==1800")]:
  assert patched.count(a)==1;patched=patched.replace(a,b)
 save(g['o']/'source_adaptation.json',dict(original_sha256=hashlib.sha256(source.encode()).hexdigest(),effective_sha256=hashlib.sha256(patched.encode()).hexdigest(),changes='admit600trials/step1800; initialize bootstrap from verified600 then strictly load selected1800; deployment unchanged'))
 scope=dict(original.__dict__);scope['initialize']=initialize
 exec(compile(patched,__file__,'exec'),scope);scope['execute'](g)
