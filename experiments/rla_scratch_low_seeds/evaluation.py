"""Identical frozen deployment; only the admitted finite trial budget differs."""
import inspect,hashlib
from rla_scratch_policy_aux import evaluation as original
from rla_scratch_low_seeds.common import save
def execute(g):
 source=inspect.getsource(original.execute);old="protocol['trials'] in (160,640)";new="protocol['trials']==600 and protocol.get('low_seed_replication')"
 assert source.count(old)==1;patched=source.replace(old,new)
 save(g['o']/'source_adaptation.json',dict(original_sha256=hashlib.sha256(source.encode()).hexdigest(),effective_sha256=hashlib.sha256(patched.encode()).hexdigest(),replacement=[old,new]))
 scope=dict(original.__dict__);exec(compile(patched,__file__,'exec'),scope);scope['execute'](g)
