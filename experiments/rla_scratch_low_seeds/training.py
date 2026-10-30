"""Reuse accepted training function with explicit source-selection and half weight."""
import inspect,hashlib
from rla_scratch_policy_aux import training as original
from rla_scratch_low_seeds.common import weight,LAMBDA,save
def execute(g):
 source=inspect.getsource(original.execute)
 old="target_arm='C' if preflight or arm=='C-fixed' else 'D'"
 new="target_arm='C' if preflight or arm in ('C-fixed','C-low') else 'D'"
 assert source.count(old)==1
 patched=source.replace(old,new)
 save(g['o']/'source_adaptation.json',dict(original_sha256=hashlib.sha256(source.encode()).hexdigest(),effective_sha256=hashlib.sha256(patched.encode()).hexdigest(),replacement=[old,new],lambda_low=LAMBDA))
 scope=dict(original.__dict__);scope.update(weight=weight,LAMBDA=LAMBDA)
 exec(compile(patched,__file__,'exec'),scope);scope['execute'](g)
