"""Identical frozen deployment; admit only two C-low models x20 states x5 appearances."""
import inspect
from rla_scratch_policy_aux import evaluation as frozen
OLD="protocol['trials'] in (160,640)"
NEW="protocol['trials']==200 and protocol.get('C_low_completion')"
def execute(g):
 source=inspect.getsource(frozen.execute);assert source.count(OLD)==1
 namespace=dict(vars(frozen));exec(compile(source.replace(OLD,NEW),__file__,'exec'),namespace)
 namespace['execute'](g)
