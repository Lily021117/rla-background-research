"""Reuse the frozen training function verbatim except explicit C-low target dispatch.

No historical source edits: bind a constant half weight, and select C labels for
the new arm. The exact source replacement is asserted and recorded before training.
"""
import inspect,hashlib
import pickle
from rla_scratch_policy_aux import training as frozen
from rla_scratch_c_low.common import weight
OLD="target_arm='C' if preflight or arm=='C-fixed' else 'D'"
NEW="target_arm='C' if preflight or arm in ('C-fixed','C-low') else 'D'"
def execute(g):
 source=inspect.getsource(frozen.execute);assert source.count(OLD)==1
 transformed=source.replace(OLD,NEW)
 g['save']('training_source_adaptation.json',dict(source_module=frozen.__file__,source_sha256=hashlib.sha256(source.encode()).hexdigest(),replacement_from=OLD,replacement_to=NEW,lambda_RLA=weight('C-low',0),effective_source_sha256=hashlib.sha256(transformed.encode()).hexdigest(),other_training_statements_unchanged=True))
 namespace=dict(vars(frozen));namespace['weight']=weight
 def checked_initialize(context):
  result=frozen.initialize(context)
  expected=frozen.read(context['a'].run.parent/'expected_initialization.json');h=hashlib.sha256()
  for k,m in context['mods'].items():
   for n,p in m.named_parameters():
    if p.requires_grad:h.update((k+'.'+n).encode());h.update(p.detach().cpu().contiguous().view(frozen.torch.uint8).numpy().tobytes())
  r=hashlib.sha256(pickle.dumps((frozen.random.getstate(),frozen.np.random.get_state())))
  for t in [frozen.torch.get_rng_state()]+frozen.torch.cuda.get_rng_state_all():r.update(t.cpu().numpy().tobytes())
  assert h.hexdigest()==expected['parameter_sha256'] and r.hexdigest()==expected['RNG_hash']
  context['save']('paired_initialization_guard.json',dict(status='PASS',reference='corresponding original C-fixed',parameter_sha256=h.hexdigest(),RNG_hash=r.hexdigest(),verified_before_any_update=True))
  return result
 namespace['initialize']=checked_initialize
 exec(compile(transformed,__file__,'exec'),namespace)
 namespace['execute'](g)
