"""Exact historical deployment with a frozen new-state manifest, no training."""
import hashlib, inspect, json
from pathlib import Path
from rla_scratch_policy_aux import evaluation as original
from rla_low_lr_continue.evaluation import initialize

def execute(g):
 root=g['a'].run.parent
 protocol=json.loads((root/'evaluation_protocol.json').read_text())
 gate=json.loads((root.parent/'scoring_gate.json').read_text())
 assert gate['status']=='FROZEN_COMMITTED_REMOTE_VERIFIED'
 assert hashlib.sha256((root/'evaluation_protocol.json').read_bytes()).hexdigest()==gate['evaluation_protocol_sha256']
 frozen=Path(protocol['frozen_state_protocol'])
 assert hashlib.sha256(frozen.read_bytes()).hexdigest()==gate['state_protocol_sha256']
 assert protocol['trials']==30*len(protocol['states']) and 20<=len(protocol['states'])<=50
 source=inspect.getsource(original.execute)
 changes=[("protocol['trials'] in (160,640)","protocol['trials']==30*len(protocol['states']) and protocol.get('generated_independent_states')"),
          ("['successful_updates']==600","['successful_updates']==1800"),
          ('physical_states=20','physical_states=len(protocol["states"])'),
          ('heldout_read=False','heldout_read=True, sealed_read_purpose="source audit only; scored states newly generated"')]
 patched=source
 for a,b in changes:
  assert patched.count(a)==1,(a,patched.count(a));patched=patched.replace(a,b)
 g['save']('source_adaptation.json',dict(original_sha256=hashlib.sha256(source.encode()).hexdigest(),effective_sha256=hashlib.sha256(patched.encode()).hexdigest(),changes=changes,deployment_control_unchanged=True))
 scope=dict(original.__dict__);scope['initialize']=initialize
 exec(compile(patched,__file__,'exec'),scope);scope['execute'](g)
