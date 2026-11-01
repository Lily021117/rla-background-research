"""Same accepted drawer deployment; gate each method independently after all Clean."""
import inspect,hashlib
from rla_new_task_baseline import evaluation as original
from rla_drawer_baseline_gap.common import *
def execute(g):
 source=inspect.getsource(original.execute);base=source
 for old,new in [
  ("root/'dataset_v2/development_protocol/evaluation_protocol.json'","DATA/'development_protocol/evaluation_protocol.json'"),
  ("root/'dataset_v2/normalization_statistics_expanded_train_only.json'","DATA/'normalization_statistics_expanded_train_only.json'"),
  ("'formal/step0600.pt'","'formal/step1800.pt'"),
  ("['successful_updates']==600","['successful_updates']==1800"),
  ("ck['global_step']==600","ck['global_step']==1800"),
  ('global_step=600','global_step=1800'),
  ("gate='one method both sampling orders clean>=12/20'","gate='per method all three full policy seeds clean>=16/20'"),
  ("gate=any(all(counts[f'{seed}_{m}']>=12 for seed in plan['orders']) for m in methods)","passed={m:all(counts[f'{seed}_{m}']>=16 for seed in plan['orders']) for m in methods};gate=any(passed.values())"),
  ('denominator=20,passed=gate','denominator=20,passed=gate,passed_methods=passed'),
  ("for key in models:\n     if (seed,condition,key) in completed:","for key in models:\n     if stage=='background' and not passed[key.split('_',1)[1]]:continue\n     if (seed,condition,key) in completed:"),
  ('assert len(rows) in [120,600]','assert len(rows) in [120,360,600]'),
 ]:source=replace(source,old,new)
 save(g['o']/'source_adaptation.json',dict(original_sha256=hashlib.sha256(base.encode()).hexdigest(),effective_sha256=hashlib.sha256(source.encode()).hexdigest(),deployment_unchanged=True,changes='final1800; methods and full seeds; per-method all-three>=16 gate; accepted existing drawer states'))
 scope=dict(original.__dict__);scope['DATA']=DATA
 exec(compile(source,__file__,'exec'),scope);scope['execute'](g)
