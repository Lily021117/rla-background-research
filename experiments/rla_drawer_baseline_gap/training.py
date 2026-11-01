"""Version-checked adaptation of the accepted drawer trainer, not a new loss graph."""
import inspect,hashlib
from rla_new_task_baseline import training as original
from rla_drawer_baseline_gap.common import *
def execute(g):
 source=inspect.getsource(original.execute);base=source
 for old,new in [
  ('else .22408536585365854','else .11204268292682927'),
  ('assert len(order)==4800','assert len(order)==14400 and sha(plan[\'inputs\'][\'B20_sampling_order\'])==plan[\'inputs\'][\'B20_sampling_sha256\']'),
  ('range(start_step+1,601)','range(start_step+1,1801)'),
  ("lr=opt.param_groups[0]['lr'];opt.step();sched.step()","lr=opt.param_groups[0]['lr'];assert abs(lr-(2e-4 if step<=600 else 2e-5))<1e-12;opt.step();sched.step()"),
  ('if step in [100,300,600]:snapshot(step);validate(step)','if step in [300,600,900,1200,1500,1800]:snapshot(step)\n  if step in [600,1200,1800]:validate(step)'),
  ("out/'step0600.pt'","out/'step1800.pt'"),
  ("assert ck['global_step']==600","assert ck['global_step']==1800"),
  ('range(start_step*8,4800,8)','range(start_step*8,14400,8)'),
  ('successful_updates=600,strict_reload=True','successful_updates=1800,new_successful_updates=1800-start_step,actual_update_computations=1800-start_step,replayed_updates=0,seconds=time.time()-started,strict_reload=True'),
 ]:source=replace(source,old,new)
 save(g['o']/'source_adaptation.json',dict(original_sha256=hashlib.sha256(base.encode()).hexdigest(),effective_sha256=hashlib.sha256(source.encode()).hexdigest(),changes='new seeds and targets in plan; fixed half weight; 1800 updates; original scheduler milestone600; no graph/reduction changes'))
 scope=dict(original.__dict__);scope['sha']=sha
 exec(compile(source,__file__,'exec'),scope);scope['execute'](g)
