import argparse,collections
from common import *
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();coverage=list(csv.DictReader((a.out/'demo_phase_coverage.csv').open()));first={r['episode']:int(r['first_lift']) if r['first_lift'] else None for r in coverage};rows=[]
for arm,path in [('C',S/'policy_run/C/isolated_smoke/validation_metrics.csv'),('D-center',CENTER/'D-center/isolated_smoke/validation_metrics.csv')]:
 groups=collections.defaultdict(list)
 for r in csv.DictReader(path.open()):
  t=int(r['timestep']);phase='postlift' if first[r['episode']] is not None and t>=first[r['episode']] else 'prelift';groups[int(r['step']),r['appearance'],phase].append(r)
 for (step,appearance,phase),rs in groups.items():rows.append(dict(model=arm,step=step,appearance=appearance,phase=phase,windows=len(rs),**{k:float(np.mean([float(x[k]) for x in rs])) for k in ('action_L1','translation_L1','rotation_L1','gripper_L1')}))
csvsave(a.out/'historical_validation_phase_curves.csv',rows)
save(a.out/'training_phase_log_limit.json',dict(training_update_logs='aggregate batch loss only; no per-window per-phase training errors are recoverable without new inference',sampling_order_recovered=True,postlift_exposures=2596,total_physical_exposures=4800,postlift_fraction=2596/4800,interpretation='postlift windows not absent; exposure count alone does not prove phase learnability or sufficient variability'))
