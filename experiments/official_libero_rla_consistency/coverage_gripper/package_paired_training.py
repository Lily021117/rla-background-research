"""Publish only sanitized configuration and aggregate results, never raw artifacts."""
import argparse,json,csv,collections,re
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args();a.destination.mkdir(exist_ok=False)
def clean(x):
    text=json.dumps(x,ensure_ascii=False) if not isinstance(x,str) else x
    dataset=Path(json.loads((a.run/'B20/training_plan.json').read_text())['manifest']).parent
    text=text.replace(str(a.run),'${RUN_DIR}').replace(str(dataset),'${DATASET_RUN}')
    return re.sub(r'/data/[^\s"\n,;]+','${LOCAL_PATH}',text)
for name in ('report.md','failure_stage_analysis.md','initialization_and_sampling_audit.json','delivery_verification.json','success_summary.json'):
    (a.destination/name).write_text(clean((a.run/name).read_text()))
for arm in ('B4','B20'):
    (a.destination/f'training_config_{arm}.json').write_text(clean((a.run/arm/'training_plan.json').read_text()))
for name in ('training_metrics.csv','paired_results.csv'):
    (a.destination/name).write_text(clean((a.run/name).read_text()))
groups=collections.defaultdict(list)
for r in csv.DictReader((a.run/'validation_metrics.csv').open()):groups[(r['arm'],r['step'],r['set'],r['appearance'],r['position'])].append(r)
rows=[]
for key,rs in groups.items():
    row=dict(zip(['arm','step','set','appearance','position'],key));row['windows']=sum(int(x['windows']) for x in rs)
    for k in ('action_L1','translation_L1','rotation_L1','gripper_L1'):row[k]=float(np.average([float(x[k]) for x in rs],weights=[int(x['windows']) for x in rs]))
    rows.append(row)
with (a.destination/'validation_summary.csv').open('x') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
