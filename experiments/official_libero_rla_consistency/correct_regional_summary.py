"""Verify a pure robot/base category subdivision, preserving all original pixels."""
import argparse,csv,json
from pathlib import Path
import numpy as np,mujoco
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
old=list(csv.DictReader((a.run/'regional_audit/metrics.csv').open()));new=list(csv.DictReader((a.run/'regional_audit_v2/metrics.csv').open()))
def key(x):return (x['episode'],x['timestep'],x['camera'],x['region'])
lookup={key(x):x for x in new};assert len(old)==len(new)==len(lookup)
robot=('robot_and_gripper_interior','robot_base_interior')
for x in old:
    if x['region'] not in robot:assert x==lookup[key(x)]
    elif x['region']==robot[0]:
        pair=[lookup[(*key(x)[:3],r)] for r in robot]
        assert sum(int(z['pixels']) for z in pair)==int(x['pixels'])
        assert sum(int(z['changed_pixels']) for z in pair)==int(x['changed_pixels'])
        np.testing.assert_allclose(sum(int(z['pixels'])*float(z['MAE_255'] or 0) for z in pair),int(x['pixels'])*float(x['MAE_255'] or 0),rtol=1e-12,atol=1e-8)
        assert max(int(z['max_channel_255'] or 0) for z in pair)==int(x['max_channel_255'] or 0)
summary=[]
for name in sorted({x['region'] for x in new}):
    rows=[x for x in new if x['region']==name and int(x['pixels'])];n=sum(int(x['pixels']) for x in rows)
    summary.append(dict(region=name,pixels=n,changed_pixels=sum(int(x['changed_pixels']) for x in rows),MAE_255=sum(int(x['pixels'])*float(x['MAE_255']) for x in rows)/n if n else None,max_channel_255=max((int(x['max_channel_255']) for x in rows),default=None)))
e=json.loads((a.run/'episode_manifest.json').read_text())[0];m=mujoco.MjModel.from_xml_path(str(Path(e['geometry_path']).parent/'clean.xml'));bid=m.body('robot0_link0').id;ancestors=[]
while bid:
    ancestors.append(dict(body=m.body(bid).name,joints=int(m.body_jntnum[bid])));bid=int(m.body_parentid[bid])
assert all(x['joints']==0 for x in ancestors)
for name,obj in [('regional_summary_v2.json',summary),('regional_correction.json',dict(status='PASS_CATEGORY_CONSERVATION',rows=len(new),static_base_ancestry=ancestors,unchanged_nonrobot_regions=True,robot_union_pixel_counts_and_errors_unchanged=True,RGB_modified=False,new_policy_rollouts=0,previous_outputs_preserved=True))]:
    with (a.run/name).open('x') as f:json.dump(obj,f,indent=2)
print(json.dumps(summary,indent=2))
