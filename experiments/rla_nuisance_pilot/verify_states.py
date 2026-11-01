import sys
from pathlib import Path
import numpy as np,mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_nuisance_pilot.common import *
from rla_nuisance_pilot.prepare import render
def main():
 records=[]
 for s in read(ROOT/'state_protocol.json')['new_development_states']:
  state=np.load(s['state_file']);folder=Path(s['state_file']).parent;baseline=None
  for condition in s['conditions']:
   p=folder/(condition['condition']+'.xml');assert sha(p)==condition['XML_sha256'];rgb,dep,seg,geo,m=render(p.read_text(),state)
   if baseline is None:baseline=(dep,seg,geo)
   else:
    np.testing.assert_array_equal(dep,baseline[0]);np.testing.assert_array_equal(seg,baseline[1])
    for k in geo:np.testing.assert_array_equal(geo[k],baseline[2][k])
   targets={name:np.isin(seg[0,:,:,0],[i for i in range(m.ngeom) if name in (m.geom(i).name or '')])&(seg[0,:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM)) for name in ['akita_black_bowl_1','plate_1']}
   assert all(mask.any() and rgb[0][mask].mean()>2 for mask in targets.values())
   assert 5<rgb.mean()<250 and rgb.std()>5
   records.append(dict(state=s['seed'],condition=condition['condition'],depth_segmentation_physics_camera_exact=True,RGB_mean=float(rgb.mean()),RGB_std=float(rgb.std()),target_pixels={k:int(v.sum()) for k,v in targets.items()},physical_steps=0))
  print('SELECTED_STATE_STATIC_PASS',s['seed'],flush=True)
 save(ROOT/'selected_state_static_acceptance.json',dict(status='PASS',records=records,policy_calls=0,physics_steps=0))
if __name__=='__main__':main()
