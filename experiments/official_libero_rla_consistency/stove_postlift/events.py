import argparse,collections,pickle,mujoco
from scipy.spatial.transform import Rotation
from common import *
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();groups=collections.defaultdict(list)
for r in csv.DictReader((a.out/'stage_timeline_local_only.csv').open()):groups[r['model'],r['seed'],r['appearance']].append(r)
events=[];aligned=[]
for (model,seed,appearance),rs in groups.items():
 post=[r for r in rs if r['relative_to_lift'] and int(r['relative_to_lift'])>=0];op=next((r for r in post if r['close_command']=='False'),None)
 events.append(dict(model=model,seed=seed,appearance=appearance,first_postlift_open_step=None if op is None else op['step'],first_open_xy_m=None if op is None else op['xy_distance_m'],first_open_outside_goal=None if op is None else op['xy_condition']=='False',first_open_postlift_delay=None if op is None else op['relative_to_lift']))
 for lag in (0,8,16,32,64):
  x=next((r for r in post if int(r['relative_to_lift'])==lag),None)
  if x:
   q=np.array([float(x['bowl_quat_'+str(i)]) for i in range(4)]);z=Rotation.from_quat(np.roll(q,-1)).as_matrix()[:,2];aligned.append(dict(model=model,seed=seed,appearance=appearance,lag=lag,xy_m=float(x['xy_distance_m']),height_m=float(x['vertical_bowl_plate_m']),bowl_tilt_radians=float(np.arccos(np.clip(z[2],-1,1))),**{k:float(x[k]) for k in x if k.startswith(('bowl_','ee_')) and k not in ('ee_bowl_distance_m',)},finger_contact=x['finger_contact']))
csvsave(a.out/'release_events.csv',events);csvsave(a.out/'lift_aligned_state_metrics.csv',aligned)
prefixrows=[]
for st in states()[:5]:
 for arm in ('C','D-center'):
  f=a.out/'handoffs'/f"{st['seed']}_{arm}_prefix/snapshot_local_only.pkl"
  if not f.exists():continue
  with f.open('rb') as stream:s=pickle.load(stream)
  e=env_make(st,'clean');m,d=e.sim.model._model,e.sim.data._data;mujoco.mj_setState(m,d,s['integration'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d);x=metrics(e);bq=np.array([x['bowl_quat_'+str(i)] for i in range(4)]);eq=np.array([x['ee_quat_'+str(i)] for i in range(4)]);relative=(Rotation.from_quat(np.roll(eq,-1)).inv()*Rotation.from_quat(np.roll(bq,-1))).as_rotvec();prefixrows.append(dict(prefix=arm,seed=st['seed'],**x,relative_orientation_x=relative[0],relative_orientation_y=relative[1],relative_orientation_z=relative[2]));e.close()
csvsave(a.out/'prefix_geometry_metrics.csv',prefixrows)
