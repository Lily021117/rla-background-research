"""Read stored finger joint response; never steps or queries a policy."""
import argparse,json
from pathlib import Path
import numpy as np,mujoco
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);a=p.parse_args()
e=next(x for x in json.loads((a.previous/'episode_manifest.json').read_text()) if x['split']=='validation');m=mujoco.MjModel.from_xml_path(str(Path(e['geometry_path']).parent/'clean.xml'));addresses=[int(m.joint('gripper0_finger_joint'+str(i)).qposadr[0]) for i in (1,2)];initial=np.load(a.previous/'episodes'/e['episode']/'trajectory_labels.npz')['states'][0];rows=[]
for condition in ('clean','background'):
    with np.load(a.previous/'development'/condition/'trajectory.npz') as f:
        states=np.r_[initial[None],f['states']];q=states[:,1+np.array(addresses)];opening=q[:,0]-q[:,1];delta=np.diff(opening);commands=f['executed_actions'][:,6];closed=commands==1;opened=commands==-1
        intervals=[];start=None
        for i,value in enumerate(np.r_[closed,False]):
            if value and start is None:start=i
            elif not value and start is not None:
                intervals.append(dict(first_step=start,last_step_inclusive=i-1,opening_before_m=float(opening[start]),opening_after_m=float(opening[i]),opening_change_m=float(opening[i]-opening[start])));start=None
        rows.append(dict(appearance=condition,opening_definition='recorded finger1 qpos minus finger2 qpos, meters; joint separation proxy',initial_opening_m=float(opening[0]),final_opening_m=float(opening[-1]),minimum_opening_m=float(opening.min()),maximum_opening_m=float(opening.max()),close_intervals=intervals,mean_step_opening_change_when_close_m=float(delta[closed].mean()) if closed.any() else None,mean_step_opening_change_when_open_m=float(delta[opened].mean()),actuator_ctrl_recorded=False,controller_internal_current_action_recorded=False,caveat='observed qpos response plus source inspection; not a reconstruction of all controller/actuator internal state'))
with (a.out/'gripper_execution_response.json').open('x') as f:json.dump(rows,f,indent=2)
print(json.dumps(rows,indent=2))
