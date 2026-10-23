"""Saved-state approach/contact diagnosis; no new rollout or dynamics."""
import argparse,json,csv
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from label_rollout_audit import geometry
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();protocol=json.loads((a.run/'evaluation_protocol.json').read_text());trials=json.loads((a.run/'per_trial_results.json').read_text());rows=[]
for r in trials:
    source=next(t for t in protocol['trials'] if t['arm']==r['arm'] and t['episode']==r['episode'] and t['appearance']==r['appearance']);lab=np.load(Path(source['root'])/'trajectory_labels.npz');m=mujoco.MjModel.from_xml_path(str(Path(source['xml_directory'])/'clean.xml'));raw=lab['actions'][:,6];events=np.flatnonzero((raw[:-1]<0)&(raw[1:]>0))+1
    expert_indices=sorted(set(i for e in events for i in range(max(0,e-2),min(len(raw),e+3))));assert expert_indices
    expert=geometry(m,lab['states'][expert_indices]);ep=np.array([x['relative'] for x in expert]);er=np.array([x['rotation'] for x in expert]);d=np.load(r['trajectory']);actual=d['executed_actions'];pred=d['predicted_chunks'].reshape(-1,7)[:len(actual)].copy();pred[:,6]=-np.sign(2*pred[:,6]-1);np.testing.assert_array_equal(actual,pred)
    pre=geometry(m,np.concatenate([lab['states'][0:1],d['states'][:-1]]));post=geometry(m,d['states']);positions=np.array([x['relative'] for x in pre]);error=np.linalg.norm(positions[:,None]-ep[None],axis=2);nearest=error.argmin(1);t=int(error.min(1).argmin());closed=np.flatnonzero(actual[:,6]>0);first=int(closed[0]) if len(closed) else None
    rows.append(dict(arm=r['arm'],episode=r['episode'],appearance=r['appearance'],success=r['success'],commands_exact=True,steps=len(actual),queries=len(d['predicted_chunks']),close_steps=len(closed),first_close_step=first,first_close_nearest_expert_position_error_m=float(error[first].min()) if first is not None else None,nearest_expert_position_error_m=float(error[t,nearest[t]]),orientation_error_at_nearest_position_rad=float(Rotation.from_matrix(er[nearest[t]].T@pre[t]['rotation']).magnitude()),nearest_position_step=t,command_at_nearest_position=float(actual[t,6]),finger_contact_steps=sum(x['finger_contact']>0 for x in post),robot_contact_steps=sum(x['robot_contact']>0 for x in post),target_max_lift_m=r['target_max_lift_m']))
with (a.run/'rollout_pose_and_timing.csv').open('x') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (a.run/'rollout_pose_caveat.json').open('x') as f:json.dump(dict(source='saved states restored with mj_forward, no mj_step',expert='same development source trajectory, all close events +/-2 steps; no selection by result',position_only_does_not_prove_valid_grasp=True,controller_internal_state_not_available=True,optimizer_updates=0,backward=0,new_rollouts=0),f,indent=2)
