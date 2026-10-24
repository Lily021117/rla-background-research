"""Uniform recorded-state gripper timing; mj_forward only, no new rollout."""
import argparse,os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stove_diagnosis'))
from common import *
import mujoco
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();plan=read(a.out/'run_plan.json');trials=read(a.out/'per_trial_results.json');states={x['seed']:x for x in read(Path(plan['previous'])/'development_protocol/evaluation_protocol.json')['new_development_states']};rows=[]
for r in trials:
 assert r['status']=='COMPLETE';t=np.load(r['trajectory']);state=states[r['seed']];folder=Path(state['state_file']).parent;m=mujoco.MjModel.from_xml_path(str(folder/(r['condition']+'.xml')));d=mujoco.MjData(m);bid=m.body('akita_black_bowl_1_main').id;sid=m.site('gripper0_grip_site').id;initial=np.load(state['state_file']);before=np.vstack([initial,t['states'][:-1]]);distances=[];contacts=[]
 flat=np.concatenate(t['predicted_chunks'],axis=0)[:len(t['executed_actions'])].copy();flat[:,6]=-np.sign(2*flat[:,6]-1);np.testing.assert_array_equal(flat,t['executed_actions'])
 for i,s in enumerate(before):
  d.time=s[0];d.qpos[:]=s[1:1+m.nq];d.qvel[:]=s[1+m.nq:];mujoco.mj_forward(m,d);distances.append(float(np.linalg.norm(d.site_xpos[sid]-d.xpos[bid])));n=0
  for contact in d.contact[:d.ncon]:
   b1,b2=int(m.geom_bodyid[contact.geom1]),int(m.geom_bodyid[contact.geom2]);other=b2 if b1==bid else b1 if b2==bid else None
   if other is not None and 'finger' in (m.body(other).name or ''):n+=1
  contacts.append(n)
 close=t['executed_actions'][:,6]>0;events=np.flatnonzero(close&~np.r_[False,close[:-1]]);near=np.flatnonzero(np.array(distances)<=.10);first=int(events[0]) if len(events) else None
 rows.append(dict(arm=r['model'],seed=r['seed'],condition=r['condition'],success=r['success'],stage=r['stage'],steps=r['steps'],issued_close=bool(close.any()),first_close_step=first,close_events=len(events),first_close_EE_target_distance_m=distances[first] if first is not None else None,first_near_step=int(near[0]) if len(near) else None,closed_before_10cm=bool(first is not None and (not len(near) or first<int(near[0]))),finger_contact_steps=int(np.sum(np.array(contacts)>0)),max_lift_m=r['max_lift_m'],predicted_executed_action_match=True))
csvsave(a.out/'rollout_gripper_timing.csv',rows);save(a.out/'rollout_timing_verification.json',dict(trials=len(rows),new_rollouts=0,mj_steps=0,scope='same saved pre-command states, mj_forward geometry/contact reconstruction only; contact is not proof of grasp force or stable grasp',cached_pose_warning='historical post-mj_step xpos not assumed bitwise equal to state-restored mj_forward; no time shift or repair applied',all_predicted_executed_actions_match=True))
# Uniform first frozen state, all conditions; not selected by model outcome.
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
panels=a.out/'video_panels_local_only';panels.mkdir(exist_ok=False);seed=min(states)
for condition in ('clean','seen_background','unseen_background'):
 fig,axes=plt.subplots(2,6,figsize=(18,6))
 for i,arm in enumerate(('D-old','D-center')):
  trial=next(r for r in trials if r['model']==arm and r['seed']==seed and r['condition']==condition);reader=imageio.get_reader(trial['video'])
  for j,step in enumerate((0,40,80,120,160,208)):
   frame=min(step//8,int(np.ceil(trial['steps']/8))-1);axes[i,j].imshow(reader.get_data(frame));axes[i,j].set_title(f'{arm}, control {frame*8}');axes[i,j].axis('off')
  reader.close()
 fig.suptitle(f'First frozen state {seed}, {condition}; recorded images only');fig.tight_layout();fig.savefig(panels/f'{condition}.png');plt.close(fig)
print('RECORDED TIMING COMPLETE',len(rows),flush=True)
