import argparse,collections
from common import *
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();ss={s['seed']:s for s in states()};summary=[];timeline=[];fixed=[]
for n,r in enumerate(trials()):
 e=env_make(ss[r['seed']],r['condition']);t=np.load(r['trajectory']);initial=metrics(e);trace=[];lift=None
 for j,s in enumerate(t['states']):
  e.set_state(s);e.sim.forward();x=metrics(e);x['step']=j+1;x['close_command']=bool(t['executed_actions'][j,6]>0);x['lift_m']=x['bowl_2']-initial['bowl_2']
  if lift is None and x['lift_m']>=.03:lift=j+1
  trace.append(x)
 for x in trace:timeline.append(dict(model=r['model'],seed=r['seed'],appearance=r['condition'],relative_to_lift=None if lift is None else x['step']-lift,**x))
 post=trace[lift-1:] if lift else [];end=trace[-1];row=dict(model=r['model'],seed=r['seed'],appearance=r['condition'],recorded_success=r['success'],recomputed_terminal_success=end['goal'],first_lift=lift,postlift_steps=len(post),final_xy_m=end['xy_distance_m'],final_vertical_m=end['vertical_bowl_plate_m'],final_height=end['height_condition'],final_contact=end['contact_condition'],final_xy=end['xy_condition'],postlift_finger_contact_steps=sum(x['finger_contact'] for x in post),postlift_open_command_steps=sum(not x['close_command'] for x in post),first_contact_loss_after_lift=next((x['step'] for i,x in enumerate(post[1:],1) if post[i-1]['finger_contact'] and not x['finger_contact']),None),first_return_below_lift=next((x['step'] for x in post if x['lift_m']<.03),None))
 for c in ('height_condition','contact_condition','xy_condition','goal'):
  row[c+'_first']=next((x['step'] for x in post if x[c]),None);row[c+'_lost_count']=sum(post[i-1][c] and not post[i][c] for i in range(1,len(post)))
 summary.append(row)
 if lift:
  boundary=((lift+7)//8)*8
  for k in sorted(set(min(v,len(trace)-1) for v in (boundary,boundary+16,boundary+40))):fixed.append(dict(source='rollout',model_source=r['model'],seed=r['seed'],appearance=r['condition'],pre_step=k,state_index=k-1,trajectory=r['trajectory']))
 e.close();print('HISTORY',n+1,flush=True)
csvsave(a.out/'stage_summary.csv',summary);csvsave(a.out/'stage_timeline_local_only.csv',timeline);save(a.out/'fixed_rollout_samples.json',fixed)
save(a.out/'history_verification.json',dict(trials=len(summary),new_control_steps=0,terminal_disagreements=sum(x['recorded_success']!=x['recomputed_terminal_success'] for x in summary),cached_pose_note='same saved qpos/qvel recomputed with mj_forward; previous post-step cached poses may differ by integrator phase; never substitute recomputed labels for original evaluation',contacts='geometric contact, not force/stable grasp truth',drop_note='loss of finger contact or height proxy not independently verified drop'))
