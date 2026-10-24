"""Existing trajectory/video analysis only; mj_forward never mj_step."""
import argparse,collections,os
os.environ['CUDA_VISIBLE_DEVICES']=''
from common import *
import mujoco,imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=Path(read(a.out/'configuration.json')['source']);a.out=a.out/'rollout_details';a.out.mkdir(exist_ok=False);trials=read(root/'policy_run/per_trial_results.json');states={x['seed']:x for x in read(root/'ae_run/development_protocol/evaluation_protocol.json')['new_development_states']};rows=[];times=[];(a.out/'figures').mkdir(exist_ok=False);(a.out/'video_panels_local_only').mkdir(exist_ok=False)
for r in trials:
 if r['model'] not in ('C','D'):continue
 t=np.load(r['trajectory']);folder=Path(states[r['seed']]['state_file']).parent;m=mujoco.MjModel.from_xml_path(str(folder/(r['condition']+'.xml')));d=mujoco.MjData(m);bid=m.body('akita_black_bowl_1_main').id;sid=m.site('gripper0_grip_site').id;finger=[int(m.joint('gripper0_finger_joint'+str(i)).qposadr[0]) for i in (1,2)];initial=np.load(states[r['seed']]['state_file']);before=np.vstack([initial,t['states'][:-1]]);trace=[]
 for i,s in enumerate(before):
  d.time=s[0];d.qpos[:]=s[1:1+m.nq];d.qvel[:]=s[1+m.nq:];mujoco.mj_forward(m,d);contacts=0
  for contact in d.contact[:d.ncon]:
   b1,b2=int(m.geom_bodyid[contact.geom1]),int(m.geom_bodyid[contact.geom2]);other=b2 if b1==bid else b1 if b2==bid else None
   if other is not None and 'finger' in (m.body(other).name or ''):contacts+=1
  relative=d.site_xpos[sid]-d.xpos[bid];cmd=t['executed_actions'][i];trace.append(dict(step=i,distance=float(np.linalg.norm(relative)),horizontal=float(np.linalg.norm(relative[:2])),vertical=float(relative[2]),finger_opening=float(d.qpos[finger[0]]-d.qpos[finger[1]]),finger_contacts=contacts,close_command=bool(cmd[6]>0)))
 close=np.array([x['close_command'] for x in trace]);events=np.flatnonzero(close&~np.r_[False,close[:-1]]);first=int(events[0]) if len(events) else None;firstrow=trace[first] if first is not None else None;near=np.flatnonzero(np.array([x['distance'] for x in trace])<=.10);contact=np.flatnonzero(np.array([x['finger_contacts'] for x in trace])>0)
 row=dict(model=r['model'],seed=r['seed'],appearance=r['condition'],stage=r['stage'],success=r['success'],close_command_ever=bool(close.any()),first_close_step=first,close_events=len(events),first_close_distance_m=None if firstrow is None else firstrow['distance'],first_close_horizontal_m=None if firstrow is None else firstrow['horizontal'],first_close_vertical_m=None if firstrow is None else firstrow['vertical'],first_close_before_near=bool(first is not None and (not len(near) or first<near[0])),first_near_step=int(near[0]) if len(near) else None,recomputed_finger_contact_steps=len(contact),first_contact_step=int(contact[0]) if len(contact) else None,max_lift_m=r['max_lift_m'],max_pose_reproduction_error=0.)
 # Validate recorded poststep bowl and EE labels against the same-state geometry.
 for j in (0,len(t['states'])//2,len(t['states'])-1):
  s=t['states'][j];d.time=s[0];d.qpos[:]=s[1:1+m.nq];d.qvel[:]=s[1+m.nq:];mujoco.mj_forward(m,d);err=max(float(abs(d.xpos[bid]-t['target_bowl_positions'][j]).max()),float(abs(d.site_xpos[sid]-t['eef_positions'][j]).max()));row['max_pose_reproduction_error']=max(row['max_pose_reproduction_error'],err)
 row['cached_poststep_pose_exact_at_1e8']=row['max_pose_reproduction_error']<1e-8;rows.append(row)
 times.extend(dict(model=r['model'],seed=r['seed'],appearance=r['condition'],**x) for x in trace)
csvsave(a.out/'rollout_grasp_timing.csv',rows);csvsave(a.out/'rollout_timing_local_only.csv',times)
summary=[]
for model in ('C','D'):
 for c in ('clean','seen_background','unseen_background'):
  rs=[r for r in rows if r['model']==model and r['appearance']==c];valid=[r for r in rs if r['first_close_step'] is not None];summary.append(dict(model=model,appearance=c,trials=len(rs),successes=sum(r['success'] for r in rs),ever_close=sum(r['close_command_ever'] for r in rs),close_before_10cm=sum(r['first_close_before_near'] for r in rs),ever_recomputed_finger_contact=sum(r['recomputed_finger_contact_steps']>0 for r in rs),ever_lift_3cm=sum(r['max_lift_m']>=.03 for r in rs),first_close_distance_m_mean=float(np.mean([r['first_close_distance_m'] for r in valid])) if valid else None,first_close_horizontal_m_mean=float(np.mean([r['first_close_horizontal_m'] for r in valid])) if valid else None,first_close_vertical_m_mean=float(np.mean([r['first_close_vertical_m'] for r in valid])) if valid else None))
csvsave(a.out/'rollout_timing_summary.csv',summary)
seed=min(states)
for condition in ('clean','seen_background','unseen_background'):
 fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True)
 for model in ('C','D'):
  rs=[r for r in times if r['seed']==seed and r['appearance']==condition and r['model']==model]
  for ax,key in zip(axes,['distance','finger_opening','close_command']):ax.plot([x['step'] for x in rs],[x[key] for x in rs],label=model);ax.set_ylabel(key);ax.legend()
 axes[0].axhline(.1,color='grey',ls='--');axes[-1].set_xlabel('executed control step (0-based)');fig.suptitle(f'Fixed first seed {seed}, {condition}');fig.tight_layout();fig.savefig(a.out/'figures'/f'timing_{condition}.png');plt.close(fig)
 chosen={r['model']:r for r in trials if r['seed']==seed and r['condition']==condition and r['model'] in ('C','D')};readers={k:imageio.get_reader(v['video']) for k,v in chosen.items()};indices=np.linspace(0,int(np.ceil(chosen['C']['steps']/8))-1,5,dtype=int);fig,axes=plt.subplots(2,5,figsize=(18,5))
 for i,model in enumerate(('C','D')):
  for j,index in enumerate(indices):
   k=min(index,int(np.ceil(chosen[model]['steps']/8))-1);axes[i,j].imshow(readers[model].get_data(k));axes[i,j].set_title(f'{model}, control {k*8}');axes[i,j].axis('off')
  readers[model].close()
 fig.suptitle(f'Same frozen state {seed}: {condition}, C/D stored videos');fig.tight_layout();fig.savefig(a.out/'video_panels_local_only'/f'{condition}.png');plt.close(fig)
save(a.out/'rollout_diagnostic_verification.json',dict(trials=120,new_rollouts=0,simulation_steps=0,state_geometry_recomputed=True,cached_poststep_pose_exact_trials=sum(r['cached_poststep_pose_exact_at_1e8'] for r in rows),max_cached_pose_error=max(r['max_pose_reproduction_error'] for r in rows),cached_pose_warning='Saved simulator state reconstructed with mj_forward is not assumed bitwise identical to cached xpos after mj_step; discrepancies retained, no time-offset correction. Original phase labels remain separately reported.',contacts='mj_forward contacts recomputed from recorded state and same XML; not independently logged forces or proof of stable grasp',selection='first frozen seed per appearance, uniform C duration; D clamped only if shorter',stage_thresholds='inherited predeclared .10m near/.03m lift, descriptive proxies'))
print('ROLLOUT DIAGNOSTIC COMPLETE',flush=True)
