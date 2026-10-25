"""Four-arm paired development analysis. Never changes recorded success labels."""
import argparse,json,csv,collections,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stove_postlift.common import read,save,csvsave,states,env_make,metrics,S,CENTER
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();o=a.out
post=S.parents[1]/'stove_postlift/run_20261025_01'
new=read(o/'per_trial_results.json');assert len(new)==120 and all(r['status']=='COMPLETE' for r in new)
old=[r for r in read(S/'policy_run/per_trial_results.json') if r['model']=='C']+[r for r in read(CENTER/'per_trial_results.json') if r['model']=='D-center']
rows=old+new;arms=['C','C-low','D-center','D-center-low'];conditions=read(o/'evaluation_protocol.json')['conditions'];seeds=[s['seed'] for s in states()]
index={(r['model'],r['seed'],r['condition']):r for r in rows};assert len(index)==240
for seed in seeds:
 for cond in conditions:
  rs=[index[m,seed,cond] for m in arms]
  for key in ('state_sha256','xml_sha256','language'):assert len({r[key] for r in rs})==1,(key,seed,cond)
save(o/'historical_protocol_audit.json',dict(matched_trials=240,identical_state_XML_language=True,control_horizon=220,chunk=8,source='same accepted evaluation code; unchanged process_action and official On predicate',historical_results_unmodified=True))
summary=[]
for m in arms:
 for cond in conditions:
  rs=[index[m,s,cond] for s in seeds];summary.append(dict(model=m,appearance=cond,successes=sum(r['success'] for r in rs),trials=20,success_rate=np.mean([r['success'] for r in rs])))
csvsave(o/'success_rates.csv',summary)
rng=np.random.default_rng(20261025);draw=rng.integers(0,20,(10000,20));stats=[];flips=[]
for left,right in [('D-center-low','D-center'),('C-low','C'),('D-center-low','C-low'),('D-center-low','C')]:
 delta=np.array([[int(index[left,s,c]['success'])-int(index[right,s,c]['success']) for c in conditions] for s in seeds],float)
 for j,c in enumerate(conditions):
  d=delta[:,j];ci=np.quantile(d[draw].mean(1),[.025,.975]);stats.append(dict(comparison=left+' minus '+right,appearance=c,difference=float(d.mean()),CI95=ci.tolist(),wins=int((d>0).sum()),losses=int((d<0).sum())))
  for i,s in enumerate(seeds):flips.append(dict(comparison=left+' minus '+right,seed=s,appearance=c,left_success=index[left,s,c]['success'],right_success=index[right,s,c]['success'],difference=int(d[i])))
 d=delta.mean(1);stats.append(dict(comparison=left+' minus '+right,appearance='equal_mean_three_conditions_state_clustered',difference=float(d.mean()),CI95=np.quantile(d[draw].mean(1),[.025,.975]).tolist()))
 for j in (1,2):
  d=delta[:,j]-delta[:,0];stats.append(dict(comparison=left+' minus '+right,appearance=conditions[j]+' interaction vs clean',difference=float(d.mean()),CI95=np.quantile(d[draw].mean(1),[.025,.975]).tolist()))
save(o/'paired_statistics.json',dict(method='paired bootstrap, physical state resampling shared across appearances, 10000 draws fixed seed20261025',independent_states=20,single_training_seed=True,development_not_final_test=True,small_discrete_sample_limitation=True,comparisons=stats));csvsave(o/'paired_results.csv',flips)
ss={s['seed']:s for s in states()};timeline=[];stage=[]
for n,r in enumerate(new):
 e=env_make(ss[r['seed']],r['condition']);t=np.load(r['trajectory']);initial=metrics(e);trace=[];lift=None
 for j,s in enumerate(t['states']):
  e.set_state(s);e.sim.forward();x=metrics(e);x['step']=j+1;x['close_command']=bool(t['executed_actions'][j,6]>0);x['lift_m']=x['bowl_2']-initial['bowl_2']
  if lift is None and x['lift_m']>=.03:lift=j+1
  trace.append(x)
 for x in trace:timeline.append(dict(model=r['model'],seed=r['seed'],appearance=r['condition'],relative_to_lift=None if lift is None else x['step']-lift,**x))
 posttrace=trace[lift-1:] if lift else [];end=trace[-1]
 stage.append(dict(model=r['model'],seed=r['seed'],appearance=r['condition'],recorded_success=r['success'],recomputed_terminal_success=end['goal'],first_lift=lift,ever_close=any(x['close_command'] for x in trace),final_xy_m=end['xy_distance_m'],final_vertical_m=end['vertical_bowl_plate_m'],simultaneous_conditions_steps=sum(x['goal'] for x in posttrace),first_open_after_lift=next((x['step'] for x in posttrace if not x['close_command']),None),finger_contact_loss_after_lift=next((x['step'] for i,x in enumerate(posttrace[1:],1) if posttrace[i-1]['finger_contact'] and not x['finger_contact']),None)))
 e.close();print('STAGE_ANALYSIS',n+1,flush=True)
csvsave(o/'stage_summary_new.csv',stage);csvsave(o/'stage_timeline_local_only.csv',timeline)
alltimelines=list(csv.DictReader((post/'stage_timeline_local_only.csv').open()))+timeline
groups=collections.defaultdict(list)
for x in alltimelines:groups[x['model'],int(x['seed']),x['appearance']].append(x)
def truth(v):return v is True or v=='True'
stage4=[]
for (m,s,c),rs in groups.items():
 all_close=any(truth(x['close_command']) for x in rs)
 rs=[x for x in rs if x['relative_to_lift'] not in ('',None) and int(x['relative_to_lift'])>=0]
 if not rs:stage4.append(dict(model=m,seed=s,appearance=c,lifted=False,ever_close=all_close));continue
 # Fixed analysis only: descending control intervals after first lift. No deployment feedback.
 descent=[(x,y) for x,y in zip(rs[:-1],rs[1:]) if float(y['bowl_2'])<float(x['bowl_2'])]
 drift=sum(np.hypot(float(y['bowl_0'])-float(x['bowl_0']),float(y['bowl_1'])-float(x['bowl_1'])) for x,y in descent)
 xy=[x for x in rs if truth(x['xy_condition'])];contact=[x for x in rs if truth(x['contact_condition'])];opened=[x for x in rs if not truth(x['close_command'])]
 stage4.append(dict(model=m,seed=s,appearance=c,lifted=True,descent_horizontal_path_m=drift,descent_intervals=len(descent),mean_descent_horizontal_increment_m=drift/len(descent) if descent else None,descent_xy_error_change_m=sum(float(y['xy_distance_m'])-float(x['xy_distance_m']) for x,y in descent),ever_simultaneous=any(truth(x['goal']) for x in rs),final_xy_m=float(rs[-1]['xy_distance_m']),first_xy=int(xy[0]['step']) if xy else None,last_xy=int(xy[-1]['step']) if xy else None,first_plate_contact=int(contact[0]['step']) if contact else None,first_open=int(opened[0]['step']) if opened else None,first_finger_contact_loss=next((int(y['step']) for x,y in zip(rs[:-1],rs[1:]) if truth(x['finger_contact']) and not truth(y['finger_contact'])),None),ever_close=any(truth(x['close_command']) for x in rs),minimum_height_when_centered_m=min(float(x['vertical_bowl_plate_m']) for x in xy) if xy else None))
 stage4[-1]['ever_close']=all_close
csvsave(o/'stage_metrics_four_arms.csv',stage4)
coverage=list(csv.DictReader((post/'demo_phase_coverage.csv').open()));first={r['episode']:int(r['first_lift']) if r['first_lift'] else None for r in coverage};phase=[]
for arm,path in [('C',S/'policy_run/C/isolated_smoke'),('D-center',CENTER/'D-center/isolated_smoke')]+[(m,o/'combined_metrics'/m) for m in ('C-low','D-center-low')]:
 groups=collections.defaultdict(list)
 for r in csv.DictReader((path/'validation_metrics.csv').open()):
  t=int(r['timestep']);ph='postlift' if first[r['episode']] is not None and t>=first[r['episode']] else 'prelift';groups[int(r['step']),r['appearance'],ph].append(r)
 for (step,c,ph),rs in groups.items():phase.append(dict(model=arm,step=step,appearance=c,phase=ph,windows=len(rs),**{k:float(np.mean([float(x[k]) for x in rs])) for k in ('action_L1','translation_L1','rotation_L1','gripper_L1','auxiliary_L1')}))
csvsave(o/'validation_phase_metrics.csv',phase)
save(o/'stage_analysis_limitations.json',dict(new_control_steps=0,recomputed_labels_do_not_replace_live_success=True,cached_pose_integrator_phase_difference='mj_forward of saved states is not identical to post-substep cached predicate timing',descent_rule='negative bowl-z difference between consecutive saved states after original 3cm lift',finger_contact_not_force_or_stable_grasp_truth=True,drop_not_independently_validated=True))
