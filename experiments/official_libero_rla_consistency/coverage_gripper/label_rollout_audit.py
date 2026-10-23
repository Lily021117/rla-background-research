"""CPU audit of labels, actual sampling exposure, and existing rollout states."""
import argparse,csv,json,hashlib,os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,h5py,mujoco
from scipy.spatial.transform import Rotation

def dump(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2)
def csvwrite(p,rows):
    with p.open('x') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def geometry(model,states):
    d=mujoco.MjData(model);site=model.site('gripper0_grip_site').id;hand=model.body('robot0_right_hand').id;bowl=model.body('akita_black_bowl_1_main').id;plate=model.body('plate_1_main').id
    output=[]
    for s in states:
        d.time=s[0];d.qpos[:]=s[1:1+model.nq];d.qvel[:]=s[1+model.nq:];mujoco.mj_forward(model,d)
        target_contact=0;finger_contact=0
        for c in d.contact[:d.ncon]:
            b1,b2=int(model.geom_bodyid[c.geom1]),int(model.geom_bodyid[c.geom2]);other=b2 if b1==bowl else b1 if b2==bowl else None
            if other is not None:
                name=model.body(other).name or ''
                target_contact+=int('robot0' in name or 'gripper' in name or 'finger' in name);finger_contact+=int('finger' in name)
        relative=d.xpos[bowl];R=Rotation.from_quat(np.roll(d.xquat[bowl],-1)).as_matrix()
        output.append(dict(eef=d.site_xpos[site].copy(),bowl=d.xpos[bowl].copy(),plate=d.xpos[plate].copy(),relative=R.T@(d.site_xpos[site]-relative),rotation=R.T@Rotation.from_quat(np.roll(d.xquat[hand],-1)).as_matrix(),distance=float(np.linalg.norm(d.site_xpos[site]-relative)),robot_contact=target_contact,finger_contact=finger_contact))
    return output

def main():
    p=argparse.ArgumentParser();p.add_argument('--previous',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    entries=json.loads((a.previous/'episode_manifest.json').read_text());order=json.loads((a.previous/'formal_B/sampling_order.json').read_text());samples={}
    for e,t in order:samples[e,t]=samples.get((e,t),0)+1
    coverage=[];events=[];phases=[];experts=[];protected=[]
    (a.out/'label_traces').mkdir()
    for e in entries:
        lab=np.load(a.previous/'episodes'/e['episode']/'trajectory_labels.npz');act=lab['actions'];N=len(act);raw=act[:,6];close=raw>0;target=1-np.clip(raw,0,1);starts=np.flatnonzero(~close[:-1]&close[1:])+1;releases=np.flatnonzero(close[:-1]&~close[1:])+1
        count=np.zeros(N,int);possible=np.zeros(N,int)
        for t in range(N-8):
            possible[t:t+8]+=1
            if e['split']=='train':count[t:t+8]+=2*samples.get((e['episode'],t),0)
        assert set(np.unique(raw))<={-1.,1.}
        m=mujoco.MjModel.from_xml_path(str(Path(e['geometry_path']).parent/'clean.xml'));geom=geometry(m,lab['states'])
        with h5py.File(e['source_hdf5'],'r') as f:
            original=f['data'][e['source_demo']]
            np.testing.assert_array_equal(original['actions'][:],act)
            success=bool(np.any(original['rewards'][:]>0)) if 'rewards' in original else None;last_done=bool(original['dones'][-1]) if 'dones' in original else None
        closing=np.zeros(N,bool)
        for t in starts:closing[t:min(t+4,N)]=True
        before=np.arange(N)<(int(starts[0]) if len(starts) else N);lift=np.array([x['bowl'][2]-geom[0]['bowl'][2] for x in geom]);transport=close&(lift>.02);release=np.zeros(N,bool)
        for t in releases:
            end=next((int(s) for s in starts if s>t),N);release[t:end]=True
        for phase,mask in [('pre_first_close_approach_candidate',before),('first4_command_close_steps',closing),('command_closed_and_target_lift_over2cm_transport_candidate',transport),('open_after_close_release_command',release)]:
            observations=sum(1 for t in range(N-8) if mask[t]);actual=sum(2*samples.get((e['episode'],t),0) for t in range(N-8) if mask[t]) if e['split']=='train' else None
            phases.append(dict(episode=e['episode'],split=e['split'],phase=phase,raw_steps=int(mask.sum()),valid_current_observation_windows=observations,supervised_action_slots=int(possible[mask].sum()),actual_training_current_appearance_exposures=actual,actual_training_action_slot_exposures=int(count[mask].sum()) if e['split']=='train' else None,interpretation='command/geometric phase proxy; not a verified grasp label'))
        for t in starts:
            j=int(t);experts.extend(geom[max(0,j-2):min(N,j+3)])
            events.append(dict(episode=e['episode'],split=e['split'],event='open_to_close',timestep=j,raw_action=float(raw[j]),BC_target=float(target[j]),valid_current_window=j<N-8,possible_action_slot_count=int(possible[j]),actual_training_action_slot_exposures=int(count[j]) if e['split']=='train' else None,actual_current_appearance_exposures=2*samples.get((e['episode'],j),0) if e['split']=='train' else None,eef_to_target_center_m=geom[j]['distance'],finger_target_contacts=geom[j]['finger_contact']))
        coverage.append(dict(episode=e['episode'],split=e['split'],source_demo=e['source_demo'],length=N,source_recorded_success=success,last_done=last_done,open_count=int((~close).sum()),close_count=int(close.sum()),open_fraction=float((~close).mean()),close_fraction=float(close.mean()),open_to_close_events=json.dumps(starts.tolist()),close_to_open_events=json.dumps(releases.tolist()),valid_windows=N-8,close_actions_omitted_by_complete_future_filter=int(np.sum(close&(possible==0))),omitted_action_indices=json.dumps(np.flatnonzero(possible==0).tolist()),close_event_action_targets_covered=bool(np.all(possible[starts]>0)),actual_sampled_physical_windows=sum(v for (ep,t),v in samples.items() if ep==e['episode']) if e['split']=='train' else 0,actual_close_BC_slot_exposures=int(count[close].sum()) if e['split']=='train' else None))
        np.savez_compressed(a.out/'label_traces'/f"{e['episode']}.npz",raw_gripper=raw,BC_gripper=target,close_events=starts,release_events=releases,action_slot_exposures=count,eef_to_target=[x['distance'] for x in geom],target_lift=lift)
    csvwrite(a.out/'gripper_label_coverage.csv',coverage);csvwrite(a.out/'gripper_events.csv',events);csvwrite(a.out/'phase_window_coverage.csv',phases)
    rollout=[];expert_positions=np.array([x['relative'] for x in experts]);expert_rotations=np.array([x['rotation'] for x in experts])
    dev=next(e for e in entries if e['split']=='validation');m=mujoco.MjModel.from_xml_path(str(Path(dev['geometry_path']).parent/'clean.xml'));initial=np.load(a.previous/'episodes'/dev['episode']/'trajectory_labels.npz')['states'][0]
    for condition in ('clean','background'):
        with np.load(a.previous/'development'/condition/'trajectory.npz') as d:
            chunks=d['predicted_chunks'];actual=d['executed_actions'];states=d['states'];pred=chunks.reshape(-1,7)[:len(actual)].copy();expected=pred.copy();expected[:,6]=-np.sign(2*expected[:,6]-1)
            np.testing.assert_array_equal(actual,expected)
            before=geometry(m,np.concatenate([initial[None],states[:-1]]));after=geometry(m,states);relative=np.array([x['relative'] for x in before]);pos_errors=np.linalg.norm(relative[:,None]-expert_positions[None],axis=2);closest=pos_errors.argmin(1)
            angles=[float(Rotation.from_matrix(expert_rotations[j].T@x['rotation']).magnitude()) for x,j in zip(before,closest)]
            transitions=[]
            for t,(b,post) in enumerate(zip(before,after)):
                transitions.append(dict(appearance=condition,step=t,query_index=t//8,chunk_position=t%8,predicted_gripper=float(pred[t,6]),actual_gripper=float(actual[t,6]),eef_to_target_pre_m=b['distance'],nearest_expert_close_relative_position_m=float(pos_errors[t,closest[t]]),orientation_error_to_position_nearest_expert_close_rad=angles[t],finger_target_contact_post=post['finger_contact'],robot_target_contact_post=post['robot_contact']))
            csvwrite(a.out/f'rollout_alignment_{condition}.csv',transitions)
            xyz=np.array([x['eef'] for x in after]);motion=np.linalg.norm(np.diff(np.concatenate([before[0]['eef'][None],xyz]),axis=0),axis=1);near=int(np.argmin(pos_errors.min(1)));query=np.arange(0,len(actual),8);close_queries=np.sum(chunks[:,:,6]<.5,axis=1)
            rollout.append(dict(appearance=condition,commands_match_model_postprocessing=True,executed_actions=len(actual),chunks=len(chunks),expected_chunks=int(np.ceil(len(actual)/8)),executed_last_chunk=len(actual)%8 or 8,close_commands=int(np.sum(actual[:,6]==1)),open_commands=int(np.sum(actual[:,6]==-1)),neutral_commands=int(np.sum(actual[:,6]==0)),translation_saturation_fraction=float(np.mean(abs(actual[:,:3])>=.999)),rotation_saturation_fraction=float(np.mean(abs(actual[:,3:6])>=.999)),fraction_sub0_1mm_eef_steps=float(np.mean(motion<1e-4)),minimum_eef_to_target_center_m=min(x['distance'] for x in before),nearest_expert_close_relative_position_m=float(pos_errors[near,closest[near]]),orientation_error_at_that_step_rad=angles[near],nearest_expert_step=near,gripper_command_at_nearest_expert_step=float(actual[near,6]),finger_target_contact_steps=sum(x['finger_contact']>0 for x in after),robot_target_contact_steps=sum(x['robot_contact']>0 for x in after),query_close_slots=close_queries.tolist(),error_caveat='nearest expert position does not prove a correct grasp pose; orientation and contact are reported separately'))
    dump(a.out/'rollout_quantitative_analysis.json',rollout)
    # Endpoint and exact threshold checks use the actual published formula.
    grid=np.array([-1.,0.,.499,.5,.501,1.,2.]);commands=-np.sign(2*grid-1)
    dump(a.out/'gripper_pipeline_numeric_audit.json',dict(raw_values=[-1,1],BC_targets=[1,0],normalization_mask_last=False,deployment_predictions=grid.tolist(),commands=commands.tolist(),threshold='p<0.5 close(+1), p>0.5 open(-1), p==0.5 neutral(0)',BC_loss='unmasked mean L1 over all 8x7 valid action entries; gripper included with 1/7 dimension share',proprio_gripper='two finger qpos, separate input; never used as action target',rollout_command_roundtrip='exact equality for all 440 existing commands',optimizer_updates=0,backward=0,new_closed_loop=0))
    print(json.dumps(dict(label_rows=coverage,rollout=rollout),indent=2))
if __name__=='__main__':main()
