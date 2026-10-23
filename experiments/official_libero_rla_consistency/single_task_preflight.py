"""CPU-only admission audit. Never opens unassigned or held-out episode contents."""
import argparse, csv, hashlib, json, os, subprocess
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def fk(m, d):
    site = m.site('gripper0_grip_site').id
    hand = m.body('robot0_right_hand').id
    fingers = [m.joint('gripper0_finger_joint'+str(i)).qposadr[0] for i in (1, 2)]
    return np.r_[d.site_xpos[site], Rotation.from_quat(np.roll(d.xquat[hand], -1)).as_rotvec(), d.qpos[fingers]]

def main():
    p = argparse.ArgumentParser(); p.add_argument('--project', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    def save(name, obj):
        with (a.out/name).open('x') as f: json.dump(obj, f, indent=2)
    split_path = a.project/'experiments/rla_background_invariance/step2_training/split_manifest.json'
    old_path = a.project/'artifacts/action_conditioned_wm/TZ_three_seed_500/independent_data/selection.json'
    split = json.loads(split_path.read_text())['splits']
    first = split['train'][0]; task = first['task']
    historical = json.loads(old_path.read_text())['demonstrations']
    prior = [x for x in historical if x['id'] == first['episode_id']]
    assert len(prior) == 1
    selected = [(role,e) for role in ('train','validation') for e in split[role] if e['task'] == task]
    protected = {(e['source_hdf5'], e['source_demo']) for e in split['test']}
    sources = {str(split_path):sha(split_path), str(old_path):sha(old_path)}
    rows=[]; audits=[]; actions=[]; proprios=[]; identities=set()
    for role,e in selected:
        assert (e['source_hdf5'],e['source_demo']) not in protected
        pair = Path(e['geometry_path']).parent
        for name in ('clean.xml','background.xml'): sources[str(pair/name)] = sha(pair/name)
        with h5py.File(e['source_hdf5'],'r') as f:
            demo=f['data'][e['source_demo']]
            states=demo['states'][:]; act=demo['actions'][:]
            assert all(k in demo['obs'] for k in ('agentview_rgb','eye_in_hand_rgb','joint_states','gripper_states'))
        identity=hashlib.sha256(act.astype(np.float32).tobytes()).hexdigest()
        assert identity == e['actions_sha256'] and identity not in identities
        identities.add(identity)
        assert len(states)==len(act) and np.isfinite(states).all() and np.isfinite(act).all()
        assert np.allclose(np.diff(states[:,0]),.05,rtol=0,atol=1e-9)
        m=mujoco.MjModel.from_xml_path(str(pair/'clean.xml'))
        d=mujoco.MjData(m); robot=mujoco.MjData(m)
        names=['robot0_joint'+str(i) for i in range(1,8)]+['gripper0_finger_joint1','gripper0_finger_joint2']
        addresses=[int(m.joint(n).qposadr[0]) for n in names]
        props=[]; errors=[]
        for s in states:
            assert len(s)==1+m.nq+m.nv and m.na==0
            d.time=s[0]; d.qpos[:]=s[1:1+m.nq]; d.qvel[:]=s[1+m.nq:]; mujoco.mj_forward(m,d)
            # Deployment reconstruction uses only nine robot joint encoders.
            # All task-object states stay at the XML defaults in this second model state.
            mujoco.mj_resetData(m,robot); robot.qpos[addresses]=d.qpos[addresses]; mujoco.mj_forward(m,robot)
            x,y=fk(m,d),fk(m,robot); errors.append(float(np.max(abs(x-y))))
            np.testing.assert_allclose(x,y,rtol=0,atol=1e-12); props.append(x)
        props=np.array(props,dtype=np.float32)
        if role=='train':
            standardized=act.copy(); standardized[:,-1]=1-np.clip(act[:,-1],0,1)
            actions.append(standardized); proprios.append(props)
        rows.append(dict(episode=e['episode_id'],split=role,source_hdf5=e['source_hdf5'],source_demo=e['source_demo'],task=task,frames=len(states),complete_h8_chunks=len(states)-8,appearance_variants=2,source_action_sha256=identity,derived_version='single_task_dual224_fk_v3',render_status='PENDING',geometry_path=e['geometry_path']))
        audits.append(dict(episode=e['episode_id'],states=len(states),robot_joint_names=names,qpos_addresses=addresses,max_fk_difference=max(errors),task_object_state_used_by_deployment=False,external_pose_ground_truth=False,raw_sensor_phase='UNRESOLVED; raw proprio not used'))
    stats={}
    for name,arrays in [('action',actions),('proprio',proprios)]:
        x=np.concatenate(arrays).astype(np.float32)
        stats[name]={k:np.asarray(v).tolist() for k,v in dict(mean=x.mean(0),std=x.std(0),min=x.min(0),max=x.max(0),q01=np.quantile(x,.01,axis=0),q99=np.quantile(x,.99,axis=0)).items()}
    stats['action']['mask']=[True]*6+[False]
    save('normalization_statistics_train_only.json',stats)
    save('deployment_fk_audit.json',dict(status='PASS_ROBOT_ENCODER_FK',episodes=audits,limitation='Requires calibrated fixed robot base and the matched kinematic XML; not independent external pose accuracy validation.'))
    save('episode_manifest.json',rows)
    save('split_provenance.json',dict(source_hashes=sources,selected_task=task,selected_episode=first['episode_id'],current_role='train',historical_role=prior[0]['split'],independent_test_eligible=False,reason='Previously inspected historical independent evaluation, subsequently explicitly assigned to Step2 training.',train_episodes=sum(x['split']=='train' for x in rows),development_episodes=sum(x['split']=='validation' for x in rows),requested_train=20,requested_development=5,additional_hdf5_episodes='UNASSIGNED: not admitted merely because files exist',heldout_contents_read=False))
    assert all(sha(k)==v for k,v in sources.items())
    save('delivery_verification.json',dict(status='PARTIAL_CPU_PREFLIGHT',optimizer_updates=0,backward=0,new_closed_loop=0,gpu_used=False,source_hashes_unchanged=True,rendering_complete=False,training_started=False))
    print(json.dumps(dict(episodes=len(rows),frames=sum(x['frames'] for x in rows),chunks=sum(x['complete_h8_chunks'] for x in rows),status='CPU preflight passed; render/loader/GPU smoke remain'),indent=2))

if __name__ == '__main__': main()
