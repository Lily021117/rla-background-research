"""Read-only source admission: no policy, dynamics step, or target extraction.

Missing source-model identity is not filled with an unrelated task XML.
The local task reset is a recovery candidate, not claimed source provenance.
"""
import argparse, csv, hashlib, io, json
from pathlib import Path
import h5py
import numpy as np
from PIL import Image

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def save(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2)
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--bddl',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    with h5py.File(a.source,'r') as f:
        g=f['data'];ids=sorted(g,key=lambda x:int(x.rsplit('_',1)[1]));assert len(ids)>=25
        # Freeze before bulk content checks. Never choose episodes by policy performance.
        save(a.out/'candidate_split.json',dict(status='CANDIDATE_NOT_ADMITTED',task='libero_goal/open_the_middle_drawer_of_the_cabinet',source=str(a.source),source_sha256=sha(a.source),train=ids[:20],development=ids[20:25],rule='numeric episode ID; first20 then5; no result-based replacement',admission='required arrays finite and complete; physical times and source model must be traceable; missing evidence is unresolved, not silently repaired',remaining_candidates=ids[25:]))
        rows=[]
        for name in ids:
            d=g[name];n=len(d['actions']);selected=name in ids[:25]
            row=dict(episode=name,split='train' if name in ids[:20] else 'development' if selected else 'unused',frames=n,states_shape=str(d['states'].shape),actions_shape=str(d['actions'].shape),agentview='agentview_rgb_jpeg' in d['obs'],wrist='eye_in_hand_rgb_jpeg' in d['obs'],source_model_file='model_file' in d.attrs,source_env_config=bool(g.attrs or f.attrs),content_read=selected)
            if selected:
                s=d['states'][:];u=d['actions'][:];j=d['obs/joint_states'][:]
                row.update(finite=bool(np.isfinite(s).all() and np.isfinite(u).all()),first_state_time=float(s[0,0]),second_state_time=float(s[1,0]),non_20hz_intervals=int(np.count_nonzero(abs(np.diff(s[:,0])-.05)>1e-9)),tail_20hz=bool(np.allclose(np.diff(s[1:,0]),.05,atol=1e-9,rtol=0)),first_state_vs_recorded_joint_max=float(abs(s[0,1:8]-j[0]).max()),later_state_vs_sensor_joint_max=float(abs(s[1:,1:8]-j[1:]).max()),states_sha256=hashlib.sha256(s.tobytes()).hexdigest(),actions_sha256=hashlib.sha256(u.tobytes()).hexdigest(),status='UNRESOLVED_SOURCE_MODEL_AND_FIRST_FRAME')
                for camera in ['agentview_rgb_jpeg','eye_in_hand_rgb_jpeg']:
                    image=np.array(Image.open(io.BytesIO(d['obs'][camera][0].tobytes())))
                    row[camera+'_shape']=str(image.shape)
            else:row['status']='NOT_SELECTED_METADATA_ONLY'
            rows.append(row)
        keys=sorted(set().union(*(r.keys() for r in rows)))
        with (a.out/'trajectory_availability.csv').open('x') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
        save(a.out/'source_summary.json',dict(available_episodes=len(ids),candidate_train=20,candidate_development=5,with_per_episode_model=sum(r['source_model_file'] for r in rows),with_global_environment_metadata=bool(g.attrs),selected_nonuniform_first_interval=sum(r.get('non_20hz_intervals',0)>0 for r in rows),policy_updates=0,AE_updates=0,model_forward=0,policy_rollouts=0,BDDL_sha256=sha(a.bddl)))

    # Test only whether a current reset model is a plausible recovery candidate.
    # Do not advance physics to invent the missing first saved state.
    import mujoco
    from libero.libero.envs.env_wrapper import ControlEnv
    env=ControlEnv(bddl_file_name=str(a.bddl),use_camera_obs=False,has_offscreen_renderer=False,has_renderer=False,control_freq=20,horizon=1000,initialization_noise=None)
    try:
        env.seed(0);env.reset();m,d=env.sim.model._model,env.sim.data._data
        xml=env.sim.model.get_xml();(a.out/'candidate_reset_model.xml').write_text(xml)
        checks=[];renderer=mujoco.Renderer(m,256,256);opt=mujoco.MjvOption();opt.geomgroup[0]=0;opt.geomgroup[1]=1;opt.sitegroup[:]=0
        with h5py.File(a.source,'r') as f:
            demo=f['data'][ids[0]]
            for t in [0,1,10]:
                s=demo['states'][t];env.set_init_state(s);env.sim.forward()
                row=dict(timestep=t,state_size=len(s),model_state_size=1+m.nq+m.nv,initial_or_current_success=bool(env.check_success()),eef_max_component_difference=float(abs(d.site_xpos[m.site('gripper0_grip_site').id]-demo['obs/ee_pos'][t]).max()),cameras=[])
                for cam,key in [('agentview','agentview_rgb_jpeg'),('robot0_eye_in_hand','eye_in_hand_rgb_jpeg')]:
                    source=np.array(Image.open(io.BytesIO(demo['obs'][key][t].tobytes())))
                    renderer.update_scene(d,camera=cam,scene_option=opt);render=renderer.render().copy()
                    # Source code stores flipud(OpenGL bottom-up), equal to top-down renderer.
                    row['cameras'].append(dict(camera=cam,source_vs_candidate_MAE=float(abs(source.astype(float)-render).mean()),source_vs_candidate_max=int(abs(source.astype(int)-render).max())))
                    Image.fromarray(np.concatenate([source,render],axis=1)).save(a.out/f'{ids[0]}_t{t}_{cam}_source_candidate.png')
                checks.append(row)
        renderer.close()
        save(a.out/'recovery_candidate_audit.json',dict(status='NOT_SOURCE_MODEL_IDENTITY_PROOF',checks=checks,model_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),simulation_steps=0,policy_calls=0,scope='current local task model; matching shape or approximate pixels cannot certify original generation version; JPEG and engine differences affect RGB'))
    finally:env.close()
if __name__=='__main__':main()
