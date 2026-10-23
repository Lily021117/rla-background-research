"""Exactly two frozen development rollouts; no checkpoint selection or retries."""
import os,json,csv,hashlib,traceback,time,random
from pathlib import Path
from collections import deque
import numpy as np
import torch
import mujoco
import imageio
from PIL import Image
from scipy.spatial.transform import Rotation

def execute(g):
    from libero.libero.envs.env_wrapper import ControlEnv
    from libero.libero import get_libero_path
    from experiments.robot.libero.run_libero_eval import process_action
    a,o,model,head,prop=g['a'],g['o'],g['model'],g['head'],g['prop']
    plan=json.loads((a.run/'training_plan.json').read_text());ckpath=a.run/'formal_B'/f"step{plan['closed_loop_checkpoint']:04}.pt"
    ck=torch.load(ckpath,map_location='cpu');assert ck['global_step']==270 and ck['scope']=='FORMAL_B_BASELINE'
    for k,m in g['mods'].items():
        expected={n for n,x in m.named_parameters() if x.requires_grad};assert expected==set(ck['parameters'][k])
        full=m.state_dict();full.update(ck['parameters'][k]);m.load_state_dict(full,strict=True);m.eval().requires_grad_(False)
    del ck
    stats=json.loads((a.run/'normalization_statistics_train_only.json').read_text());model.base_model.model.norm_stats={'derived':stats}
    rows=json.loads((a.run/'episode_manifest.json').read_text());rows=[x for x in rows if x['episode'] in plan['closed_loop_episodes']];assert len(rows)==1 and rows[0]['split']=='validation';e=rows[0]
    root=a.run/'episodes'/e['episode'];labels=np.load(root/'trajectory_labels.npz');initial=labels['states'][0].copy();pair=Path(e['geometry_path']).parent
    bddl=Path(get_libero_path('bddl_files'))/'libero_spatial'/('pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate.bddl');assert bddl.exists()
    g['save']('evaluation_protocol.json',dict(checkpoint=str(ckpath),global_step=270,episode=e['episode'],source_demo=e['source_demo'],state_index=0,seed=0,max_control_steps=220,action_chunk_execution=8,conditions=['clean','background'],settling_actions=0,reflection=1,MSAA=4,role='previously inspected development',retries=0,success_source='env.check_success()',object_truth_policy_input=False))
    def proprio(m,d):
        return np.r_[d.site_xpos[m.site('gripper0_grip_site').id],Rotation.from_quat(np.roll(d.xquat[m.body('robot0_right_hand').id],-1)).as_rotvec(),[d.qpos[m.joint('gripper0_finger_joint'+str(i)).qposadr[0]] for i in (1,2)]].astype(np.float32)
    def infer(images,p):
        prompt=g['QwenPromptBuilder']('openvla');prompt.add_turn('human',f"What action should the robot take to {e['task']}?");prompt.add_turn('gpt','')
        tokens=g['tokenizer'](prompt.get_prompt(),add_special_tokens=True).input_ids
        # Exact official miniVLM training prefix; remove the assistant terminator.
        del tokens[-3:]
        inp=dict(input_ids=torch.tensor([tokens],device='cuda'),attention_mask=torch.ones((1,len(tokens)),dtype=torch.bool,device='cuda'),pixel_values=torch.cat([g['imageproc'](Image.fromarray(im),return_tensors='pt')['pixel_values'] for im in images],dim=1).to('cuda',dtype=torch.bfloat16))
        s=stats['proprio'];lo=np.array(s['q01']);hi=np.array(s['q99']);p=np.clip(2*(p-lo)/(hi-lo+1e-8)-1,-1,1);p=np.where(np.array(s['min'])==np.array(s['max']),0,p)
        with torch.inference_mode():actions,_=model.predict_action(**inp,unnorm_key='derived',proprio=p,proprio_projector=prop,action_head=head,use_film=False)
        actions=np.asarray(actions);assert actions.shape==(8,7) and np.isfinite(actions).all();return actions
    results=[];baseline_geometry=None
    for condition in ('clean','background'):
        dest=o/condition;dest.mkdir();env=render=video=None;executed=[];chunks=[];states=[];positions=[];bowl_positions=[];start=time.time();result=dict(appearance=condition,episode=e['episode'],status='INVALID',success=None,steps=0)
        try:
            random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
            env=ControlEnv(bddl_file_name=str(bddl),use_camera_obs=False,has_offscreen_renderer=False,has_renderer=False,control_freq=20,horizon=1000,ignore_done=False)
            env.seed(0);env.reset();env.reset_from_xml_string((pair/(condition+'.xml')).read_text());env.set_init_state(initial);env.sim.forward()
            np.testing.assert_allclose(env.get_sim_state(),initial,atol=1e-8,rtol=0)
            m,d=env.sim.model._model,env.sim.data._data; geometry={k:np.array(getattr(d,k)).copy() for k in ('qpos','qvel','xpos','xquat','cam_xpos','cam_xmat')}
            bowl_id=m.body('akita_black_bowl_1_main').id;initial_bowl=d.xpos[bowl_id].copy()
            if baseline_geometry is None:baseline_geometry=geometry
            else:
                for k in geometry:np.testing.assert_allclose(geometry[k],baseline_geometry[k],atol=1e-8,rtol=0)
            render=mujoco.Renderer(m,224,224);opt=mujoco.MjvOption();opt.geomgroup[0]=0;opt.geomgroup[1]=1;opt.sitegroup[:]=0
            robot_fk=mujoco.MjData(m);joint_names=['robot0_joint'+str(i) for i in range(1,8)]+['gripper0_finger_joint1','gripper0_finger_joint2'];robot_addresses=[m.joint(n).qposadr[0] for n in joint_names]
            q=deque();video=imageio.get_writer(dest/'video.mp4',fps=2.5);assert not env.check_success()
            for step in range(220):
                if not q:
                    assert g['counts'](g['before'])==g['counts'](g['ecc']())
                    env.sim.forward();images=[]
                    for camera in ('agentview','robot0_eye_in_hand'):
                        render.update_scene(d,camera=camera,scene_option=opt);render.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=1;assert m.vis.quality.offsamples==4;images.append(render.render()[:,::-1].copy())
                    mujoco.mj_resetData(m,robot_fk);robot_fk.qpos[robot_addresses]=d.qpos[robot_addresses];mujoco.mj_forward(m,robot_fk)
                    p=proprio(m,robot_fk);np.testing.assert_array_equal(p,proprio(m,d))
                    if step==0:
                        with np.load(root/'frames/0000.npz') as original:
                            np.testing.assert_array_equal(np.stack(images).transpose(0,3,1,2),original[condition]);np.testing.assert_array_equal(p,original['proprio'])
                    video.append_data(np.concatenate(images,axis=1));actions=infer(images,p);chunks.append(actions.copy());q.extend(actions)
                action=process_action(q.popleft().copy(),'openvla');assert np.isfinite(action).all()
                _,_,done,_=env.step(action.tolist());executed.append(action);states.append(env.get_sim_state().copy());positions.append(d.site_xpos[m.site('gripper0_grip_site').id].copy());result['steps']=step+1
                bowl_positions.append(d.xpos[bowl_id].copy())
                if env.check_success() or done:break
            result.update(status='COMPLETE',success=bool(env.check_success()),timeout=result['steps']==220 and not env.check_success(),mean_abs_action=float(np.mean(abs(np.array(executed)))),translation_saturation_fraction=float(np.mean(abs(np.array(executed)[:,:3])>=.999)),gripper_values=np.unique(np.array(executed)[:,-1]).tolist(),eef_path_length_m=float(np.linalg.norm(np.diff(np.array(positions),axis=0),axis=1).sum()))
            result.update(min_eef_to_target_center_m=float(np.linalg.norm(np.array(positions)-np.array(bowl_positions),axis=1).min()),target_max_lift_m=float(np.max(np.array(bowl_positions)[:,2]-initial_bowl[2])),target_final_displacement_m=float(np.linalg.norm(bowl_positions[-1]-initial_bowl)),target_identity_evidence='task BDDL goal: On akita_black_bowl_1 plate_1; diagnostics only, never policy input')
        except Exception:result['error']=traceback.format_exc()
        finally:
            if video:video.close()
            if render:render.close()
            if env:env.close()
            np.savez_compressed(dest/'trajectory.npz',executed_actions=executed,predicted_chunks=chunks,states=states,eef_positions=positions,target_bowl_positions=bowl_positions)
        result['seconds']=time.time()-start
        with (dest/'result.json').open('x') as f:json.dump(result,f,indent=2)
        results.append(result);print('DEVELOPMENT',condition,result['status'],result['success'],flush=True)
    g['save']('results.json',dict(trials=results,optimizer_updates=0,backward=0,closed_loop_attempts=2,heldout_read=False))
