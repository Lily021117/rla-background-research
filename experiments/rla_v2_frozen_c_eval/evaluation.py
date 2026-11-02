"""Frozen historical policy inference with V2 XML and current online observations."""
import os,ast,inspect,random,time,traceback
from collections import deque
import numpy as np
import torch,mujoco,imageio
from PIL import Image
from scipy.spatial.transform import Rotation
from rla_v2_frozen_c_eval.common import *
from rla_scratch_policy_aux import evaluation as historical

PHYS=['body_mass','body_inertia','body_pos','body_quat','geom_pos','geom_quat','geom_size','geom_type','geom_friction','geom_contype','geom_conaffinity','geom_solref','geom_solimp','jnt_type','jnt_pos','jnt_axis','jnt_range','dof_damping','dof_armature','actuator_gainprm','actuator_biasprm','cam_pos','cam_quat','cam_fovy','mesh_vert','mesh_face']
VIS=['mat_rgba','mat_texid','mat_reflectance','mat_specular','mat_shininess','geom_matid','light_pos','light_dir','light_diffuse','light_ambient','light_specular']
def signature(m):return {k:digest(np.array(getattr(m,k))) for k in VIS}
def technical(task):
 """Actual ControlEnv reset inspection, no policy calls or simulation steps."""
 from libero.libero.envs.env_wrapper import ControlEnv
 from libero.libero import get_libero_path
 s=next(s for s in read(ROOT/'state_manifest.json') if s['task']==task);a=np.load(s['state_file']);records=[];baseline=None
 for condition,xml in s['xmls'].items():
  random.seed(s['seed']);np.random.seed(s['seed']);env=ControlEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/s['bddl']),use_camera_obs=False,has_offscreen_renderer=False,has_renderer=False,control_freq=20,horizon=1000,initialization_noise=None,ignore_done=False)
  try:
   env.seed(s['seed']);env.reset();env.reset_from_xml_string(Path(xml).read_text());env.set_init_state(a);env.sim.forward();np.testing.assert_allclose(env.get_sim_state(),a,atol=1e-8,rtol=0);assert not env.check_success()
   m,d=env.sim.model._model,env.sim.data._data;expected=mujoco.MjModel.from_xml_path(xml)
   for k in PHYS+VIS+['tex_data']:assert np.array_equal(getattr(m,k),getattr(expected,k)),(condition,k)
   if baseline is None:baseline={k:np.array(getattr(m,k)).copy() for k in PHYS}
   else:
    for k in PHYS:assert np.array_equal(getattr(m,k),baseline[k]),k
   records.append(dict(condition=condition,expected_visual_signature=signature(m),physics_equal=True,actual_ControlEnv_reset_preserves_V2=True,initial_success=False))
  finally:env.close()
 save(ROOT/task/'runtime_integration.json',dict(status='PASS',records=records,policy_forwards=0,policy_rollouts=0,steps=0))

def execute(g):
 from libero.libero.envs.env_wrapper import ControlEnv
 from libero.libero import get_libero_path
 from experiments.robot.libero.run_libero_eval import process_action
 from stove_postlift.common import metrics as task_metrics
 task=os.environ['V2_EVAL_TASK'];protocol=read(ROOT/'evaluation_protocol.json');commit=read(ROOT/'protocol_commit.json');assert commit['protocol_sha256']==sha(ROOT/'evaluation_protocol.json')
 technical(task)
 models=[r for r in read(ROOT/'checkpoint_inventory.json') if r['task']==task];models.sort(key=lambda r:r['policy_seed']);stateset=[s for s in read(ROOT/'state_manifest.json') if s['task']==task]
 model,prop,head=g['model'],g['prop'],g['head'];mods=dict(vla=model,proprio=prop,head=head);expected={k:{n for n,p in m.named_parameters() if p.requires_grad} for k,m in mods.items()};weights={}
 for entry in models:
  assert sha(entry['path'])==entry['sha256'];ck=torch.load(entry['path'],map_location='cpu');assert ck['global_step']==1800
  for k in mods:assert set(ck['parameters'][k])==expected[k],(k,'parameter coverage differs')
  weights[entry['model_id']]=ck['parameters'];del ck
 stats=read(models[0]['statistics']);assert all(r['statistics_sha256']==sha(r['statistics']) for r in models);model.base_model.model.norm_stats={'derived':stats}
 language=stateset[0]['language'];src=inspect.getsource(historical.execute);tree=ast.parse(src);selected=[n for n in tree.body[0].body if isinstance(n,ast.FunctionDef) and n.name in ['infer','proprio']];assert len(selected)==2
 text='\n\n'.join(ast.get_source_segment(src,n) for n in selected);scope=dict(globals(),g=g,model=model,prop=prop,stats=stats,language=language);exec(text,scope);infer=scope['infer'];proprio=scope['proprio']
 save(ROOT/task/'inference_identity.json',dict(source=str(Path(inspect.getfile(historical))),source_sha256=sha(inspect.getfile(historical)),nested_infer_proprio_sha256=__import__('hashlib').sha256(text.encode()).hexdigest(),unchanged=True,checkpoint_full_trainable_keys_verified=True,training=False,AE_loaded=False))
 out=ROOT/task/'closed_loop';out.mkdir(exist_ok=False);rows=[];forwards=0
 for s in stateset:
  seed=s['seed'];initial=np.load(s['state_file']);assert digest(initial)==s['state_sha256']
  for condition in CONDITIONS:
   xml=s['xmls'][condition];assert sha(xml)==s['xml_sha256'][condition]
   reference=mujoco.MjModel.from_xml_path(xml);expected_signature=signature(reference);expected_texture=digest(reference.tex_data);del reference
   for entry in models:
    key=entry['model_id'];dest=out/f'{seed}_{condition}_{key}';dest.mkdir();env=renderer=video=None;actions_log=[];chunks=[];trajectory=[];eef=[];target=[];timeline=[];started=time.time()
    result=dict(task=task,model=key,policy_seed=entry['policy_seed'],seed=seed,condition=condition,checkpoint_sha256=entry['sha256'],state_sha256=s['state_sha256'],xml_sha256=s['xml_sha256'][condition],language=language,status='INVALID',success=None,steps=0,policy_forward_attempts=0,trajectory=str(dest/'trajectory.npz'),video=str(dest/'video.mp4'),reused=False)
    try:
     for k,m in mods.items():
      full=m.state_dict();full.update(weights[key][k]);m.load_state_dict(full,strict=True);m.eval().requires_grad_(False)
     random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
     env=ControlEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/s['bddl']),use_camera_obs=False,has_offscreen_renderer=False,has_renderer=False,control_freq=20,horizon=1000,initialization_noise=None,ignore_done=False);env.seed(seed);env.reset();env.reset_from_xml_string(Path(xml).read_text());env.set_init_state(initial);env.sim.forward();np.testing.assert_allclose(env.get_sim_state(),initial,rtol=0,atol=1e-8)
     m,d=env.sim.model._model,env.sim.data._data;assert signature(m)==expected_signature and digest(m.tex_data)==expected_texture;assert not env.check_success() and m.vis.quality.offsamples==4
     if task=='stove':bid=m.body('akita_black_bowl_1_main').id;initial_target=d.xpos[bid].copy()
     else:joint=m.joint('wooden_cabinet_1_middle_level').qposadr[0];q0=float(d.qpos[joint])
     renderer=mujoco.Renderer(m,224,224);opt=mujoco.MjvOption();opt.geomgroup[0]=0;opt.geomgroup[1]=1;opt.sitegroup[:]=0;fk=mujoco.MjData(m);addresses=[m.joint(n).qposadr[0] for n in ['robot0_joint'+str(i) for i in range(1,8)]+['gripper0_finger_joint1','gripper0_finger_joint2']];queue=deque();video=imageio.get_writer(dest/'video.mp4',fps=2.5)
     for step in range(protocol['max_steps'][task]):
      if not queue:
       assert g['counts'](g['before'])==g['counts'](g['ecc']());assert signature(m)==expected_signature;env.sim.forward();images=[]
       for camera in ['agentview','robot0_eye_in_hand']:
        renderer.update_scene(d,camera=camera,scene_option=opt);renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=1;images.append(renderer.render()[:,::-1].copy())
       mujoco.mj_resetData(m,fk);fk.qpos[addresses]=d.qpos[addresses];mujoco.mj_forward(m,fk);p=proprio(m,fk);np.testing.assert_array_equal(p,proprio(m,d));video.append_data(np.concatenate(images,axis=1));result['policy_forward_attempts']+=1;forwards+=1;chunk=infer(images,p,head);chunks.append(chunk.copy());queue.extend(chunk)
      action=process_action(queue.popleft().copy(),'openvla');assert np.isfinite(action).all();_,_,done,_=env.step(action.tolist());actions_log.append(action);trajectory.append(env.get_sim_state().copy());eef.append(d.site_xpos[m.site('gripper0_grip_site').id].copy());result['steps']=step+1
      if task=='stove':target.append(d.xpos[bid].copy());timeline.append(dict(step=step+1,close_command=bool(action[6]>0),**task_metrics(env)))
      else:timeline.append(dict(step=step+1,middle_joint_m=float(d.qpos[joint]),goal=bool(env.check_success()),close_command=bool(action[6]>0)))
      if env.check_success() or done:break
     success=bool(env.check_success());assert signature(m)==expected_signature and digest(m.tex_data)==expected_texture
     result.update(status='COMPLETE',success=success,ending='success' if success else 'timeout' if result['steps']==protocol['max_steps'][task] else 'environment_done',visual_configuration_stable=True,translation_saturation_fraction=float(np.mean(abs(np.asarray(actions_log)[:,:3])>=.999)),issued_close_command=bool(np.any(np.asarray(actions_log)[:,6]>0)))
     if task=='stove':
      lift=float(np.max(np.array(target)[:,2]-initial_target[2]));near=float(np.linalg.norm(np.array(eef)-np.array(target),axis=1).min());result.update(max_lift_m=lift,min_ee_target_distance_m=near,stage='success' if success else 'post_lift_or_placement_failure' if lift>=.03 else 'near_target_grasp_or_contact_failure' if near<=.10 else 'approach_failure',final_xy_m=timeline[-1]['xy_distance_m'],final_vertical_m=timeline[-1]['vertical_bowl_plate_m'])
     else:
      progress=q0-min(t['middle_joint_m'] for t in timeline);result.update(max_opening_progress_m=progress,final_middle_joint_m=timeline[-1]['middle_joint_m'],stage='success' if success else 'partial_opening' if progress>=.01 else 'no_opening_progress')
    except Exception:result['error']=traceback.format_exc()
    finally:
     if video is not None:video.close()
     if renderer is not None:renderer.close()
     if env is not None:env.close()
     np.savez_compressed(dest/'trajectory.npz',executed_actions=actions_log,predicted_chunks=chunks,states=trajectory,eef_positions=eef,target_bowl_positions=target)
     save(dest/'phase_timeline.json',timeline)
    result['seconds']=time.time()-started;save(dest/'result.json',result);rows.append(result)
    (ROOT/task/'per_trial_results.json').write_text(json.dumps(rows,indent=2));print('TRIAL',task,len(rows),seed,condition,entry['policy_seed'],result['status'],result['success'],flush=True)
    assert result['status']=='COMPLETE','Infrastructure failure preserved; stop without automatic retry'
 assert len(rows)==240
 save(ROOT/task/'completion.json',dict(valid=240,actual_attempts=240,invalid=0,policy_forwards=forwards,optimizer_updates=0,AE_updates=0,DINO_updates=0,backward=0,peak_gpu_bytes=torch.cuda.max_memory_allocated()))
