"""New-task development: six final policies, Clean gate, then all four backgrounds."""
import os,json,time,random,traceback,hashlib,csv
from pathlib import Path
from collections import deque
import numpy as np
import torch,mujoco,imageio
from PIL import Image
from scipy.spatial.transform import Rotation

def execute(g):
 from libero.libero.envs.env_wrapper import ControlEnv
 from libero.libero import get_libero_path
 from experiments.robot.libero.run_libero_eval import process_action
 from prismatic.models.action_heads import L1RegressionActionHead
 root=Path(os.environ['NEW_TASK_RUN']);frozen=json.loads((root/'dataset_v2/development_protocol/evaluation_protocol.json').read_text());assert frozen['status']=='FROZEN_STATIC_CHECKED' and len(frozen['new_development_states'])==20
 plan=json.loads((root/'policy_execution_plan.json').read_text());assert plan['maximum_control_steps']==300
 model=g['model'];prop=g['prop'];rlahead=g['head'];rlahead.cpu()
 with torch.random.fork_rng(devices=[0]):basehead=L1RegressionActionHead(input_dim=model.llm_dim,hidden_dim=model.llm_dim,action_dim=7,use_pro_version=True).to(dtype=torch.bfloat16)
 methods=plan['methods'];models=[f'{seed}_{method}' for seed in plan['orders'] for method in methods]
 paths={key:root/f'seed_{key.split("_",1)[0]}'/key.split('_',1)[1]/'formal/step0600.pt' for key in models}
 def sha(p):
  h=hashlib.sha256()
  with p.open('rb') as f:
   for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
  return h.hexdigest()
 hashes={k:sha(p) for k,p in paths.items()};checkpoints={}
 for key,p in paths.items():
  assert json.loads((p.parent/'training_acceptance.json').read_text())['successful_updates']==600
  ck=torch.load(p,map_location='cpu');assert ck['global_step']==600;checkpoints[key]=ck['parameters'];del ck
 g['save']('checkpoint_inventory.json',{k:dict(path=str(paths[k]),sha256=hashes[k],global_step=600) for k in models})
 stats=json.loads((root/'dataset_v2/normalization_statistics_expanded_train_only.json').read_text());model.base_model.model.norm_stats={'derived':stats}
 bddl=Path(get_libero_path('bddl_files'))/'libero_goal/open_the_middle_drawer_of_the_cabinet.bddl';language='open the middle drawer of the cabinet'
 def proprio(m,d):return np.r_[d.site_xpos[m.site('gripper0_grip_site').id],Rotation.from_quat(np.roll(d.xquat[m.body('robot0_right_hand').id],-1)).as_rotvec(),[d.qpos[m.joint('gripper0_finger_joint'+str(i)).qposadr[0]] for i in (1,2)]].astype(np.float32)
 def infer(images,p,head):
  prompt=g['QwenPromptBuilder']('openvla');prompt.add_turn('human',f'What action should the robot take to {language}?');prompt.add_turn('gpt','');tokens=g['tokenizer'](prompt.get_prompt(),add_special_tokens=True).input_ids;del tokens[-3:]
  inputs=dict(input_ids=torch.tensor([tokens],device='cuda'),attention_mask=torch.ones((1,len(tokens)),dtype=torch.bool,device='cuda'),pixel_values=torch.cat([g['imageproc'](Image.fromarray(im),return_tensors='pt')['pixel_values'] for im in images],dim=1).to('cuda',dtype=torch.bfloat16))
  s=stats['proprio'];lo=np.array(s['q01']);hi=np.array(s['q99']);p=np.clip(2*(p-lo)/(hi-lo+1e-8)-1,-1,1);p=np.where(np.array(s['min'])==np.array(s['max']),0,p)
  with torch.inference_mode():actions,_=model.predict_action(**inputs,unnorm_key='derived',proprio=p,proprio_projector=prop,action_head=head,use_film=False)
  actions=np.asarray(actions);assert actions.shape==(8,7) and np.isfinite(actions).all();return actions
 out=root/'closed_loop';out.mkdir(exist_ok=False);rows=[];geometry={}
 g['save']('frozen_runtime_protocol.json',dict(task=str(bddl),language=language,models=models,states=[s['seed'] for s in frozen['new_development_states']],max_steps=300,frequency_hz=20,chunk=8,backgrounds=[f'new_background_{i}' for i in range(1,5)],gate='one method both sampling orders clean>=12/20',failure_proxies={'partial_opening':'initial middle joint minus min recorded joint >=0.01 m, descriptive only','no_opening_progress':'otherwise'},success='official env.check_success(), not diagnostic threshold',infra_failure='retain and stop; no silent retry',future_or_targets_used_for_deployment=False))
 for stage in ['clean','background']:
  conditions=['clean'] if stage=='clean' else [f'new_background_{i}' for i in range(1,5)]
  if stage=='background':
   counts={k:sum(r['success'] for r in rows if r['model']==k) for k in models};gate=any(all(counts[f'{seed}_{m}']>=12 for seed in plan['orders']) for m in methods)
   with (root/'clean_gate.json').open('x') as f:json.dump(dict(counts=counts,denominator=20,passed=gate),f,indent=2)
   if not gate:break
  for state in frozen['new_development_states']:
   seed=state['seed'];folder=Path(state['state_file']).parent;initial=np.load(state['state_file']);assert hashlib.sha256(initial.tobytes()).hexdigest()==state['state_sha256']
   for condition in conditions:
    xml=folder/(condition+'.xml');expected=next(c for c in state['conditions'] if c['condition']==condition);assert sha(xml)==expected['XML_sha256']
    for key in models:
     dest=out/f'{seed}_{condition}_{key}';dest.mkdir();env=renderer=video=None;executed=[];chunks=[];states=[];ee=[];timeline=[];started=time.time()
     result=dict(model=key,seed=seed,condition=condition,checkpoint_sha256=hashes[key],state_sha256=state['state_sha256'],xml_sha256=expected['XML_sha256'],language=language,status='INVALID',success=None,steps=0,trajectory=str(dest/'trajectory.npz'),video=str(dest/'video.mp4'))
     try:
      basehead.cpu();rlahead.cpu();head=basehead if key.endswith('_BC-standard') else rlahead;head.to('cuda');mods=dict(vla=model,proprio=prop,head=head)
      for k,m in mods.items():
       full=m.state_dict();assert set(checkpoints[key][k]).issubset(full);full.update(checkpoints[key][k]);m.load_state_dict(full,strict=True);m.eval().requires_grad_(False)
      random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
      env=ControlEnv(bddl_file_name=str(bddl),use_camera_obs=False,has_offscreen_renderer=False,has_renderer=False,control_freq=20,horizon=1000,initialization_noise=None,ignore_done=False);env.seed(seed);env.reset();env.reset_from_xml_string(xml.read_text());env.set_init_state(initial);env.sim.forward();np.testing.assert_allclose(env.get_sim_state(),initial,rtol=0,atol=1e-8)
      m,d=env.sim.model._model,env.sim.data._data;current={k:np.array(getattr(d,k)).copy() for k in ['qpos','qvel','xpos','xquat','cam_xpos','cam_xmat']}
      if seed not in geometry:geometry[seed]=current
      else:
       for k in current:np.testing.assert_allclose(current[k],geometry[seed][k],rtol=0,atol=1e-8)
      assert not env.check_success() and m.vis.quality.offsamples==4
      joint=m.joint('wooden_cabinet_1_middle_level').qposadr[0];q0=float(d.qpos[joint]);renderer=mujoco.Renderer(m,224,224);opt=mujoco.MjvOption();opt.geomgroup[0]=0;opt.geomgroup[1]=1;opt.sitegroup[:]=0
      fk=mujoco.MjData(m);addresses=[m.joint(n).qposadr[0] for n in ['robot0_joint'+str(i) for i in range(1,8)]+['gripper0_finger_joint1','gripper0_finger_joint2']];queue=deque();video=imageio.get_writer(dest/'video.mp4',fps=2.5)
      for step in range(300):
       if not queue:
        assert g['counts'](g['before'])==g['counts'](g['ecc']());env.sim.forward();images=[]
        for camera in ['agentview','robot0_eye_in_hand']:
         renderer.update_scene(d,camera=camera,scene_option=opt);renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=1;images.append(renderer.render()[:,::-1].copy())
        mujoco.mj_resetData(m,fk);fk.qpos[addresses]=d.qpos[addresses];mujoco.mj_forward(m,fk);p=proprio(m,fk);np.testing.assert_array_equal(p,proprio(m,d));video.append_data(np.concatenate(images,axis=1));actions=infer(images,p,head);chunks.append(actions.copy());queue.extend(actions)
       action=process_action(queue.popleft().copy(),'openvla');assert np.isfinite(action).all();_,_,done,_=env.step(action.tolist());executed.append(action);states.append(env.get_sim_state().copy());ee.append(d.site_xpos[m.site('gripper0_grip_site').id].copy());result['steps']=step+1;success=bool(env.check_success());timeline.append(dict(step=step+1,middle_joint_m=float(d.qpos[joint]),goal=success,close_command=bool(action[6]>0)))
       if success or done:break
      progress=q0-min(t['middle_joint_m'] for t in timeline);result.update(status='COMPLETE',success=bool(env.check_success()),ending='success' if env.check_success() else 'timeout' if result['steps']==300 else 'environment_done',max_opening_progress_m=progress,final_middle_joint_m=timeline[-1]['middle_joint_m'],stage='success' if env.check_success() else 'partial_opening' if progress>=.01 else 'no_opening_progress',translation_saturation_fraction=float(np.mean(abs(np.asarray(executed)[:,:3])>=.999)))
     except Exception:result['error']=traceback.format_exc()
     finally:
      if video is not None:video.close()
      if renderer is not None:renderer.close()
      if env is not None:env.close()
      np.savez_compressed(dest/'trajectory.npz',executed_actions=executed,predicted_chunks=chunks,states=states,eef_positions=ee)
      with (dest/'phase_timeline.json').open('x') as f:json.dump(timeline,f)
     result['seconds']=time.time()-started
     with (dest/'result.json').open('x') as f:json.dump(result,f,indent=2)
     rows.append(result)
     with (root/'per_trial_results.json').open('w') as f:json.dump(rows,f,indent=2)
     print('TRIAL',len(rows),key,condition,result['status'],result['success'],flush=True)
     assert result['status']=='COMPLETE','Infrastructure failure preserved; no silent retry'
 keys=sorted(set().union(*(r.keys() for r in rows)))
 with (root/'per_trial_results.csv').open('x') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 assert len(rows) in [120,600]
 g['save']('evaluation_acceptance.json',dict(status='PASS_PROCESS_NOT_METHOD',valid_trials=len(rows),invalid_trials=0,physical_states=20,optimizer_updates=0,heldout_read=False))
