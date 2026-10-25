"""Only the two frozen600-step low-LR models;120 paired development rollouts."""
import os,json,time,random,traceback,hashlib,csv
from pathlib import Path
from collections import deque
import numpy as np
import torch
import mujoco,imageio
from PIL import Image
from scipy.spatial.transform import Rotation
from rla_lr_budget_sweep.interface import initialize

def execute(g):
 from libero.libero.envs.env_wrapper import ControlEnv
 from libero.libero import get_libero_path
 from experiments.robot.libero.run_libero_eval import process_action
 from stove_postlift.common import metrics as task_metrics
 root=g['a'].run.parent;plan=json.loads((root/'run_plan.json').read_text());prev=Path(plan['previous']);protocol=json.loads((root/'evaluation_protocol.json').read_text());assert protocol['trials']==120
 basehead=g['head'].cpu();_,_,_=initialize(g);rlahead=g['head'];rlahead.cpu();torch.cuda.empty_cache();model=g['model'];prop=g['prop'];mods=g['mods']
 registry=json.loads((root/os.environ.get('RLA_POLICY_REGISTRY','policy_registry.json')).read_text());paths={k:Path(v) for k,v in registry.items()}
 for arm in paths:
  if arm!='B20':assert json.loads((paths[arm].parent/'training_acceptance.json').read_text())['successful_updates']==600
 def sha(p):
  h=hashlib.sha256()
  with p.open('rb') as f:
   for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
  return h.hexdigest()
 hashes={k:sha(p) for k,p in paths.items()};g['save']('checkpoint_inventory.json',{k:dict(path=str(v),sha256=hashes[k],global_step=int(k.rsplit('_s',1)[1])) for k,v in paths.items()})
 if 'B20' in paths:
  with (prev.parent/'run_20261023_06/checkpoint_inventory.csv').open() as f:baseline_inventory=list(csv.DictReader(f))
  assert any(r.get('sha256')==hashes['B20'] and 'B20' in r.values() for r in baseline_inventory),'B20 final hash does not match frozen inventory'
 checkpoints={}
 for k,p in paths.items():
  ck=torch.load(p,map_location='cpu');assert ck['global_step']==int(k.rsplit('_s',1)[1]);checkpoints[k]={'parameters':ck['parameters']};del ck
 frozen=json.loads((prev/'development_protocol/evaluation_protocol.json').read_text());stats=json.loads(Path(plan['inputs']['action_proprio_statistics']).read_text());model.base_model.model.norm_stats={'derived':stats}
 bddl=Path(get_libero_path('bddl_files'))/'libero_spatial'/protocol.get('bddl','pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate.bddl');language=protocol.get('task','pick up the black bowl on the cookie box and place it on the plate')
 def proprio(m,d):return np.r_[d.site_xpos[m.site('gripper0_grip_site').id],Rotation.from_quat(np.roll(d.xquat[m.body('robot0_right_hand').id],-1)).as_rotvec(),[d.qpos[m.joint('gripper0_finger_joint'+str(i)).qposadr[0]] for i in (1,2)]].astype(np.float32)
 def infer(images,p,head):
  prompt=g['QwenPromptBuilder']('openvla');prompt.add_turn('human',f'What action should the robot take to {language}?');prompt.add_turn('gpt','');tokens=g['tokenizer'](prompt.get_prompt(),add_special_tokens=True).input_ids;del tokens[-3:]
  inp=dict(input_ids=torch.tensor([tokens],device='cuda'),attention_mask=torch.ones((1,len(tokens)),dtype=torch.bool,device='cuda'),pixel_values=torch.cat([g['imageproc'](Image.fromarray(im),return_tensors='pt')['pixel_values'] for im in images],dim=1).to('cuda',dtype=torch.bfloat16))
  s=stats['proprio'];lo=np.array(s['q01']);hi=np.array(s['q99']);p=np.clip(2*(p-lo)/(hi-lo+1e-8)-1,-1,1);p=np.where(np.array(s['min'])==np.array(s['max']),0,p)
  with torch.inference_mode():actions,_=model.predict_action(**inp,unnorm_key='derived',proprio=p,proprio_projector=prop,action_head=head,use_film=False)
  actions=np.asarray(actions);assert actions.shape==(8,7) and np.isfinite(actions).all();return actions
 out=root/os.environ.get('RLA_EVAL_DIRECTORY','closed_loop');resume=os.environ.get('CENTER_RESUME_EVALUATION')=='1';out.mkdir(exist_ok=resume);rows=[];geometries={}
 completed=set();interrupted=[]
 if resume:
  rows=json.loads((root/'per_trial_results.json').read_text())
  order=[(s['seed'],c,a) for s in frozen['new_development_states'] for c in protocol['conditions'] for a in protocol['models']]
  assert [(r['seed'],r['condition'],r['model']) for r in rows]==order[:len(rows)]
  for r in rows:
   assert r['status']=='COMPLETE' and r['checkpoint_sha256']==hashes[r['model']]
   assert Path(r['trajectory']).is_file() and Path(r['video']).is_file()
   assert json.loads(Path(r['trajectory']).with_name('result.json').read_text())==r
   completed.add((r['seed'],r['condition'],r['model']))
  for seed,condition,arm in order[len(rows):]:
   partial=out/f'{seed}_{condition}_{arm}'
   if partial.exists():
    assert not (partial/'result.json').exists(),'unexpected completed result outside ledger'
    interrupted.append(dict(seed=seed,condition=condition,model=arm,path=str(partial),files=[str(p) for p in partial.iterdir()],reason='process interrupted during network interruption; no completed result',status='INVALID_INFRASTRUCTURE_INTERRUPTION'))
  with (root/'evaluation_resume_audit.json').open('x') as f:json.dump(dict(preserved_valid_trials=len(rows),interrupted_attempts=interrupted,remaining_trials=120-len(rows),protocol_unchanged=True),f,indent=2)
 # Descriptive stages fixed before viewing outcomes, not ground-truth grasp labels.
 g['save']('failure_stage_rule.json',dict(source='recorded simulator target lift and EE-to-target distance; descriptive proxies only',rules=['success if env.check_success','if target max lift>=0.03m: post_lift_or_placement_failure','else if minimum EE-target center distance<=0.10m: near_target_grasp_or_contact_failure','else approach_failure'],no_new_rollouts_for_diagnosis=True))
 for state in frozen['new_development_states']:
  seed=state['seed'];folder=Path(state['state_file']).parent;initial=np.load(state['state_file']);assert hashlib.sha256(initial.tobytes()).hexdigest()==state['state_sha256'] # frozen protocol hashes array contents, not NPY container header
  for condition in protocol['conditions']:
   xml=folder/(condition+'.xml');expected=next(c for c in state['conditions'] if c['condition']==condition);assert sha(xml)==expected['XML_sha256']
   for arm in protocol['models']:
    if (seed,condition,arm) in completed:
     prior=next(r for r in rows if (r['seed'],r['condition'],r['model'])==(seed,condition,arm))
     assert prior['state_sha256']==state['state_sha256'] and prior['xml_sha256']==expected['XML_sha256'] and prior['language']==language
     continue
    dest=out/f'{seed}_{condition}_{arm}'
    if resume and dest.exists():dest=out/f'{seed}_{condition}_{arm}_resume1'
    dest.mkdir(exist_ok=False);env=render=video=None;executed=[];chunks=[];states=[];ee=[];bowl=[];phase=[];result=dict(model=arm,seed=seed,condition=condition,checkpoint_sha256=hashes[arm],state_sha256=state['state_sha256'],xml_sha256=expected['XML_sha256'],language=language,status='INVALID',success=None,steps=0,video=str(dest/'video.mp4'),trajectory=str(dest/'trajectory.npz'));start=time.time()
    try:
     basehead.cpu();rlahead.cpu();head=basehead if arm in ('B20','B') else rlahead;head.to('cuda');mods['head']=head
     ck=checkpoints[arm]
     for k,m in mods.items():
      full=m.state_dict();assert set(ck['parameters'][k]).issubset(full);full.update(ck['parameters'][k]);m.load_state_dict(full,strict=True);m.eval().requires_grad_(False)
     random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
     env=ControlEnv(bddl_file_name=str(bddl),use_camera_obs=False,has_offscreen_renderer=False,has_renderer=False,control_freq=20,horizon=1000,initialization_noise=None,ignore_done=False);env.seed(seed);env.reset();env.reset_from_xml_string(xml.read_text());env.set_init_state(initial);env.sim.forward();np.testing.assert_allclose(env.get_sim_state(),initial,rtol=0,atol=1e-8)
     m,d=env.sim.model._model,env.sim.data._data;geom={k:np.array(getattr(d,k)).copy() for k in ('qpos','qvel','xpos','xquat','cam_xpos','cam_xmat')}
     if seed not in geometries:geometries[seed]=geom
     else:
      for k in geom:np.testing.assert_allclose(geom[k],geometries[seed][k],rtol=0,atol=1e-8)
     assert not env.check_success();assert m.vis.quality.offsamples==4;bid=m.body('akita_black_bowl_1_main').id;initial_bowl=d.xpos[bid].copy();render=mujoco.Renderer(m,224,224);opt=mujoco.MjvOption();opt.geomgroup[0]=0;opt.geomgroup[1]=1;opt.sitegroup[:]=0
     robot_fk=mujoco.MjData(m);names=['robot0_joint'+str(i) for i in range(1,8)]+['gripper0_finger_joint1','gripper0_finger_joint2'];addresses=[m.joint(n).qposadr[0] for n in names]
     q=deque();video=imageio.get_writer(dest/'video.mp4',fps=2.5)
     for step in range(220):
      if not q:
       assert g['counts'](g['before'])==g['counts'](g['ecc']());env.sim.forward();images=[]
       for camera in ('agentview','robot0_eye_in_hand'):
        render.update_scene(d,camera=camera,scene_option=opt);render.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=1;images.append(render.render()[:,::-1].copy())
       mujoco.mj_resetData(m,robot_fk);robot_fk.qpos[addresses]=d.qpos[addresses];mujoco.mj_forward(m,robot_fk);p=proprio(m,robot_fk);np.testing.assert_array_equal(p,proprio(m,d));video.append_data(np.concatenate(images,axis=1));acts=infer(images,p,head);chunks.append(acts.copy());q.extend(acts)
      action=process_action(q.popleft().copy(),'openvla');assert np.isfinite(action).all();_,_,done,_=env.step(action.tolist());executed.append(action);states.append(env.get_sim_state().copy());ee.append(d.site_xpos[m.site('gripper0_grip_site').id].copy());bowl.append(d.xpos[bid].copy());result['steps']=step+1
      phase.append(dict(step=step+1,close_command=bool(action[6]>0),**task_metrics(env)))
      if env.check_success() or done:break
     lift=float(np.max(np.array(bowl)[:,2]-initial_bowl[2]));near=float(np.linalg.norm(np.array(ee)-np.array(bowl),axis=1).min());success=bool(env.check_success())
     result.update(issued_close_command=bool(np.any(np.asarray(executed)[:,6]>0)),close_command_count=int(np.sum(np.asarray(executed)[:,6]>0)),status='COMPLETE',success=success,ending='success' if success else 'timeout' if result['steps']==220 else 'environment_done',max_lift_m=lift,min_ee_target_distance_m=near,stage='success' if success else 'post_lift_or_placement_failure' if lift>=.03 else 'near_target_grasp_or_contact_failure' if near<=.10 else 'approach_failure',geometry_pair_check=True,translation_saturation_fraction=float(np.mean(abs(np.asarray(executed)[:,:3])>=.999)))
     first_lift=next((i for i,b in enumerate(bowl) if b[2]-initial_bowl[2]>=.03),None)
     post=phase[first_lift:] if first_lift is not None else []
     descending=[(x,y) for x,y in zip(post[:-1],post[1:]) if y['bowl_2']<x['bowl_2']]
     result.update(first_lift_step=None if first_lift is None else first_lift+1,descent_intervals=len(descending),descent_horizontal_path_m=sum(float(np.hypot(y['bowl_0']-x['bowl_0'],y['bowl_1']-x['bowl_1'])) for x,y in descending),descent_xy_error_change_m=sum(y['xy_distance_m']-x['xy_distance_m'] for x,y in descending),final_xy_m=phase[-1]['xy_distance_m'],final_vertical_m=phase[-1]['vertical_bowl_plate_m'],postlift_plate_contact=any(x['contact_condition'] for x in post),postlift_centered=any(x['xy_condition'] for x in post),postlift_contact_height_xy=any(x['goal'] for x in post),postlift_open=any(not x['close_command'] for x in post))
    except Exception:result['error']=traceback.format_exc()
    finally:
     if video:video.close()
     if render:render.close()
     if env:env.close()
     np.savez_compressed(dest/'trajectory.npz',executed_actions=executed,predicted_chunks=chunks,states=states,eef_positions=ee,target_bowl_positions=bowl)
     with (dest/'phase_timeline.json').open('x') as f:json.dump(phase,f)
    result['seconds']=time.time()-start;(dest/'result.json').write_text(json.dumps(result,indent=2));rows.append(result)
    with (root/'per_trial_results.json').open('w') as f:json.dump(rows,f,indent=2)
    print('TRIAL',len(rows),arm,seed,condition,result['status'],result['success'],flush=True)
    assert result['status']=='COMPLETE','infrastructure failure retained; stop, no silent retry'
 keys=sorted(set().union(*(r.keys() for r in rows)))
 with (root/'per_trial_results.csv').open('x') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 g['save']('evaluation_acceptance.json',dict(status='PASS_PROCESS_NOT_METHOD',valid_trials=len(rows),invalid_trials=len(interrupted),physical_states=20,optimizer_updates=0,heldout_read=False,peak_allocated=torch.cuda.max_memory_allocated()))

