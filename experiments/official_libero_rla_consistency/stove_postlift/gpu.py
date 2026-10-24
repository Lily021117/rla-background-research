"""Frozen same-input inference and bounded clean prefix handoffs, no backward."""
import os,time,random,copy,pickle,collections
import numpy as np
import torch,mujoco,imageio.v2 as imageio
from PIL import Image
from scipy.spatial.transform import Rotation
from stove_postlift.common import *

def primitive(v):
 if v is None or isinstance(v,(str,bool,int,float,np.number,np.ndarray)):return True
 if isinstance(v,(list,tuple)):return all(primitive(x) for x in v)
 if isinstance(v,dict):return all(isinstance(k,(str,int)) and primitive(x) for k,x in v.items())
 return False
def capture_obj(obj):
 out={};skipped=[]
 for k,v in vars(obj).items():
  if primitive(v):out[k]=('value',copy.deepcopy(v))
  elif type(v).__module__.startswith('robosuite.utils.buffers'):out[k]=('buffer',copy.deepcopy(vars(v)))
  else:skipped.append((k,type(v).__name__))
 return out,skipped
def object_map(e):
 r=e.robots[0];out={'environment':e.env,'robot':r,'controller':r.controller,'gripper':r.gripper}
 out.update({'observable:'+k:v for k,v in e.env._observables.items()})
 return out
def snapshot(e):
 m,d=e.sim.model._model,e.sim.data._data;flag=mujoco.mjtState.mjSTATE_INTEGRATION;x=np.empty(mujoco.mj_stateSize(m,flag));mujoco.mj_getState(m,d,x,flag)
 assert e.robots[0].controller.interpolator_pos is None and e.robots[0].controller.interpolator_ori is None
 return dict(integration=x,objects={k:capture_obj(v) for k,v in object_map(e).items()},python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),action_queue=[],policy_history='stateless single observation; no KV cache between chunks')
def restore(e,s):
 m,d=e.sim.model._model,e.sim.data._data;mujoco.mj_setState(m,d,s['integration'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
 for name,obj in object_map(e).items():
  for k,(kind,value) in s['objects'][name][0].items():
   if kind=='value':setattr(obj,k,copy.deepcopy(value))
   else:vars(getattr(obj,k)).update(copy.deepcopy(value))
 random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch']);torch.cuda.set_rng_state_all(s['cuda'])
 x=np.empty_like(s['integration']);mujoco.mj_getState(m,d,x,mujoco.mjtState.mjSTATE_INTEGRATION)
 # mj_forward recomputes solver warmstart; restore exact integration inputs again.
 mujoco.mj_setState(m,d,s['integration'],mujoco.mjtState.mjSTATE_INTEGRATION)

def execute(g):
 from experiments.robot.libero.run_libero_eval import process_action
 root=g['a'].run.parent;protocol=read(root/'diagnostic_protocol.json');plan=read(root/'run_plan.json');registry=read(root/'policy_registry.json');checkpoints={k:torch.load(v,map_location='cpu') for k,v in registry.items()};mods=g['mods'];model,head,prop=mods['vla'],mods['head'],mods['proprio'];stats=read(plan['inputs']['action_proprio_statistics']);model.base_model.model.norm_stats={'derived':stats};loaded=None
 for k,p in registry.items():assert sha(p)==read(root/'source_version.json')['policy_hashes'][k] and checkpoints[k]['global_step']==600
 def load(arm):
  nonlocal loaded
  if arm==loaded:return
  for k,m in mods.items():
   sd=m.state_dict();assert set(checkpoints[arm]['parameters'][k]).issubset(sd);sd.update(checkpoints[arm]['parameters'][k]);m.load_state_dict(sd,strict=True);m.eval().requires_grad_(False)
  loaded=arm
 def infer(images,p):
  prompt=g['QwenPromptBuilder']('openvla');prompt.add_turn('human','What action should the robot take to pick up the black bowl on the stove and place it on the plate?');prompt.add_turn('gpt','');tokens=g['tokenizer'](prompt.get_prompt(),add_special_tokens=True).input_ids;del tokens[-3:]
  inp=dict(input_ids=torch.tensor([tokens],device='cuda'),attention_mask=torch.ones((1,len(tokens)),dtype=torch.bool,device='cuda'),pixel_values=torch.cat([g['imageproc'](Image.fromarray(im),return_tensors='pt')['pixel_values'] for im in images],dim=1).to('cuda',dtype=torch.bfloat16))
  s=stats['proprio'];lo=np.array(s['q01']);hi=np.array(s['q99']);p=np.clip(2*(p-lo)/(hi-lo+1e-8)-1,-1,1);p=np.where(np.array(s['min'])==np.array(s['max']),0,p)
  with torch.inference_mode():v,_=model.predict_action(**inp,unnorm_key='derived',proprio=p,proprio_projector=prop,action_head=head,use_film=False)
  v=np.asarray(v);assert v.shape==(8,7) and np.isfinite(v).all();return v
 def observe(e,render):
  e.sim.forward();m,d=e.sim.model._model,e.sim.data._data;opt=mujoco.MjvOption();opt.geomgroup[0]=0;opt.geomgroup[1]=1;opt.sitegroup[:]=0;images=[]
  for camera in ('agentview','robot0_eye_in_hand'):
   render.update_scene(d,camera=camera,scene_option=opt);render.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=1;images.append(render.render()[:,::-1].copy())
  fk=mujoco.MjData(m);names=['robot0_joint'+str(i) for i in range(1,8)]+['gripper0_finger_joint1','gripper0_finger_joint2'];idx=[int(m.joint(n).qposadr[0]) for n in names];fk.qpos[idx]=d.qpos[idx];mujoco.mj_forward(m,fk)
  p=np.r_[fk.site_xpos[m.site('gripper0_grip_site').id],Rotation.from_quat(np.roll(fk.xquat[m.body('robot0_right_hand').id],-1)).as_rotvec(),fk.qpos[idx[-2:]]].astype(np.float32);return images,p
 def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
 historical={(r['model'],r['seed']):r for r in trials() if r['condition']=='clean'};results=[];prefixes=[];gates=[];out=root/'handoffs';out.mkdir(exist_ok=False)
 for st in states()[:5]:
  for prefix in ('C','D-center'):
   assert len(prefixes)<10;seed(st['seed']);load(prefix);e=env_make(st,'clean');m,d=e.sim.model._model,e.sim.data._data;render=mujoco.Renderer(m,224,224);bid=m.body('akita_black_bowl_1_main').id;z0=float(d.xpos[bid,2]);h=np.load(historical[prefix,st['seed']]['trajectory']);recorded_states=h['states'];recorded_actions=h['executed_actions'];q=collections.deque();step=0;first=None;maxerr=0.;maxact=0.;dest=out/f"{st['seed']}_{prefix}_prefix";dest.mkdir();prefixes.append(dict(seed=st['seed'],prefix=prefix,status='STARTED'));save(dest/'started.json',prefixes[-1]);writer=imageio.get_writer(dest/'video.mp4',fps=2.5)
   while step<220:
    if not q:
     images,p=observe(e,render);writer.append_data(np.concatenate(images,axis=1));q.extend(infer(images,p))
    act=process_action(q.popleft().copy(),'openvla');e.step(act.tolist());maxact=max(maxact,float(abs(act-recorded_actions[step]).max()));maxerr=max(maxerr,float(abs(e.get_sim_state()-recorded_states[step]).max()));step+=1
    if first is None and float(d.xpos[bid,2])-z0>=.03:first=step
    if first is not None and not q:break
   writer.close();assert first is not None and step<220,'no eligible prefix boundary';assert maxerr<=1e-8 and maxact<=1e-6,'prefix differs from historical run'
   snap=snapshot(e);prefixes[-1].update(status='COMPLETE',boundary=step,first_lift=first,historical_state_max_error=maxerr,historical_action_max_error=maxact)
   with (dest/'snapshot_local_only.pkl').open('xb') as f:pickle.dump(snap,f)
   save(dest/'snapshot_inventory.json',dict(components={k:dict(saved=list(v[0]),skipped=v[1]) for k,v in snap['objects'].items()},queue_empty=True,remaining_budget=220-step))
   # Self continuation first. Cross continuation forbidden unless exact historical suffix reproduces.
   for continuation in (prefix,'D-center' if prefix=='C' else 'C'):
    assert len(results)<20;load(continuation);restore(e,snap);q=collections.deque();suffix=[];actions=[];video_dir=out/f"{st['seed']}_{prefix}_to_{continuation}";video_dir.mkdir();writer=imageio.get_writer(video_dir/'video.mp4',fps=2.5);err=0.;aerr=0.;start_step=step
    for t in range(start_step,220):
     if not q:
      assert g['counts'](g['before'])==g['counts'](g['ecc']());images,p=observe(e,render);writer.append_data(np.concatenate(images,axis=1));q.extend(infer(images,p))
     act=process_action(q.popleft().copy(),'openvla');_,_,done,_=e.step(act.tolist());actions.append(act);suffix.append(e.get_sim_state().copy())
     if continuation==prefix:
      assert t<len(recorded_states),'restored self lasts longer than original';err=max(err,float(abs(suffix[-1]-recorded_states[t]).max()));aerr=max(aerr,float(abs(act-recorded_actions[t]).max()))
     if e.check_success() or done:break
    writer.close();success=bool(e.check_success());np.savez_compressed(video_dir/'trajectory.npz',states=suffix,actions=actions);row=dict(seed=st['seed'],prefix=prefix,continuation=continuation,boundary=start_step,remaining_budget=220-start_step,executed_steps=len(suffix),success=success,video=str(video_dir/'video.mp4'),trajectory=str(video_dir/'trajectory.npz'),status='COMPLETE',diagnostic_not_formal=True)
    if continuation==prefix:
     passed=err<=1e-8 and aerr<=1e-6 and len(suffix)+start_step==len(recorded_states) and success==historical[prefix,st['seed']]['success'];gate=dict(seed=st['seed'],prefix=prefix,state_max_error=err,action_max_error=aerr,terminal_success_matches=success==historical[prefix,st['seed']]['success'],PASS=passed);gates.append(gate);save(video_dir/'restore_gate.json',gate)
     if not passed:
      row['status']='RESTORE_GATE_FAILED';results.append(row);save(root/'handoff_acceptance.json',dict(status='BLOCKED_RESTORE',prefixes=prefixes,continuations=results,gates=gates,optimizer_updates=0));render.close();e.close();return
    # Recompute geometric predicates after stopping; original success label remains separately recorded.
    e.sim.forward();row.update({'terminal_'+k:v for k,v in metrics(e).items()});results.append(row);save(video_dir/'result.json',row);print('HANDOFF',len(results),prefix,continuation,success,flush=True)
   render.close();e.close()
 save(root/'handoff_acceptance.json',dict(status='PASS_RESTORE',prefixes=prefixes,continuations=results,gates=gates,optimizer_updates=0));csvsave(root/'handoff_results.csv',results)
 # Same-input comparisons, never treat C action as ground truth on rollout states.
 fixed=read(root/'fixed_demo_samples.json');fixed+=read(root/'fixed_rollout_samples.json');save(root/'same_input_sample_manifest.json',fixed);rows=[];env=None;render=None;last_key=None
 def compare(x,images,p,target=None):
  pred={}
  for arm in ('C','D-center'):load(arm);seed(20261023);pred[arm]=infer(images,p)
  for horizon in range(8):
   row=dict(source=x['source'],split=x.get('split','development_rollout'),episode=x.get('episode',''),seed=x.get('seed',''),appearance=x['appearance'],t=x.get('t',x.get('pre_step')),model_source=x.get('model_source',''),horizon=horizon+1)
   for i in range(7):
    row['dim'+str(i)+'_output_difference']=float(pred['D-center'][horizon,i]-pred['C'][horizon,i])
    for arm in pred:
     row[arm+'_dim'+str(i)+'_prediction']=float(pred[arm][horizon,i])
     if target is not None:row[arm+'_dim'+str(i)+'_absolute_error']=float(abs(pred[arm][horizon,i]-target[horizon,i]))
   rows.append(row)
 for n,x in enumerate(fixed):
  if x['source']=='demo':
   f=np.load(Path(x['root'])/'frames'/f"{x['t']:04}.npz");lab=np.load(Path(x['root'])/'trajectory_labels.npz');target=lab['actions'][x['t']:x['t']+8].astype(float).copy();target[:,:6]=np.clip(target[:,:6],np.array(stats['action']['q01'])[:6],np.array(stats['action']['q99'])[:6]);target[:,6]=1-np.clip(target[:,6],0,1);compare(x,f[x['appearance']].transpose(0,2,3,1),f['proprio'],target)
  else:
   key=(x['seed'],x['appearance'])
   if key!=last_key:
    if render:render.close();env.close()
    env=env_make(next(s for s in states() if s['seed']==x['seed']),x['appearance']);render=mujoco.Renderer(env.sim.model._model,224,224);last_key=key
   env.set_state(np.load(x['trajectory'])['states'][x['state_index']]);images,p=observe(env,render);compare(x,images,p)
  if n%30==0:print('SAME_INPUT',n,len(fixed),flush=True)
 if render:render.close();env.close()
 csvsave(root/'same_input_action_metrics.csv',rows);save(root/'forward_acceptance.json',dict(samples=len(fixed),rows=len(rows),optimizer_updates=0,backward=0,target_coordinates='de-normalized policy action; translation/rotation are controller delta commands, not Cartesian displacement in metres; gripper0close/1open; demo continuous actions clipped at fixed train Q99 bounds; no ground truth labels for rollout samples',future_inputs=False))
