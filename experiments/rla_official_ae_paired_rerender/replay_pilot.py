"""One source-action replay per fixed pilot; no policy, no learned models.

Candidate states are NOT automatically source states. Clean correspondence gates
any subsequent paired rerender. Uses official LIBERO, CPU OSMesa only.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',LP_NUM_THREADS='2')
import argparse,json,sys,time,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,h5py,cv2
from PIL import Image,ImageDraw
from prepare import ROOT,RUN,sha,dump,csvout
LIB=RUN/'libero_official';sys.path.insert(0,str(LIB));os.environ['LIBERO_CONFIG_PATH']=str(RUN/'libero_config')
CAMS=['agentview','robot0_eye_in_hand']
def filtered(actions):
 ids=[];prev=None
 for t,a in enumerate(actions):
  noop=np.linalg.norm(a[:-1])<1e-4 and (prev is None or a[-1]==prev[-1])
  if noop:continue
  ids.append(t);prev=a
 return np.array(ids,dtype=int)
def main():
 p=argparse.ArgumentParser();p.add_argument('--suite',required=True);a=p.parse_args()
 out=RUN/'pilot_replay'/a.suite;out.mkdir(parents=True,exist_ok=False)
 identity=json.loads((RUN/f'hdf5_{a.suite}.json').read_text());source=Path(identity['path']);assert sha(source)==identity['sha256']
 rlds=np.load(RUN/'rlds_mapping_v2'/a.suite/'pilot_rlds.npz');records=json.loads((RUN/'rlds_mapping_v2'/a.suite/'progress.json').read_text())
 match=next(x for x in records if x.get('trajectory_id')=='000010');assert match['rlds_match']=='PASS'
 candidates=[];checked=[]
 with h5py.File(source) as f:
  g=f['data'];envmeta={k:str(v) for k,v in g.attrs.items()}
  for demo in sorted(g,key=lambda x:int(x.split('_')[-1])):
   actions=g[demo]['actions'][:];ids=filtered(actions);ok=actions[ids].shape==rlds['action'].shape and np.array_equal(actions[ids].astype(np.float32),rlds['action'])
   checked.append(dict(demo=demo,raw_frames=len(actions),kept=len(ids),full_filtered_actions_exact=bool(ok)))
   if ok:candidates.append((demo,ids))
  dump(out/'hdf5_action_candidates.json',dict(candidates=[x[0] for x in candidates],rows=checked,
   status='CANDIDATE_ONLY_NOT_CONFIRMED_SOURCE',hdf5=identity,env_metadata=envmeta))
  if len(candidates)!=1:
   dump(out/'result.json',dict(status='BLOCKED_NONUNIQUE_SOURCE_CANDIDATE',candidate_count=len(candidates),replay_control_steps=0,render_calls=0));return
  demo,ids=candidates[0];d=g[demo];initial=d['states'][0];rawxml=str(d.attrs.get('model_file',''));assert rawxml
  source_actions=d['actions'][:];source_states=d['states'][:]
  (out/'original_source.xml').write_text(rawxml)
  assert np.isfinite(source_states).all() and np.isfinite(source_actions).all()
  rawlayout=[(e.tag,e.get('name'),e.get('type')) for e in ET.fromstring(rawxml).iter() if e.tag in ['joint','body']]
  dump(out/'source_model_identity.json',dict(demo=demo,xml_sha256=hashlib.sha256(rawxml.encode()).hexdigest(),state_shape=list(source_states.shape),
    state_action_lengths_equal=len(source_states)==len(source_actions),XML_body_joint_layout=rawlayout,
    source_note='Original HDF5 states are not regenerated-RLDS states; only initial used, followed by author-defined settling and filtered actions'))
 import mujoco,robosuite
 from robosuite.utils import binding_utils as binding
 counters=dict(render_calls=0,physics_substeps=0,state_restores=0,action_control_steps=0,settling_control_steps=0)
 original_render=binding.MjSim.render;original_step=binding.MjSim.step;original_set=binding.MjSim.set_state_from_flattened
 def render_count(self,*args,**kwargs):counters['render_calls']+=1;return original_render(self,*args,**kwargs)
 def step_count(self,*args,**kwargs):counters['physics_substeps']+=1;return original_step(self,*args,**kwargs)
 def set_count(self,*args,**kwargs):counters['state_restores']+=1;return original_set(self,*args,**kwargs)
 binding.MjSim.render=render_count;binding.MjSim.step=step_count;binding.MjSim.set_state_from_flattened=set_count
 from libero.libero.envs import OffScreenRenderEnv
 bddl=LIB/'libero/libero/bddl_files'/a.suite/(identity['filename'].split('/')[-1].removesuffix('_demo.hdf5')+'.bddl');assert bddl.exists()
 env=None;writers=[];started=time.time();rows=[]
 try:
  env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256)
  env.seed(0);env.reset();m=env.sim.model;data=env.sim.data
  assert len(initial)==1+m.nq+m.nv and m.na==0
  runtime_xml=env.sim.model.get_xml();(out/'runtime_model.xml').write_text(runtime_xml)
  runtime_joints=[m.joint_id2name(i) for i in range(m.njnt)]
  source_joints=[e.get('name') for e in ET.fromstring(rawxml).find('worldbody').iter('joint')]
  assert source_joints==runtime_joints, 'Source/runtime joint order differs; do not use same-size state'
  env.set_init_state(initial)
  for _ in range(10):obs,*_=env.step([0,0,0,0,0,0,-1]);counters['settling_control_steps']+=1
  arm=[m.get_joint_qpos_addr('robot0_joint'+str(i)) for i in range(1,8)]
  site=m.site_name2id('gripper0_grip_site')
  dump(out/'environment.json',dict(libero_commit='8f1084e3132a39270c3a13ebe37270a43ece2a01',mujoco=mujoco.__version__,robosuite=robosuite.__version__,
    bddl=str(bddl),bddl_sha256=sha(bddl),runtime_xml_sha256=sha(out/'runtime_model.xml'),joint_order_exact=True,
    joint_names=runtime_joints,source_demo=demo,first_original_state_time=float(initial[0]),first_candidate_time=float(data.time),
    no_policy_loaded=True,render_backend='OSMesa CPU',init_seed=0,source_env_version='not recorded in published dataset',
    note='Pinned official environment, not LIBERO-Plus. No alternate environment search. Replay is candidate reconstruction.'))
  writer=cv2.VideoWriter(str(out/'full_clean_comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(512,512));assert writer.isOpened();writers.append(writer)
  # Compare to original published converted video, never replace it with replay.
  caps=[cv2.VideoCapture(str(ROOT/'data/libero_converted'/a.suite/'traj_000010'/f'{name}_rgb.mp4')) for name in ['agentview_camera','wrist_camera']]
  states=[];candidate_rgb=[];camera=[];bodypos=[]
  fixed=sorted(set(np.linspace(0,len(ids)-1,7,dtype=int).tolist()+[len(ids)//2]));sheet=Image.new('RGB',(1024,len(fixed)*282),'#222');draw=ImageDraw.Draw(sheet)
  for t,raw_t in enumerate(ids):
   state=env.sim.get_state().flatten().copy();states.append(state);bodypos.append(data.body_xpos.copy())
   camera.append(np.concatenate([data.cam_xpos.reshape(-1),data.cam_xmat.reshape(-1)]))
   rgb=np.stack([env.sim.render(camera_name=c,width=256,height=256)[::-1,::-1].copy() for c in CAMS]);candidate_rgb.append(rgb)
   assert np.array_equal(state,env.sim.get_state().flatten()),'render advanced physics'
   orig=[]
   for cap in caps:ok,bgr=cap.read();assert ok;orig.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
   orig=np.stack(orig);delta=np.abs(rgb.astype(np.int16)-orig.astype(np.int16))
   row=dict(frame=t,raw_action_index=int(raw_t),candidate_time=float(data.time),
     joint_max_abs=float(np.max(abs(data.qpos[arm]-rlds['joints'][t]))),eef_max_abs_m=float(np.max(abs(data.site_xpos[site]-rlds['state'][t,:3]))),
     obs_joint_max_abs=float(np.max(abs(obs['robot0_joint_pos']-rlds['joints'][t]))),
     front_mae=float(delta[0].mean()),wrist_mae=float(delta[1].mean()),front_p99=float(np.quantile(delta[0],.99)),wrist_p99=float(np.quantile(delta[1],.99)),
     obs_vs_direct_max=int(max(np.max(np.abs(obs[c+'_image'][::-1,::-1].astype(np.int16)-rgb[i].astype(np.int16))) for i,c in enumerate(CAMS))))
   rows.append(row)
   panel=np.concatenate([np.concatenate([orig[0],rgb[0]],1),np.concatenate([orig[1],rgb[1]],1)],0);writer.write(cv2.cvtColor(panel,cv2.COLOR_RGB2BGR))
   if t in fixed:
    y=fixed.index(t)*282
    for j,img in enumerate([orig[0],rgb[0],orig[1],rgb[1]]):sheet.paste(Image.fromarray(img),(j*256,y+26));draw.text((j*256+2,y+2),f't={t} '+['original front','candidate front','original wrist','candidate wrist'][j],fill='white')
   obs,*_=env.step(source_actions[raw_t].tolist());counters['action_control_steps']+=1
   if t%25==0:print(a.suite,t,len(ids),row['joint_max_abs'],row['front_mae'],flush=True)
  writer.release();writers=[]
  for cap in caps:cap.release()
  sheet.save(out/'fixed_clean_comparison.png');csvout(out/'frame_alignment.csv',rows)
  np.savez_compressed(out/'candidate_replay.npz',states=np.stack(states),rgb=np.stack(candidate_rgb),source_action_indices=ids,actions=source_actions[ids],camera_world=np.stack(camera),body_positions=np.stack(bodypos))
  numeric=all(r['joint_max_abs']<=.002 and r['eef_max_abs_m']<=.002 for r in rows)
  pixels=all(r['front_mae']<=8 and r['wrist_mae']<=8 and r['front_p99']<=40 and r['wrist_p99']<=40 for r in rows)
  dump(out/'result.json',dict(status='PENDING_VISUAL_REVIEW' if numeric and pixels else 'BLOCKED_CLEAN_REPRODUCTION',source_candidate=demo,
    source_hdf5_sha256=identity['sha256'],numeric_gate=numeric,pixel_gate=pixels,
    max_joint_error=max(r['joint_max_abs'] for r in rows),max_eef_error=max(r['eef_max_abs_m'] for r in rows),
    frames=len(rows),first_frame=rows[0],counters=counters,seconds=time.time()-started,
    recovered_candidate_states=len(states),certified_original_states=0,paired_frames=0,
    original_source_identity='action candidate; confirmation requires complete replay correspondence, not actions alone',
    AE_forwards=0,DINO_forwards=0,policy_forwards=0,updates=0,backward=0,policy_rollouts=0))
 except BaseException as e:
  dump(out/'failure.json',dict(error=repr(e),counters=counters,frames_completed=len(rows)));raise
 finally:
  for w in writers:w.release()
  if env is not None:env.close()
if __name__=='__main__':main()
