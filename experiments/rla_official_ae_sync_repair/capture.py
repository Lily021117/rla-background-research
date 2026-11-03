"""One bounded recapture per pilot; all diagnostic mj_forward calls on copies.

Primary export is the explicit pre-next-action boundary time. Cached observations
retain separate acquisition time. A sensor-time render is diagnostic only: it is
not used to shift/relabel frames or replace the primary export.
"""
from common import *
import argparse,copy,time,sys,h5py,cv2
from PIL import Image,ImageDraw
sys.path.insert(0,str(OLD/'libero_official'))
os.environ['LIBERO_CONFIG_PATH']=str(OLD/'libero_config')

def main(suite):
 out=RUN/'capture'/suite;out.mkdir(parents=True,exist_ok=False)
 assert (RUN/'offline'/suite/'result.json').exists(), 'offline first'
 src=OLD/'pilot_replay'/suite;previous=np.load(src/'candidate_replay.npz')
 ident=json.loads((OLD/f'hdf5_{suite}.json').read_text());demo=json.loads((src/'result.json').read_text())['source_candidate']
 assert sha(ident['path'])==ident['sha256']
 with h5py.File(ident['path']) as f:initial=f['data'][demo]['states'][0];actions=f['data'][demo]['actions'][:]
 from libero.libero.envs import OffScreenRenderEnv
 from robosuite.utils import binding_utils as binding
 counts=dict(source_actions=0,wait_steps=0,physics_substeps=0,env_render_calls=0,diagnostic_rgb_renders=0,state_restores=0,copy_forwards=0)
 step0=binding.MjSim.step;render0=binding.MjSim.render;forward0=binding.MjSim.forward;set0=binding.MjSim.set_state_from_flattened
 latest={};sensor_log={}
 def step(self,*a,**k):counts['physics_substeps']+=1;return step0(self,*a,**k)
 def rend(self,*a,**k):counts['env_render_calls']+=1;return render0(self,*a,**k)
 def forward(self,*a,**k):
  ret=forward0(self,*a,**k)
  latest['forward_state']=packed(self.model._model,self.data._data);latest['geometry']=geometry(self.data._data)
  return ret
 def setstate(self,*a,**k):counts['state_restores']+=1;return set0(self,*a,**k)
 binding.MjSim.step=step;binding.MjSim.render=rend;binding.MjSim.forward=forward;binding.MjSim.set_state_from_flattened=setstate
 env=None;r=None;rr=None;writers=[];rows=[];started=time.time()
 try:
  bddl=json.loads((src/'environment.json').read_text())['bddl']
  env=OffScreenRenderEnv(bddl_file_name=bddl,camera_heights=256,camera_widths=256);env.seed(0);env.reset();env.set_init_state(initial)
  m=env.sim.model._model;d=env.sim.data._data
  runtime=env.sim.model.get_xml();(out/'runtime_model.xml').write_text(runtime)
  xmlmodel=mujoco.MjModel.from_xml_path(str(out/'runtime_model.xml'))
  model_deltas={key:float(np.max(abs(getattr(m,key)-getattr(xmlmodel,key)))) for key in ['body_pos','body_quat','body_inertia','geom_pos','geom_quat','cam_pos','cam_quat','mesh_vert']}
  mujoco.mj_saveModel(m,str(out/'runtime_model.mjb'),None)
  mr=mujoco.MjModel.from_binary_path(str(out/'runtime_model.mjb'));dr=mujoco.MjData(mr)
  dump(out/'model_serialization.json',dict(method='exact compiled MJB',sha256=sha(out/'runtime_model.mjb'),xml_roundtrip_max_deltas=model_deltas,prior_spatial_capture='XML roundtrip retained as failed evidence; not rerun'))
  # Preserve actual legacy rendering options, not a hand-approximated renderer.
  context=env.sim._render_context_offscreen;o=copy.copy(context.vopt)
  r=mujoco.Renderer(m,256,256);rr=mujoco.Renderer(mr,256,256)
  r.scene.flags[:]=context.scn.flags;rr.scene.flags[:]=context.scn.flags
  dump(out/'render_options.json',dict(geomgroup=o.geomgroup.tolist(),sitegroup=o.sitegroup.tolist(),flags=o.flags.tolist(),render_flags=context.scn.flags.tolist(),
   raw_env_orientation='bottom-up -> reverse both axes',native_orientation='top-down -> horizontal flip',resolution=[256,256],cameras=CAMS,
   mj_state_type='mjSTATE_INTEGRATION',includes=['time','qpos','qvel','act','warmstart','ctrl','applied forces','eq_active','mocap','userdata','plugin'],
   missing_in_legacy=['ctrl','warmstart','applied forces','eq_active','mocap','userdata','derived transforms'],
   source_note='Controller/cache state is not claimed resumable; needed here only for image reproduction, not control continuation'))
  for name in [c+'_image' for c in CAMS]+['robot0_joint_pos','robot0_eef_pos']:
   ob=env.env._observables[name];orig=ob._sensor
   def tracked(cache,orig=orig,name=name):
    value=orig(cache)
    sensor_log[name]=dict(acquisition_time=float(d.time),integration_state=packed(m,d),geometry_state=latest['forward_state'].copy(),geometry=geometry(d))
    return value
   tracked.__modality__=orig.__modality__
   ob._sensor=tracked
  for _ in range(10):
   counts['wait_steps']+=1
   obs,*_=env.step([0,0,0,0,0,0,-1])
  arm=[env.sim.model.get_joint_qpos_addr('robot0_joint'+str(i)) for i in range(1,8)];site=env.sim.model.site_name2id('gripper0_grip_site')
  rlds=np.load(OLD/'rlds_mapping_v2'/suite/'pilot_rlds.npz')
  caps=[cv2.VideoCapture(str(ROOT/'data/libero_converted'/suite/'traj_000010'/f'{c}_rgb.mp4')) for c in ['agentview_camera','wrist_camera']]
  paths=['obs','direct','sync','restored','sensor_geometry'];rgbs={k:[] for k in paths};states=[];fullstates=[];geoms={k:[] for k in ['direct','sync','restored']};times=[];sensor_states=[];sensor_geoms=[];obs_values=[]
  writer=cv2.VideoWriter(str(out/'four_paths_full.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(1536,512));assert writer.isOpened();writers.append(writer)
  fixed=sorted(set(np.linspace(0,len(previous['states'])-1,7,dtype=int)));sheet=Image.new('RGB',(1536,len(fixed)*542));draw=ImageDraw.Draw(sheet)
  for t,raw in enumerate(previous['source_action_indices']):
   before=packed(m,d);st=env.sim.get_state().flatten().copy();states.append(st);fullstates.append(before)
   rgbs['obs'].append(np.stack([obs[c+'_image'][::-1,::-1] for c in CAMS]));rgbs['direct'].append(render(r,d,o));counts['diagnostic_rgb_renders']+=2
   env_direct=np.stack([env.sim.render(camera_name=c,width=256,height=256)[::-1,::-1] for c in CAMS])
   synced=copy.copy(d);mujoco.mj_forward(m,synced);counts['copy_forwards']+=1
   rgbs['sync'].append(render(r,synced,o));counts['diagnostic_rgb_renders']+=2
   # Explicit disk roundtrip of the complete integration state, one frame at a time.
   file=out/'states'/f'{t:06d}.npy';file.parent.mkdir(exist_ok=True);np.save(file,before)
   restore(mr,dr,np.load(file));counts['state_restores']+=1;counts['copy_forwards']+=1
   rgbs['restored'].append(render(rr,dr,o));counts['diagnostic_rgb_renders']+=2
   camera_records=[sensor_log[c+'_image'] for c in CAMS]
   assert np.array_equal(camera_records[0]['geometry_state'],camera_records[1]['geometry_state'])
   sensor_states.append(np.stack([q['integration_state'] for q in camera_records]));sensor_geoms.append(np.stack([q['geometry'] for q in camera_records]))
   sd=mujoco.MjData(mr);restore(mr,sd,camera_records[0]['geometry_state']);counts['state_restores']+=1;counts['copy_forwards']+=1
   rgbs['sensor_geometry'].append(render(rr,sd,o));counts['diagnostic_rgb_renders']+=2
   assert np.array_equal(before,packed(m,d)), 'diagnostic mutated main replay state'
   orig=[]
   for cap in caps:ok,bgr=cap.read();assert ok;orig.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
   orig=np.stack(orig)
   row=dict(frame=t,source_action_index=int(raw),boundary_time=float(d.time),obs_acquisition_time=camera_records[0]['acquisition_time'],obs_geometry_time=float(camera_records[0]['geometry_state'][0]),
    main_previous_state_delta=float(np.max(abs(st-previous['states'][t]))),main_previous_geometry_delta=float(np.max(abs(d.xpos-previous['body_positions'][t]))),
    direct_sync_body_delta=float(np.max(abs(d.xpos-synced.xpos))),direct_sync_camera_delta=float(np.max(abs(d.cam_xpos-synced.cam_xpos))),
    A_geometry_delta=float(np.max(abs(geometry(synced)-geometry(dr)))),A_qpos_delta=float(np.max(abs(synced.qpos-dr.qpos))),A_qvel_delta=float(np.max(abs(synced.qvel-dr.qvel))),
    B_joint_error=float(np.max(abs(synced.qpos[arm]-rlds['joints'][t]))),B_eef_error_m=float(np.max(abs(synced.site_xpos[site]-rlds['state'][t,:3]))),
    obs_joint_error=float(np.max(abs(obs['robot0_joint_pos']-rlds['joints'][t]))))
   row.update(metrics(rgbs['direct'][-1],env_direct,'renderer_equivalence'))
   row.update(metrics(rgbs['sync'][-1],rgbs['restored'][-1],'A'))
   row.update(metrics(rgbs['sync'][-1],orig,'B'))
   row.update(metrics(rgbs['obs'][-1],orig,'obs_original'))
   row.update(metrics(rgbs['obs'][-1],rgbs['sensor_geometry'][-1],'obs_geometry_recovery'))
   row.update(metrics(rgbs['direct'][-1],rgbs['sync'][-1],'forward_effect'))
   for k,data in [('direct',d),('sync',synced),('restored',dr)]:geoms[k].append(geometry(data))
   times.append({k:dict(acquisition_time=v['acquisition_time'],geometry_time=float(v['geometry_state'][0])) for k,v in sensor_log.items()})
   obs_values.append(np.r_[obs['robot0_joint_pos'],obs['robot0_eef_pos']]);rows.append(row)
   panels=[orig]+[rgbs[k][-1] for k in paths];panel=np.concatenate([np.concatenate([p[0] for p in panels],1),np.concatenate([p[1] for p in panels],1)],0)
   writer.write(cv2.cvtColor(panel,cv2.COLOR_RGB2BGR))
   if t in fixed:
    y=fixed.index(t)*542;sheet.paste(Image.fromarray(panel),(0,y+30));draw.text((0,y+5),f'frame={t} | original | cached obs | direct | synchronized | restored | sensor-geometry diagnostic',fill='white')
   obs,*_=env.step(actions[raw].tolist());counts['source_actions']+=1
   if t%25==0:
    dump(out/'progress.json',dict(completed=len(rows),counts=counts));print(suite,t,'A',row['A_wrist_max'],'B',row['B_wrist_p99'],'joint',row['B_joint_error'],flush=True)
  writer.release();writers=[]
  for c in caps:c.release()
  sheet.save(out/'fixed_comparison.jpg');csvout(out/'frame_results.csv',rows);dump(out/'sensor_timing.json',times)
  np.savez_compressed(out/'synchronized.npz',states=np.stack(states),integration_states=np.stack(fullstates),rgb=np.stack(rgbs['sync']),actions=previous['actions'],source_action_indices=previous['source_action_indices'],**{k+'_geometry':np.stack(v) for k,v in geoms.items()})
  np.savez_compressed(out/'diagnostic_paths.npz',**{k:np.stack(v) for k,v in rgbs.items() if k!='sync'},sensor_integration_states=np.stack(sensor_states),sensor_geometry_transforms=np.stack(sensor_geoms),obs_robot=np.stack(obs_values))
  A=all(x['A_geometry_delta']<=1e-12 and all(x[f'A_{c}_max']==0 for c in ['front','wrist']) for x in rows)
  B=all(x['B_joint_error']<=.002 and x['B_eef_error_m']<=.002 and all(x[f'B_{c}_mae']<=8 and x[f'B_{c}_p99']<=40 for c in ['front','wrist']) for x in rows)
  dump(out/'result.json',dict(status='PENDING_VISUAL_REVIEW' if A and B else 'BLOCKED_ORIGINAL_CORRESPONDENCE',A_self_restore=A,B_original_numeric_pixel=B,
    frames=len(rows),counts=counts,seconds=time.time()-started,main_replay_exact_previous=all(x['main_previous_state_delta']==0 for x in rows),
    unified_renderer_exact=all(x[f'renderer_equivalence_{c}_max']==0 for x in rows for c in ['front','wrist']),
    runtime_xml_sha256=sha(out/'runtime_model.xml'),source_checkpoint='none',model_forwards=0,updates=0,backward=0,policy_rollouts=0))
 except BaseException as e:
  dump(out/'failure.json',dict(error=repr(e),counts=counts,completed_frames=len(rows)));raise
 finally:
  for w in writers:w.release()
  if r:r.close()
  if rr:rr.close()
  if env:env.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--suite',choices=SUITES,required=True);main(p.parse_args().suite)
