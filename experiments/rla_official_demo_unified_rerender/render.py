"""Direct HDF5 static-state render. No integration, controllers or neural nets."""
from common import *
import argparse, copy, time, shutil, xml.etree.ElementTree as ET
import h5py, mujoco, cv2
from PIL import Image, ImageDraw

def remap(xml,out):
 root=ET.fromstring(xml);assets=[]
 roots=[OLD/'libero_official/libero/libero/assets',Path('/data/lyc/projects/vla-adapter-ref/.venv-cpu/lib/python3.10/site-packages/robosuite/models/assets')]
 for e in root.find('asset'):
  old=e.get('file')
  if old is None:continue
  assert '/assets/' in old,old
  tail=os.path.normpath(old.split('/assets/',1)[1]);assert not tail.startswith('../')
  possibilities=[p/tail for p in roots if (p/tail).is_file()]
  assert possibilities,('Missing logical asset',old)
  assert len({sha(p) for p in possibilities})==1,('Ambiguous logical asset',old)
  p=possibilities[0];e.set('file',str(p));assets.append(dict(kind=e.tag,name=e.get('name'),logical_asset=tail,path=str(p),sha256=sha(p),historical_bytes='not recorded; pinned official current asset'))
 ET.ElementTree(root).write(out/'source_local_assets.xml',encoding='unicode')
 dump(out/'assets.json',assets)
 return root

def augment(m,out):
 names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) for i in range(m.ngeom)]
 selected=[i for i,n in enumerate(names) if n in WHITELIST];assert selected,'No approved background'
 tids={};aliases=[]
 for gid in selected:
  mat=int(m.geom_matid[gid]);assert mat>=0
  tid=int(m.mat_texid[mat,1]);assert tid>=0,('Background lacks RGB texture',names[gid])
  users=[i for i in range(m.ngeom) if m.geom_matid[i]>=0 and tid in m.mat_texid[m.geom_matid[i]]]
  assert set(users)<=set(selected),('Foreground texture alias; isolate before admission',tid,[names[i] for i in users])
  tids[tid]=users;aliases.append(dict(geom=names[gid],material_id=mat,texture_id=tid,all_users=[names[i] for i in users]))
 a=copy.copy(m);textures=[]
 for tid,users in tids.items():
  start=int(m.tex_adr[tid]);h,w,c=map(int,[m.tex_height[tid],m.tex_width[tid],m.tex_nchannel[tid]]);assert c==3
  rgb=m.tex_data[start:start+h*w*c].reshape(h,w,c);hsv=cv2.cvtColor(rgb.astype('float32')/255,cv2.COLOR_RGB2HSV);before=hsv.copy()
  hsv[...,0]=(hsv[...,0]+72)%360;hsv[...,1]+=.35*(1-hsv[...,1])
  changed=np.rint(np.clip(cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB),0,1)*255).astype('uint8');a.tex_data[start:start+h*w*c]=changed.ravel()
  textures.append(dict(texture_id=tid,before_sha256=digest(rgb),after_sha256=digest(changed),value_quantization_max=float(np.max(abs(changed.max(2)/255-before[...,2]))),source_low_saturation_fraction=float((before[...,1]<.05).mean())))
 exact=[]
 for key in dir(m):
  val=getattr(m,key)
  if isinstance(val,np.ndarray) and key!='tex_data':assert val.tobytes()==getattr(a,key).tobytes(),key;exact.append(key)
 dump(out/'appearance.json',dict(delta_h=.2,saturation_gain=.35,texture_records=textures,alias_audit=aliases,background_geoms=[names[i] for i in selected],all_other_model_arrays_exact=exact,reflection_lighting_preserved=True,shared_foreground_aliases=0))
 return a,selected

def geom(d):return np.r_[d.xpos.ravel(),d.xquat.ravel(),d.cam_xpos.ravel(),d.cam_xmat.ravel(),d.site_xpos.ravel()]
def restore(m,d,s,counts):
 mujoco.mj_resetData(m,d);d.time=s[0];d.qpos[:]=s[1:1+m.nq];d.qvel[:]=s[1+m.nq:];mujoco.mj_forward(m,d)
 # Pinned LIBERO FlatStove.turn_on -> ObjectState.update_state -> set_visualization.
 # This is a deterministic render-state update, NOT physics integration or a light change.
 for sid in range(m.nsite):
  name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_SITE,sid)
  if name and name.startswith('flat_stove_') and name.endswith('_burner'):
   joint=name.removesuffix('_burner')+'_button';jid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,joint);assert jid>=0
   visible=d.qpos[m.jnt_qposadr[jid]]>=.5;alpha=float(m.site_rgba[sid,3])
   if (alpha<=0 and visible) or (alpha>0 and not visible):m.site_rgba[sid,3]=1-alpha
 counts['state_restores']+=1;counts['kinematic_forwards']+=1
def rgb(r,d,o,counts):
 images=[]
 for c in CAMS:
  r.update_scene(d,camera=c,scene_option=o);images.append(r.render().copy()[:,::-1]);counts['rgb_renders']+=1
 return np.stack(images)

def main(suite,tid,manifest=None):
 row=next(x for x in csv.DictReader(open(manifest or RUN/'source_manifest.csv')) if x['suite']==suite and x['trajectory_id']==tid)
 assert row['source_status']=='MULTI_EVIDENCE_JOIN' and row['split_conflict']=='False',row
 out=RUN/'trajectories'/suite/f'traj_{tid}'
 if (out/'result.json').exists():
  result=json.loads((out/'result.json').read_text());assert result['status']=='NUMERIC_PASS_PENDING_VISUAL' or result['status']=='PASS';print('ALREADY_COMPLETE',suite,tid);return
 assert not out.exists(),f'Incomplete prior attempt preserved: {out}; explicit recovery required'
 assert shutil.disk_usage(RUN).free>100*2**30
 out.mkdir(parents=True);started=time.time();counts=dict(state_restores=0,kinematic_forwards=0,rgb_renders=0,segmentation_renders=0,depth_renders=0,physics_steps=0,action_replay=0)
 engines={};writers=[]
 try:
  with h5py.File(row['source_file']) as f:
   d=f['data'][row['demo']];xml=str(d.attrs['model_file']);rawstates=d['states'][:];rawactions=d['actions'][:];ids=np.array(json.loads(row['filtered_indices']),dtype=int);states=rawstates[ids];actions=rawactions[ids]
   source_meta={k:str(v) for k,v in f['data'].attrs.items()}
  assert np.array_equal(filtered(rawactions),ids);assert len(states)==int(row['frames'])
  assert np.isfinite(states).all() and np.isfinite(actions).all() and np.all(np.diff(rawstates[:,0])>0)
  (out/'original_source.xml').write_text(xml);root=remap(xml,out)
  m=mujoco.MjModel.from_xml_path(str(out/'source_local_assets.xml'));assert m.na==0 and states.shape[1]==1+m.nq+m.nv
  assert m.nmocap==0 and m.nplugin==0,'Unrecorded rendering-relevant dynamic state requires explicit audit'
  xmljoints=[e.get('name') for e in root.find('worldbody').iter('joint')]
  modeljoints=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(m.njnt)];assert xmljoints==modeljoints
  # MJB preserves the compiled constants exactly; never round-trip compiled model through XML.
  mujoco.mj_saveModel(m,str(out/'clean_model.mjb'),None);reload=mujoco.MjModel.from_binary_path(str(out/'clean_model.mjb'));aug,bgids=augment(m,out)
  mujoco.mj_saveModel(aug,str(out/'background_model.mjb'),None)
  for key in dir(m):
   v=getattr(m,key)
   if isinstance(v,np.ndarray):assert v.tobytes()==getattr(reload,key).tobytes(),key
  opts=json.loads((REPAIR/'capture'/suite/'render_options.json').read_text());o=mujoco.MjvOption();o.geomgroup[:]=opts['geomgroup'];o.sitegroup[:]=opts['sitegroup'];o.flags[:]=opts['flags']
  for name,model in [('clean',m),('background',aug),('reload',reload)]:
   r=mujoco.Renderer(model,256,256);r.scene.flags[:]=opts['render_flags'];engines[name]=(model,mujoco.MjData(model),r)
  armnames=['robot0_joint'+str(i) for i in range(1,8)]+['gripper0_finger_joint1','gripper0_finger_joint2']
  qids=[m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in armnames]
  assert all(mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)>=0 for n in armnames)
  site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'gripper0_grip_site');assert site>=0
  cameraids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_CAMERA,c) for c in CAMS];assert min(cameraids)>=0
  n=len(states);fixed=sorted(set(np.linspace(0,n-1,7,dtype=int)));sheet=Image.new('RGB',(1024,len(fixed)*540),'#202020');draw=ImageDraw.Draw(sheet)
  writer=cv2.VideoWriter(str(out/'full_comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(1024,512));assert writer.isOpened();writers.append(writer)
  original=ROOT/'data/libero_converted'/suite/f'traj_{tid}';caps=[cv2.VideoCapture(str(original/(s+'_rgb.mp4'))) for s in STREAMS]
  for appearance in ['clean','background']:
   for stream in STREAMS:(out/appearance/(stream+'_rgb')).mkdir(parents=True)
  frame_rows=[];color_rows=[];qpos=[];eef=[];cpos=[];crot=[];visible=[];depthchecks=[]
  for t,s in enumerate(states):
   images={};geometry={}
   for name,(model,data,r) in engines.items():
    restore(model,data,s,counts);before=np.r_[data.time,data.qpos,data.qvel].copy();images[name]=rgb(r,data,o,counts)
    assert np.array_equal(before,np.r_[data.time,data.qpos,data.qvel]);geometry[name]=geom(data)
   assert np.array_equal(images['clean'],images['reload']),('self RGB',t)
   assert max(np.max(abs(geometry[k]-geometry['clean'])) for k in geometry)<=1e-12
   for model,_,_ in engines.values():assert np.array_equal(model.site_rgba,m.site_rgba)
   data=engines['clean'][1];r=engines['clean'][2];assert np.array_equal(np.r_[data.time,data.qpos,data.qvel],s)
   masks=[];vcounts=[]
   for ci,cam in enumerate(CAMS):
    r.enable_segmentation_rendering();r.update_scene(data,camera=cam,scene_option=o);seg=r.render().copy()[:,::-1];r.disable_segmentation_rendering();counts['segmentation_renders']+=1
    mask=(seg[...,1]==int(mujoco.mjtObj.mjOBJ_GEOM))&np.isin(seg[...,0],bgids);masks.append(mask)
    vv={str(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(g))):int(((seg[...,1]==int(mujoco.mjtObj.mjOBJ_GEOM))&(seg[...,0]==g)).sum()) for g in np.unique(seg[...,0][seg[...,1]==int(mujoco.mjtObj.mjOBJ_GEOM)])};vcounts.append(vv)
    diff=images['background'][ci].astype(float)-images['clean'][ci];lum=diff@np.array([.2126,.7152,.0722]);area=int(mask.sum());mae=float(abs(diff[mask]).mean()) if area else None
    color_rows.append(dict(frame=t,camera=cam,background_pixels=area,background_fraction=area/65536,background_rgb_mae=mae,outside_rgb_mae=float(abs(diff[~mask]).mean()) if (~mask).any() else None,visible_but_weak=bool(area and mae<1),luminance_signed_mean=float(lum.mean()),luminance_absolute_mean=float(abs(lum).mean()),changed_pixels=int((abs(diff).max(2)>1).sum())))
   visible.append(dict(frame=t,cameras=dict(zip(CAMS,vcounts))))
   # Geometry is checked at every frame; depth and segmentation image equality at fixed checkpoints.
   if t in fixed:
    for ci,cam in enumerate(CAMS):
     depths=[]
     for name in ['clean','background']:
      _,dd,rr=engines[name];rr.enable_depth_rendering();rr.update_scene(dd,camera=cam,scene_option=o);depths.append(rr.render().copy());rr.disable_depth_rendering();counts['depth_renders']+=1
     delta=float(np.max(abs(depths[0]-depths[1])));assert delta<=1e-6;depthchecks.append(dict(frame=t,camera=cam,depth_max_m=delta))
   originals=[]
   for cap in caps:
    ok,bgr=cap.read();assert ok,'Original reference video missing frame';originals.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
   originals=np.stack(originals);delta=abs(originals.astype(float)-images['clean'])
   frame_rows.append(dict(frame=t,source_index=int(ids[t]),source_time_s=float(s[0]),dt_from_previous_s=float(s[0]-states[t-1,0]) if t else '',state_sha256=digest(s),action_sha256=digest(actions[t]),self_rgb_exact=True,geometry_exact=True,original_front_mae=float(delta[0].mean()),original_wrist_mae=float(delta[1].mean()),original_comparison_role='descriptive_only'))
   for appearance in ['clean','background']:
    for ci,stream in enumerate(STREAMS):Image.fromarray(images[appearance][ci]).save(out/appearance/(stream+'_rgb')/f'{t:06d}.png')
   overlay=images['clean'].copy()
   for ci in range(2):overlay[ci][masks[ci]]=(.4*overlay[ci][masks[ci]]+.6*np.array([255,0,255])).astype('uint8')
   panel=np.concatenate([np.concatenate([originals[i],images['clean'][i],overlay[i],images['background'][i]],1) for i in range(2)],0);writer.write(cv2.cvtColor(panel,cv2.COLOR_RGB2BGR))
   if t in fixed:
    y=fixed.index(t)*540;draw.text((0,y+3),f'{suite}/{tid} frame={t} raw={ids[t]} time={s[0]:.4f}: old video | new Clean | BG mask | H+S',fill='white');sheet.paste(Image.fromarray(panel),(0,y+28))
   qpos.append(data.qpos[qids].copy());quat=np.empty(4);mujoco.mju_mat2Quat(quat,data.site_xmat[site]);eef.append(np.r_[data.site_xpos[site],quat]);cpos.append(data.cam_xpos.copy());crot.append(data.cam_xmat.copy())
   dump(out/'progress.json',dict(frame=t,frames_total=n,counts=counts,seconds=time.time()-started,status='RUNNING'))
   if t%50==0:print(suite,tid,t,n,flush=True)
  writer.release();writers=[]
  for cap in caps:cap.release()
  sheet.save(out/'fixed_comparison.jpg');csvout(out/'frame_mapping.csv',frame_rows);csvout(out/'background_metrics.csv',color_rows);dump(out/'visible_geoms.json',visible)
  np.savez_compressed(out/'source_state_geometry.npz',states=states,actions=actions,source_indices=ids,qpos=np.array(qpos),eef_pose=np.array(eef),camera_positions=np.array(cpos),camera_rotations=np.array(crot))
  dump(out/'static_state_audit.json',dict(source_file=row['source_file'],source_sha256=row['source_sha256'],demo=row['demo'],xml_sha256=sha(out/'original_source.xml'),raw_state_count=len(rawstates),selected_state_count=n,joint_names=modeljoints,na=m.na,nmocap=m.nmocap,nplugin=m.nplugin,visual_state_rule='Pinned FlatStove: burner visible iff button qpos >= 0.5; official alpha toggle applied identically to all model copies at each state; canonical MJB is a template requiring this rule',
   source_state_time_start=float(states[0,0]),source_state_time_end=float(states[-1,0]),raw_dt_min=float(np.diff(rawstates[:,0]).min()),raw_dt_max=float(np.diff(rawstates[:,0]).max()),filtered_dt_min=float(np.diff(states[:,0]).min()),filtered_dt_max=float(np.diff(states[:,0]).max()),filtered_gap_count=int((np.diff(ids)!=1).sum()),
   source_metadata=source_meta,missing_controller_history='not required for this static geometric render; no dynamics continuation claimed',depth_checks=depthchecks,original_video_exactness_not_required=True,old_failure_unchanged=True))
  export_loader(out,row,m,np.array(qpos),np.array(eef),np.array(cpos),np.array(crot),actions,ids,states[:,0])
  dump(out/'result.json',dict(status='NUMERIC_PASS_PENDING_VISUAL' if tid=='000010' else 'PASS',suite=suite,trajectory_id=tid,split=row['split'],source_demo=row['demo'],frames=n,counts=counts,seconds=time.time()-started,
   clean_model_sha256=sha(out/'clean_model.mjb'),background_model_sha256=sha(out/'background_model.mjb'),state_cache_sha256=sha(out/'source_state_geometry.npz'),weak_camera_frames=sum(x['visible_but_weak'] for x in color_rows),invisible_camera_frames=sum(x['background_pixels']==0 for x in color_rows),learned_forwards=0,updates=0,backward=0,policy_rollouts=0))
 except BaseException as e:
  dump(out/'failure.json',dict(error=repr(e),counts=counts,seconds=time.time()-started));raise
 finally:
  for w in writers:w.release()
  for _,_,r in engines.values():r.close()

def export_loader(out,row,m,qpos,eef,cpos,crot,actions,ids,times):
 """Rebuild all geometric fields; do not copy stale author-camera/proprio arrays."""
 n=len(qpos);original=ROOT/'data/libero_converted'/row['suite']/f"traj_{row['trajectory_id']}";values=dict(qpos=qpos.astype('float32'),target_qpos=np.r_[qpos[1:],qpos[-1:]].astype('float32'),eef_pose=eef.astype('float32'),action=actions.astype('float32'))
 cameras=[]
 for cam,stream in zip(CAMS,STREAMS):
  cid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_CAMERA,cam);f=128/np.tan(m.cam_fovy[cid]*np.pi/360);K=np.array([[-f,0,128],[0,f,128],[0,0,1.]])
  extr=np.concatenate([crot[:,cid].reshape(n,3,3)@np.diag([1.,-1.,-1.]),cpos[:,cid,:,None]],axis=2)
  values[stream+'_intrinsics']=np.tile(K,(n,1,1,1)).astype('float32');values[stream+'_extrinsics']=extr[:,None].astype('float32')
  cameras.append(dict(name=cam,stream=stream,K=K.tolist(),c2w='camera_rotations[:,id] @ diag(1,-1,-1) and camera_positions',model_id=cid,dynamic=cam=='robot0_eye_in_hand'))
 for appearance in ['clean','background']:
  dest=RUN/'official_loader'/appearance/row['suite'];dest.mkdir(parents=True,exist_ok=True);traj=dest/f"traj_{row['trajectory_id']}";traj.mkdir(exist_ok=False)
  with h5py.File(original/'metadata.h5') as src,h5py.File(traj/'metadata.h5','w') as h:
   for k,v in src.attrs.items():h.attrs[k]=v
   for k in src:
    if k in values:h.create_dataset(k,data=values[k])
    else:src.copy(k,h)
   h.create_dataset('source_frame_index',data=ids);h.create_dataset('source_time_s',data=times)
   assert np.array_equal(h['action'][:],src['action'][:])
   h.attrs['data_identity']='official_demo_unified_rerender';h.attrs['source_demo']=row['demo'];h.attrs['source_sha256']=row['source_sha256']
  for stream in STREAMS:(traj/(stream+'_rgb')).symlink_to(out/appearance/(stream+'_rgb'),target_is_directory=True)
  # Official all-white foreground placeholders retained solely for loader compatibility, not segmentation.
  for f in original.glob('*foreground_mask*'):(traj/f.name).symlink_to(f)
  if not (dest/'metadata.json').exists():shutil.copy2(original.parent/'metadata.json',dest/'metadata.json')
 dump(out/'camera_time.json',dict(cameras=cameras,resolution=[256,256],source_times=times.tolist(),source_indices=ids.tolist(),future_indices='t,t+h-1; t+h<N',horizon_is='retained frame count, NOT uniform physical duration',last_target_qpos='legacy loader achieved-qpos field only; not used as extrapolated future RGB',root_poses='retained official identity placeholder, NOT measured robot base pose; not used to derive camera or FK',success_attribute='original source label, no new policy or success evaluation',camera_geometry_source='per-demo HDF5 XML and synchronized source state'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--suite',required=True,choices=SUITES);p.add_argument('--id',default='000010');p.add_argument('--manifest');a=p.parse_args();main(a.suite,a.id,a.manifest)
