"""Recover one fixed training episode, no dynamics stepping; CPU OSMesa."""
import os
os.environ.update(MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',CUDA_VISIBLE_DEVICES='')
import argparse,json,csv,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,h5py,mujoco
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--episode');p.add_argument('--split',choices=['train','validation'],default='train');p.add_argument('--selection-manifest',type=Path);a=p.parse_args();o=a.out
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(n,v):
 with (o/n).open('x') as f:json.dump(v,f,indent=2)
selection=json.load(open(a.selection_manifest)) if a.selection_manifest else None
entries=[e for e in selection['episodes'] if e['split']==a.split] if selection else json.load(open(a.project/'experiments/rla_background_invariance/step2_training/split_manifest.json'))['splits'][a.split]
if a.episode:
 entries=[e for e in entries if e['episode_id']==a.episode]
 assert len(entries)==1,'Episode must belong to the requested permitted split'
checked=[]
for e in entries:
 if selection:assert not any(x['source_hdf5']==e['source_hdf5'] and x['source_demo']==e['source_demo'] for x in selection['protected_sources'])
 pair=Path(e['model_dir']) if selection else Path(e['geometry_path']).parent
 with h5py.File(e['source_hdf5'],'r') as f:
  d=f['data'][e['source_demo']];ok=all(k in d for k in ['states','actions','obs']) and all((pair/(v+'.xml')).exists() for v in ['clean','background'])
  checked.append(dict(episode=e['episode_id'],complete=ok))
  if ok:states=d['states'][:];actions=d['actions'][:];raw_ee=d['obs/ee_states'][:];raw_g=d['obs/gripper_states'][:];raw_j=d['obs/joint_states'][:] if selection else None;break
else:raise RuntimeError('No restorable train trajectory')
assert states.ndim==2 and len(states)==len(actions) and actions.shape[1]==7
N=len(states);assert np.isfinite(states).all() and np.isfinite(actions).all();assert np.allclose(np.diff(states[:,0]),.05,atol=1e-9,rtol=0)
if selection:
 assert hashlib.sha256(states.tobytes()).hexdigest()==e['source_states_sha256']
 assert hashlib.sha256(actions.astype(np.float32).tobytes()).hexdigest()==e['source_actions_sha256']
models=[mujoco.MjModel.from_xml_path(str(pair/(v+'.xml'))) for v in ['clean','background']];data=[mujoco.MjData(m) for m in models]
robot_fk=mujoco.MjData(models[0]) if selection else None
changes=[]
for x,y in zip(ET.parse(pair/'clean.xml').getroot().iter(),ET.parse(pair/'background.xml').getroot().iter()):
 assert x.tag==y.tag
 if x.attrib!=y.attrib:
  assert x.tag=='texture' and x.get('name') in ['tex-table','tex-wall'] and {k for k in set(x.attrib)|set(y.attrib) if x.get(k)!=y.get(k)}=={'file'}
  changes.append(dict(name=x.get('name'),clean_sha256=sha(x.get('file')),background_sha256=sha(y.get('file'))))
assert changes
for field in ['body_pos','body_quat','geom_pos','geom_quat','geom_rgba','geom_matid','geom_contype','geom_conaffinity','mat_rgba','mat_texid','mat_reflectance','mat_specular','mat_shininess','cam_fovy','cam_pos','cam_quat','light_pos','light_dir','light_diffuse','light_specular','light_ambient']:
 np.testing.assert_array_equal(getattr(models[0],field),getattr(models[1],field))
render=[mujoco.Renderer(m,224,224) for m in models];cams=['agentview','robot0_eye_in_hand'];opt=mujoco.MjvOption();opt.geomgroup[0]=0;opt.geomgroup[1]=1;opt.sitegroup[:]=0
for name in ['frames','projection_overlays']: (o/name).mkdir()
rows=[];prop=[];geomtests=[];Ks=[];Cs=[];fk=[];hashes={str(pair/(v+'.xml')):sha(pair/(v+'.xml')) for v in ['clean','background']}
if not selection:hashes[e['source_hdf5']]=sha(e['source_hdf5'])
def frame(j,cam,kind):
 r=render[j];r.disable_depth_rendering();r.disable_segmentation_rendering()
 if kind=='seg':r.enable_segmentation_rendering()
 if kind=='depth':r.enable_depth_rendering()
 r.update_scene(data[j],camera=cam,scene_option=opt);r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=1
 assert models[j].vis.quality.offsamples==4
 return r.render().copy()[:,::-1].copy() # modern top-down renderer -> official mirrored camera convention
for t,s in enumerate(states):
 for m,d in zip(models,data):
  assert m.na==0 and len(s)==1+m.nq+m.nv
  d.time=s[0];d.qpos[:]=s[1:1+m.nq];d.qvel[:]=s[1+m.nq:];mujoco.mj_forward(m,d)
 for name in ['qpos','qvel','xpos','xquat','cam_xpos','cam_xmat']:np.testing.assert_array_equal(getattr(data[0],name),getattr(data[1],name))
 m,d=models[0],data[0];site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'gripper0_grip_site');hand=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot0_right_hand');fingers=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'gripper0_finger_joint'+str(i)) for i in [1,2]]
 assert min(site,hand,*fingers)>=0
 proprio=np.r_[d.site_xpos[site],Rotation.from_quat(np.roll(d.xquat[hand],-1)).as_rotvec(),d.qpos[m.jnt_qposadr[fingers]]].astype(np.float32);fk.append(proprio)
 if selection:
  arm=[m.joint('robot0_joint'+str(i)).qposadr[0] for i in range(1,8)];finger_addr=m.jnt_qposadr[fingers];addresses=arm+finger_addr.tolist()
  mujoco.mj_resetData(m,robot_fk);robot_fk.qpos[addresses]=d.qpos[addresses];mujoco.mj_forward(m,robot_fk)
  robot_proprio=np.r_[robot_fk.site_xpos[site],Rotation.from_quat(np.roll(robot_fk.xquat[hand],-1)).as_rotvec(),robot_fk.qpos[finger_addr]].astype(np.float32)
  np.testing.assert_array_equal(proprio,robot_proprio)
  if t:
   np.testing.assert_allclose(d.qpos[arm],raw_j[t-1],rtol=0,atol=1e-8)
   np.testing.assert_allclose(d.qpos[finger_addr],raw_g[t-1],rtol=0,atol=1e-8)
 diff=proprio-np.r_[raw_ee[t-1],raw_g[t-1]] if t else np.full(8,np.nan)
 prop.append(dict(source_state=t,raw_obs=t-1 if t else '',xyz_max_m=float(abs(diff[:3]).max()) if t else '',rot_max_rad=float(abs(diff[3:6]).max()) if t else '',finger_max_m=float(abs(diff[6:]).max()) if t else '',chosen='same-state FK',external_pose_truth=False))
 imgs={v:[] for v in ['clean','background']};kk=[];cc=[]
 for ci,cam in enumerate(cams):
  cid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_CAMERA,cam);f=112/np.tan(np.deg2rad(m.cam_fovy[cid])/2);K=np.array([[-f,0,112],[0,f,112],[0,0,1.]])
  C=np.eye(4);C[:3,:3]=d.cam_xmat[cid].reshape(3,3)@np.diag([1,-1,-1]);C[:3,3]=d.cam_xpos[cid];kk.append(K);cc.append(C)
  rgb=[frame(j,cam,'rgb') for j in [0,1]];seg=[frame(j,cam,'seg') for j in [0,1]];dep=[frame(j,cam,'depth') for j in [0,1]]
  assert np.array_equal(*seg) and np.array_equal(*dep) and np.isfinite(dep).all()
  if t in [0,8,N-1]:assert np.array_equal(rgb[0],frame(0,cam,'rgb'))
  delta=abs(rgb[0].astype(float)-rgb[1].astype(float));rows.append(dict(timestep=t,camera=cam,seg_equal=True,depth_equal=True,rgb_MAE=float(delta.mean()),changed_pixels=int((delta.max(2)>0).sum())))
  for j,v in enumerate(imgs):imgs[v].append(rgb[j].transpose(2,0,1))
  # Independent rasterization check: visible surface points from metric depth,
  # then isolated marker geometry rendered by MuJoCo (not drawn at projected pixels).
  if t in [0,8,N-1]:
   preview=Image.fromarray(rgb[0]);draw=ImageDraw.Draw(preview)
   for row,col in [(80,80),(112,112),(144,144)]:
    z=float(dep[0][row,col]);q=np.linalg.inv(K)@np.array([col+.5,row+.5,1.]);point=C[:3,:3]@(q*z*.98)+C[:3,3]
    uv=K@(C[:3,:3].T@(point-C[:3,3]));uv=uv[:2]/uv[2]
    r=render[0];r.disable_depth_rendering();r.disable_segmentation_rendering();r.update_scene(d,camera=cam,scene_option=opt);r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=1
    g=r.scene.geoms[r.scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([z/f*2]*3),point,np.eye(3).ravel(),np.array([1,0,1,1.]));g.emission=1;r.scene.ngeom+=1
    marker=r.render()[:,::-1].copy();mask=(marker[:,:,0]>200)&(marker[:,:,2]>200)&(marker[:,:,1]<70)
    yy,xx=np.where(mask & (abs(np.indices(mask.shape)[0]-row)<8)&(abs(np.indices(mask.shape)[1]-col)<8))
    assert len(xx)>0,(t,cam,row,col)
    measured=np.array([xx.mean()+.5,yy.mean()+.5]);err=float(np.linalg.norm(measured-uv));assert err<=2.,(t,cam,err)
    geomtests.append(dict(t=t,camera=cam,world=point.tolist(),projected=uv.tolist(),rendered_centroid=measured.tolist(),pixel_error=err))
    draw.ellipse((uv[0]-3,uv[1]-3,uv[0]+3,uv[1]+3),outline='magenta')
   preview.save(o/'projection_overlays'/f'{t}_{cam}.png')
 Ks.append(kk);Cs.append(cc)
 np.testing.assert_array_equal(np.r_[d.time,d.qpos,d.qvel],s)
 np.savez_compressed(o/'frames'/f'{t:04}.npz',**{v:np.stack(x) for v,x in imgs.items()},K=np.array(kk),c2w=np.array(cc),state=s,proprio=proprio,action=actions[t])
 if t==16:js('prefix_acceptance.json',dict(frames=17,valid_chunks=9,status='PASS'))
 print('frame',t,'/',N,flush=True)
for r in render:r.close()
assert all(sha(p)==h for p,h in hashes.items())
np.savez_compressed(o/'trajectory_labels.npz',states=states,actions=actions,proprio=np.array(fk),K=np.array(Ks),c2w=np.array(Cs),valid_horizon_mask=np.arange(N)[:,None]+np.arange(1,9)[None,:]<N)
for n,records in [('proprio_alignment_audit.csv',prop),('regional_sequence_metrics.csv',rows)]:
 with (o/n).open('x') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
js('trajectory_provenance.json',dict(selected=e['episode_id'],checked=checked,source_hdf5=e['source_hdf5'],source_demo=e['source_demo'],instruction=e['task'],frames=N,valid_full_chunks=N-8,split=a.split,data_version='dual224_fk_v2',source_hashes=hashes,derived=True,temporal_definition='state[s] is pre-action state; action[s:s+8] reaches state[s+8]; raw obs[s-1] nominal post previous action with unknown sensor phase',RLDS_mapping='No cross-source identity inferred; explicit HDF5 source identity preserved; RLDS match remains unknown',optimizer_updates=0,backward=0,rollouts=0))
js('paired_sequence_manifest.json',dict(episode=e['episode_id'],frames=N,conditions=['clean','background'],cameras=cams,resolution=224,reflection=1,MSAA=4,texture_changes=changes,valid_full_chunks=N-8,tail='stored clamped official targets must carry false validity for missing futures',frames_pattern='frames/{timestep:04}.npz',identity=['source_episode','source_timestep','appearance_id','dual224_fk_v2']))
js('camera_geometry_audit.json',dict(status='PASS_RASTER_PROJECTION',tests=geomtests,max_projection_error_px=max(x['pixel_error'] for x in geomtests),method='Surface points unprojected from depth; isolated visual marker rasterized independently by MuJoCo; never stored in formal RGB',pixel_centers='index+0.5',orientation='horizontal mirror of modern top-down Renderer output; K fx negative, cx=W/2',K_index='[t,camera,:,:]',extrinsics='camera-to-world OpenCV; wrist recomputed every state',wrist_motion_max=float(np.max(np.abs(np.array(Cs)[:,1]-np.array(Cs)[0,1]))),external_pose_truth=False))
