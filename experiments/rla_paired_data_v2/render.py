"""CPU-only saved-state renderer. No environment step, policy or ML imports."""
import argparse,time
import mujoco
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
from common import *

def setup(path):
 m=mujoco.MjModel.from_xml_path(str(path));m.vis.quality.offsamples=4
 d=mujoco.MjData(m);r=mujoco.Renderer(m,224,224)
 opt=mujoco.MjvOption();opt.geomgroup[0]=0;opt.geomgroup[1]=1;opt.sitegroup[:]=0
 return m,d,r,opt
def restore(m,d,s):
 d.time=s[0];d.qpos[:]=s[1:1+m.nq];d.qvel[:]=s[1+m.nq:];mujoco.mj_forward(m,d)
def frame(r,d,opt,cam,mode='rgb'):
 if mode=='depth':r.enable_depth_rendering()
 elif mode=='seg':r.enable_segmentation_rendering()
 r.update_scene(d,camera=cam,scene_option=opt);a=r.render().copy()[:,::-1]
 if mode=='depth':r.disable_depth_rendering()
 elif mode=='seg':r.disable_segmentation_rendering()
 return a
def labels(m,d,task):
 site=m.site('gripper0_grip_site').id; hand=m.body('robot0_right_hand').id
 fingers=[d.qpos[m.jnt_qposadr[m.joint(n).id]] for n in ['gripper0_finger_joint1','gripper0_finger_joint2']]
 target=m.body('akita_black_bowl_1_main' if task=='stove' else 'wooden_cabinet_1_cabinet_middle').id
 out={'eef':d.site_xpos[site].copy(),'target':d.xpos[target].copy(),'target_quat':d.xquat[target].copy(),
 'proprio':np.r_[d.site_xpos[site],Rotation.from_quat(np.roll(d.xquat[hand],-1)).as_rotvec(),fingers]}
 if task=='stove':out['plate']=d.xpos[m.body('plate_1_main').id].copy()
 else:out['drawer_joint']=np.array(d.qpos[m.jnt_qposadr[m.joint('wooden_cabinet_1_middle_level').id]])
 for c in CAMS:
  cid=m.camera(c).id;f=112/np.tan(np.deg2rad(m.cam_fovy[cid])/2)
  out[c+'_K']=np.array([[-f,0,112],[0,f,112],[0,0,1.]])
  ext=np.eye(4);ext[:3,:3]=d.cam_xmat[cid].reshape(3,3)@np.diag([1,-1,-1]);ext[:3,3]=d.cam_xpos[cid];out[c+'_c2w']=ext
 return out
def mask(m,seg,task):
 root=m.body('akita_black_bowl_1_main' if task=='stove' else 'wooden_cabinet_1_cabinet_middle').id
 bodies={root}
 for i in range(m.nbody):
  if int(m.body_parentid[i]) in bodies:bodies.add(i)
 ids=[i for i in range(m.ngeom) if int(m.geom_bodyid[i]) in bodies]
 return (seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))&np.isin(seg[:,:,0],ids)
PHYS=['body_mass','body_inertia','body_pos','body_quat','geom_pos','geom_quat','geom_size','geom_type','geom_friction','geom_contype','geom_conaffinity','geom_solref','geom_solimp','jnt_type','jnt_pos','jnt_axis','jnt_range','dof_damping','dof_armature','actuator_gainprm','actuator_biasprm','cam_pos','cam_quat','cam_fovy','mesh_vert','mesh_face']
def run(ep,sample):
 start=time.time();src=Path(ep['root']);arr=np.load(src/'source_arrays.npz');states=arr['states'];actions=arr['actions']
 out=ROOT/('sample' if sample else 'rendered')/ep['episode'];out.mkdir(parents=True,exist_ok=False)
 indices=np.unique(np.linspace(0,len(states)-1,5).astype(int)) if sample else np.arange(len(states))
 engines=[setup(src/(v+'.xml')) for v in VARIANTS];base=engines[0][0]
 for m,_,_,_ in engines:
  for k in PHYS:assert np.array_equal(getattr(base,k),getattr(m,k)),k
 rows=[];labs=[];images={v:[] for v in VARIANTS};calls=0;restores=0
 try:
  for t in indices:
   reference=None;rgbs=[];depths=[];segs=[]
   for v,(m,d,r,opt) in zip(VARIANTS,engines):
    restore(m,d,states[t]);restores+=1;before=np.r_[d.time,d.qpos,d.qvel];l=labels(m,d,ep['task'])
    if reference is None:reference=l;labs.append(l)
    else:
     for k in l:assert np.array_equal(reference[k],l[k]),(t,v,k)
    rgb=np.stack([frame(r,d,opt,c) for c in CAMS]);calls+=2;rgbs.append(rgb);images[v].append(rgb)
    if sample:
     depths.append(np.stack([frame(r,d,opt,c,'depth') for c in CAMS]));segs.append(np.stack([frame(r,d,opt,c,'seg') for c in CAMS]));calls+=4
    assert np.array_equal(before,np.r_[d.time,d.qpos,d.qvel]),'render changed state'
   for j,v in enumerate(VARIANTS[1:],1):
    for ci,c in enumerate(CAMS):
     delta=np.abs(rgbs[j][ci].astype(float)-rgbs[0][ci]);row={'timestep':int(t),'variant':v,'camera':c,'rgb_mae':float(delta.mean()),'changed_fraction':float((delta.max(-1)>0).mean()),'mean':float(rgbs[j][ci].mean()),'std':float(rgbs[j][ci].std())}
     if sample:
      roi=mask(base,segs[0][ci],ep['task']);row.update(depth_max_error=float(np.max(np.abs(depths[j][ci]-depths[0][ci]))),seg_equal=bool(np.array_equal(segs[j][ci],segs[0][ci])),target_pixels=int(roi.sum()),target_clean_mean=float(rgbs[0][ci][roi].mean()) if roi.any() else None,target_variant_mean=float(rgbs[j][ci][roi].mean()) if roi.any() else None,target_mae=float(delta[roi].mean()) if roi.any() else None)
      assert row['depth_max_error']<=1e-6 and row['seg_equal']
     rows.append(row)
   if not sample and (int(t)+1)%25==0:print('FRAME',ep['episode'],int(t)+1,len(states),flush=True)
   if sample:
    canvas=Image.new('RGB',(224*5,244*2),'white');draw=ImageDraw.Draw(canvas)
    for j,v in enumerate(VARIANTS):
     for ci,c in enumerate(CAMS):canvas.paste(Image.fromarray(rgbs[j][ci]),(j*224,ci*244+20));draw.text((j*224,ci*244),v+' '+c,fill='black')
    canvas.save(out/f'compare_{t:04d}.png')
  for v in VARIANTS:np.savez_compressed(out/(v+'.npz'),rgb=np.stack(images[v]),timesteps=indices)
  np.savez_compressed(out/'physical.npz',states=states[indices],actions=actions[indices],timesteps=indices,**{k:np.stack([l[k] for l in labs]) for k in labs[0]})
  table(out/'visual_metrics.csv',rows)
  result={'episode':ep['episode'],'sample':sample,'frames':len(indices),'state_restorations':restores,'render_calls':calls,'rgb_images':len(indices)*10,'geometry_exact':True,'physical_compilation_exact':True,'render_did_not_advance_state':True,'seconds':time.time()-start,'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
  if sample:
   result['visual_gate']=all(3<r['mean']<252 and r['std']>3 for r in rows)
   result['target_visibility_gate']=all(r['target_pixels']<4 or r['target_clean_mean']<=3 or max(2,.35*r['target_clean_mean'])<=r['target_variant_mean']<250 for r in rows)
   result['nonzero_effect']={v:any(r['variant']==v and r['rgb_mae']>0 for r in rows) for v in VARIANTS[1:]}
  save(out/'result.json',result);print(json.dumps(result),flush=True)
 finally:
  for _,_,r,_ in engines:r.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['sample','full'],required=True);p.add_argument('--episode');a=p.parse_args();eps=read(ROOT/'source_manifest.json')
 if a.mode=='sample':eps=[next(e for e in eps if e['task']==t) for t in ['stove','drawer']]
 if a.episode:eps=[e for e in eps if e['episode']==a.episode]
 if a.mode=='full':
  for t in ['stove','drawer']:
   e=next(e for e in read(ROOT/'source_manifest.json') if e['task']==t);s=read(ROOT/'sample'/e['episode']/'result.json');assert s['visual_gate'] and s['target_visibility_gate'] and all(s['nonzero_effect'].values())
  assert read(ROOT/'sample_review.json')['approved']
 for e in eps:run(e,a.mode=='sample')
