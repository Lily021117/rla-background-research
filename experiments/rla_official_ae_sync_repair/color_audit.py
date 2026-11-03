"""Complete same-state diagnostic renders; NOT author-data admission.

Canonical compiled model is saved to MJB (XML export may round constants).
All original candidate frames are kept, including original-correspondence failures.
"""
from common import *
import argparse,copy,time,xml.etree.ElementTree as ET,cv2
from PIL import Image,ImageDraw

PHYS=['body_mass','body_inertia','body_pos','body_quat','geom_pos','geom_quat','geom_size','geom_type','geom_friction','geom_contype','geom_conaffinity','geom_solref','geom_solimp','jnt_type','jnt_pos','jnt_axis','jnt_range','dof_damping','dof_armature','actuator_gainprm','actuator_biasprm','cam_pos','cam_quat','cam_fovy','mesh_vert','mesh_face','light_pos','light_dir','light_diffuse','light_specular','light_ambient']
def prepare(suite,out):
 src=OLD/'pilot_replay'/suite/'runtime_model.xml';protocol=json.loads((RUN/'frozen_protocol.json').read_text());allowed=set(protocol['background_whitelist'])
 old=json.loads((OLD/'paired_prepare'/suite/'background_whitelist.json').read_text())
 assets=[];geoms=[]
 for mode in ['expanded_h','expanded_hs']:
  root=ET.parse(src).getroot();asset=root.find('asset');mats={x.get('name'):x for x in asset.findall('material')};textures={x.get('name'):x for x in asset.findall('texture')};done={}
  selected=[g for g in root.iter('geom') if g.get('name') in allowed]
  for geom in selected:
   name=geom.get('name');oldmat=geom.get('material');assert oldmat in mats
   if oldmat not in done:
    mat=copy.deepcopy(mats[oldmat]);texname=mat.get('texture');assert texname in textures
    tex=copy.deepcopy(textures[texname]);source=Path(tex.get('file'));rgb=np.array(Image.open(source).convert('RGB'));hsv=cv2.cvtColor(rgb.astype('float32')/255,cv2.COLOR_RGB2HSV)
    h0=hsv.copy();delta=old['delta_h'] if mode=='expanded_h' else protocol['new_color']['delta_h'];hsv[...,0]=(hsv[...,0]+360*delta)%360
    if mode=='expanded_hs':hsv[...,1]+=.35*(1-hsv[...,1])
    changed=np.rint(np.clip(cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB),0,1)*255).astype('uint8')
    path=out/f'{mode}_{oldmat}.png';Image.fromarray(changed).save(path)
    matname=f'{mode}_{oldmat}';newtex=f'{mode}_{texname}_{oldmat}';mat.set('name',matname);mat.set('texture',newtex);tex.set('name',newtex);tex.set('file',str(path));asset.append(mat);asset.append(tex);done[oldmat]=matname
    assets.append(dict(mode=mode,source=str(source),source_sha256=sha(source),changed_sha256=sha(path),source_material=oldmat,new_material=matname,delta_h=delta,saturation_gain=.35 if mode=='expanded_hs' else 0,
     low_saturation_fraction=float((h0[...,1]<.05).mean()),texture_value_quantization_max=float(np.max(abs(changed.max(2).astype(float)/255-h0[...,2])))))
   geom.set('material',done[oldmat])
  ET.ElementTree(root).write(out/f'{mode}.xml',encoding='unicode');geoms=[g.get('name') for g in selected]
 clean=mujoco.MjModel.from_xml_path(str(src));models={k:mujoco.MjModel.from_xml_path(str(out/f'{k}.xml')) for k in ['expanded_h','expanded_hs']}
 for mode,m in models.items():
  for field in PHYS:assert np.array_equal(getattr(clean,field),getattr(m,field)),field
  changed=[mujoco.mj_id2name(clean,mujoco.mjtObj.mjOBJ_GEOM,i) for i in range(clean.ngeom) if clean.geom_matid[i]!=m.geom_matid[i]];assert set(changed)==set(geoms)
  for g in geoms:
   i=mujoco.mj_name2id(clean,mujoco.mjtObj.mjOBJ_GEOM,g);a=int(clean.geom_matid[i]);b=int(m.geom_matid[i])
   for field in ['mat_specular','mat_shininess','mat_reflectance','mat_rgba','mat_texrepeat','mat_texuniform','mat_emission']:
    assert np.array_equal(getattr(clean,field)[a],getattr(m,field)[b]),field
 dump(out/'appearance_configuration.json',dict(assets=assets,changed_geoms=geoms,old_whitelist=old['changed_geoms'],newly_included=['floor'],excluded_operable_objects=True,
  physical_fields_exact=PHYS,reflectance_and_shadows_unchanged=True,material_references_isolated=True,old_results=str(OLD/'paired_prepare'/suite),diagnostic_only=True))
 return clean,models,geoms,old

def prepare_exact(suite,out):
 """Mutate texture bytes on isolated MJB copies, with fail-closed alias audit."""
 snapshot=(RUN/'capture'/suite/'runtime_model.mjb') if suite in ['libero_goal','libero_10'] else RUN/'model_snapshot'/suite/'model.mjb'
 clean=mujoco.MjModel.from_binary_path(str(snapshot));protocol=json.loads((RUN/'frozen_protocol.json').read_text())
 old=json.loads((OLD/'paired_prepare'/suite/'background_whitelist.json').read_text());allowed=set(protocol['background_whitelist'])
 names=[mujoco.mj_id2name(clean,mujoco.mjtObj.mjOBJ_GEOM,i) for i in range(clean.ngeom) if mujoco.mj_id2name(clean,mujoco.mjtObj.mjOBJ_GEOM,i) in allowed]
 textures={};aliases=[]
 for name in names:
  gid=mujoco.mj_name2id(clean,mujoco.mjtObj.mjOBJ_GEOM,name);mat=int(clean.geom_matid[gid]);tid=int(clean.mat_texid[mat,1]);assert tid>=0
  users=[]
  for i in range(clean.ngeom):
   mi=int(clean.geom_matid[i])
   if mi>=0 and tid in clean.mat_texid[mi]:users.append(mujoco.mj_id2name(clean,mujoco.mjtObj.mjOBJ_GEOM,i))
  assert set(users)<=set(names),f'Foreground shares texture {tid}: {users}; no in-place recolor allowed'
  textures[tid]=users
  aliases.append(dict(geom=name,material=mujoco.mj_id2name(clean,mujoco.mjtObj.mjOBJ_MATERIAL,mat),texture_id=tid,all_texture_geom_users=users))
 models={};assets=[];exact_fields=[]
 for mode in ['expanded_h','expanded_hs']:
  model=copy.copy(clean)
  for tid,users in textures.items():
   start=int(clean.tex_adr[tid]);height=int(clean.tex_height[tid]);width=int(clean.tex_width[tid]);channels=int(clean.tex_nchannel[tid]);assert channels==3
   rgb=clean.tex_data[start:start+height*width*3].reshape(height,width,3);hsv=cv2.cvtColor(rgb.astype('float32')/255,cv2.COLOR_RGB2HSV);oldhsv=hsv.copy()
   delta=old['delta_h'] if mode=='expanded_h' else .20;hsv[...,0]=(hsv[...,0]+360*delta)%360
   if mode=='expanded_hs':hsv[...,1]+=.35*(1-hsv[...,1])
   changed=np.rint(np.clip(cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB),0,1)*255).astype('uint8');model.tex_data[start:start+height*width*3]=changed.ravel()
   assets.append(dict(mode=mode,texture_id=tid,users=users,source_bytes_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),changed_bytes_sha256=hashlib.sha256(changed.tobytes()).hexdigest(),
    delta_h=delta,saturation_gain=.35 if mode=='expanded_hs' else 0,low_saturation_fraction=float((oldhsv[...,1]<.05).mean()),
    texture_value_quantization_max=float(np.max(abs(changed.max(2).astype(float)/255-oldhsv[...,2])))))
  for key in dir(clean):
   value=getattr(clean,key)
   if isinstance(value,np.ndarray) and key!='tex_data':assert value.tobytes()==getattr(model,key).tobytes(),key;exact_fields.append(key)
  models[mode]=model
 dump(out/'appearance_configuration.json',dict(source_model=str(snapshot),source_model_sha256=sha(snapshot),assets=assets,aliases=aliases,changed_geoms=names,old_whitelist=old['changed_geoms'],
  newly_included=['floor'],all_nontarget_model_arrays_exact=sorted(set(exact_fields)),physics_camera_lights_reflection_unchanged=True,
  shared_reference_handling='All texture users proved background-only; separate whole-model copies, no foreground alias. Unknown/shared foreground fails closed.',
  clean_model_note='Exact compiled runtime model, no XML recompilation',diagnostic_only=True))
 return clean,models,names,old

def main(suite):
 out=RUN/'color_final'/suite;out.mkdir(parents=True,exist_ok=False);started=time.time();counts=dict(state_restores=0,kinematic_forwards=0,rgb_renders=0,segmentation_renders=0,physics_steps=0)
 m,models,names,old=prepare_exact(suite,out)
 # Save the compiled model exactly; re-importing XML is not a byte-exact model snapshot.
 mujoco.mj_saveModel(m,str(out/'canonical_model.mjb'),None);reload=mujoco.MjModel.from_binary_path(str(out/'canonical_model.mjb'))
 engines={'clean':(m,mujoco.MjData(m),mujoco.Renderer(m,256,256)),'reload':(reload,mujoco.MjData(reload),mujoco.Renderer(reload,256,256))}
 for k,model in models.items():engines[k]=(model,mujoco.MjData(model),mujoco.Renderer(model,256,256))
 options=json.loads((RUN/'capture'/suite/'render_options.json').read_text());o=opt();o.geomgroup[:]=options['geomgroup'];o.sitegroup[:]=options['sitegroup'];o.flags[:]=options['flags']
 for _,_,r in engines.values():r.scene.flags[:]=options['render_flags']
 arr=np.load(OLD/'pilot_replay'/suite/'candidate_replay.npz');rlds=np.load(OLD/'rlds_mapping_v2'/suite/'pilot_rlds.npz')
 arm=[m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'robot0_joint'+str(i))] for i in range(1,8)];site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'gripper0_grip_site')
 ids={name:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,name) for name in names};oldids=[ids[x] for x in old['changed_geoms']]
 caps=[cv2.VideoCapture(str(ROOT/'data/libero_converted'/suite/'traj_000010'/f'{c}_rgb.mp4')) for c in ['agentview_camera','wrist_camera']]
 writer=cv2.VideoWriter(str(out/'full_comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(1280,512));assert writer.isOpened()
 fixed=sorted(set(np.linspace(0,len(arr['states'])-1,7,dtype=int)));sheet=Image.new('RGB',(1280,len(fixed)*540));draw=ImageDraw.Draw(sheet)
 rows=[];pairs=[];geometry_rows=[];record={k:[] for k in ['clean','expanded_h','expanded_hs']};maskrecord=[];geomhistory=[];camera_pos=[];camera_rot=[];fk=[]
 try:
  for t,state in enumerate(arr['states']):
   images={};geometries={}
   for k,(model,data,r) in engines.items():
    legacy(model,data,state);counts['state_restores']+=1;counts['kinematic_forwards']+=1
    pre=np.r_[data.time,data.qpos,data.qvel];images[k]=render(r,data,o);counts['rgb_renders']+=2;assert np.array_equal(pre,np.r_[data.time,data.qpos,data.qvel]);geometries[k]=geometry(data)
   for k in engines:assert np.max(abs(geometries[k]-geometries['clean']))<=1e-12
   self_exact=np.array_equal(images['clean'],images['reload']);assert self_exact
   d=engines['clean'][1];r=engines['clean'][2];masks=[];segs=[]
   for c in CAMS:
    r.enable_segmentation_rendering();r.update_scene(d,camera=c,scene_option=o);seg=r.render().copy()[:,::-1];r.disable_segmentation_rendering();counts['segmentation_renders']+=1
    mask=(seg[...,1]==int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(seg[...,0],list(ids.values()));masks.append(mask);segs.append(seg)
   orig=[]
   for cap in caps:ok,bgr=cap.read();assert ok;orig.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
   orig=np.stack(orig);row=dict(frame=t,time=float(d.time),source_action_index=int(arr['source_action_indices'][t]),A_self_exact=self_exact,
    B_joint_error=float(np.max(abs(d.qpos[arm]-rlds['joints'][t]))),B_eef_error_m=float(np.max(abs(d.site_xpos[site]-rlds['state'][t,:3]))))
   row.update(metrics(images['clean'],orig,'B'));pairs.append(row)
   for i,c in enumerate(CAMS):
    mask=masks[i];seg=segs[i];visible={name:int(((seg[...,1]==int(mujoco.mjtObj.mjOBJ_GEOM)) & (seg[...,0]==ident)).sum()) for name,ident in ids.items()}
    # Count ALL visible geoms so omitted static surfaces and excluded objects remain inspectable.
    visible_all={str(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(g))):int(((seg[...,1]==int(mujoco.mjtObj.mjOBJ_GEOM)) & (seg[...,0]==g)).sum()) for g in np.unique(seg[...,0][seg[...,1]==int(mujoco.mjtObj.mjOBJ_GEOM)])}
    geometry_rows.append(dict(frame=t,camera=c,visible_geoms=visible_all,background_geoms=visible))
    for k in ['expanded_h','expanded_hs']:
     diff=np.abs(images[k][i].astype(float)-images['clean'][i].astype(float));lum=np.dot(images[k][i].astype(float)-images['clean'][i].astype(float),[.2126,.7152,.0722]);area=int(mask.sum());bgchange=float(diff[mask].mean()) if area else None
     rows.append(dict(frame=t,camera=c,mode=k,background_pixels=area,background_fraction=area/65536,old_whitelist_pixels=int(((seg[...,1]==int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(seg[...,0],oldids)).sum()),
      changed_pixels=int((diff.max(2)>1).sum()),changed_background_pixels=int(((diff.max(2)>1)&mask).sum()),background_rgb_mae=bgchange,full_rgb_mae=float(diff.mean()),
      outside_background_rgb_mae=float(diff[~mask].mean()) if (~mask).any() else None,luminance_signed_mean=float(lum.mean()),luminance_abs_mean=float(abs(lum).mean()),
      visible_but_weak=bool(area>0 and bgchange<1.),background_invisible=area==0))
   overlay=images['clean'].copy()
   for i in range(2):overlay[i][masks[i]]=(.4*overlay[i][masks[i]]+.6*np.array([255,0,255])).astype('uint8')
   panels=[orig,images['clean'],overlay,images['expanded_h'],images['expanded_hs']];panel=np.concatenate([np.concatenate([p[i] for p in panels],1) for i in range(2)],0)
   writer.write(cv2.cvtColor(panel,cv2.COLOR_RGB2BGR))
   if t in fixed:
    y=fixed.index(t)*540;sheet.paste(Image.fromarray(panel),(0,y+28));draw.text((0,y+4),f'{suite} t={t}: original | synchronized clean | GEOM background | H expanded | fixed H+S',fill='white')
   for k in record:record[k].append(images[k])
   maskrecord.append(np.stack(masks));geomhistory.append(geometry(d));camera_pos.append(d.cam_xpos.copy());camera_rot.append(d.cam_xmat.copy());fk.append(np.r_[d.qpos[arm],d.site_xpos[site]])
   if t%50==0:print('color',suite,t,len(arr['states']),flush=True)
  writer.release();sheet.save(out/'fixed_comparison.jpg');csvout(out/'frame_color_metrics.csv',rows);csvout(out/'frame_correspondence.csv',pairs);dump(out/'visible_geoms.json',geometry_rows)
  np.savez_compressed(out/'paired_rgb.npz',**{k:np.stack(v) for k,v in record.items()},background_mask=np.stack(maskrecord))
  np.savez_compressed(out/'synchronized_metadata.npz',states=arr['states'],actions=arr['actions'],source_action_indices=arr['source_action_indices'],geometry=np.stack(geomhistory),camera_positions=np.stack(camera_pos),camera_rotations=np.stack(camera_rot),joints_and_eef=np.stack(fk))
  B=all(x['B_joint_error']<=.002 and x['B_eef_error_m']<=.002 and all(x[f'B_{c}_mae']<=8 and x[f'B_{c}_p99']<=40 for c in ['front','wrist']) for x in pairs)
  dump(out/'result.json',dict(frames=len(pairs),A_self_restore=True,B_original_numeric_pixel=B,counts=counts,seconds=time.time()-started,
    canonical_model_sha256=sha(out/'canonical_model.mjb'),pairing_exact=True,admitted_as_original_augmentation=False,
    reason='Diagnostic pair only; original-correspondence and full admission separate',models_inference=0,updates=0,backward=0,policy_rollout=0))
 finally:
  writer.release()
  for cap in caps:cap.release()
  for _,_,r in engines.values():r.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--suite',choices=SUITES,required=True);main(p.parse_args().suite)
