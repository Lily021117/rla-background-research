"""Fail-closed same-state dual-view hue rerender. Never advances simulation.

May prepare isolated texture/material assets while reproduction is blocked, but
rendering requires explicit complete-sequence Clean review in addition to gates.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',LP_NUM_THREADS='2')
import argparse,copy,json,sys,hashlib,time,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,cv2,mujoco
from PIL import Image
from prepare import ROOT,RUN,sha,dump,csvout
sys.path.insert(0,str(Path(__file__).parents[2]))
from experiments.rla_official_background_hue.paired_hue import hue_parameter,hue_rgb
WHITELIST=['table_visual','wall_leftcorner_visual','wall_rightcorner_visual','wall_left_visual','wall_right_visual','wall_rear_visual','wall_front_visual']
PHYS=['body_mass','body_inertia','body_pos','body_quat','geom_pos','geom_quat','geom_size','geom_type','geom_friction','geom_contype','geom_conaffinity','geom_solref','geom_solimp','jnt_type','jnt_pos','jnt_axis','jnt_range','dof_damping','dof_armature','actuator_gainprm','actuator_biasprm','cam_pos','cam_quat','cam_fovy','mesh_vert','mesh_face','light_pos','light_dir','light_diffuse','light_specular','light_ambient']
def prepare(suite):
 src=RUN/'pilot_replay'/suite;out=RUN/'paired_prepare'/suite;out.mkdir(parents=True,exist_ok=False)
 tree=ET.parse(src/'runtime_model.xml');root=tree.getroot();asset=root.find('asset')
 delta,key=hue_parameter(20261103,'train',dict(suite=suite,trajectory_id='000010',data_version='official_ae_rerender_v1'),0)
 mats={e.get('name'):e for e in asset.findall('material')};tex={e.get('name'):e for e in asset.findall('texture')}
 selected=[e for e in root.iter('geom') if e.get('name') in WHITELIST];assert selected
 rows=[];done={}
 for geom in selected:
  old=geom.get('material');assert old in mats
  if old not in done:
   mat=copy.deepcopy(mats[old]);oldtex=mat.get('texture');assert oldtex in tex and tex[oldtex].get('file')
   texture=copy.deepcopy(tex[oldtex]);image=Path(texture.get('file'));assert image.exists()
   rgb=np.array(Image.open(image).convert('RGB'));changed=hue_rgb(rgb,np.ones(rgb.shape[:2],bool),delta)
   dest=out/f'{oldtex}_hue.png';Image.fromarray(changed).save(dest)
   newtex=oldtex+'_paired_hue_'+old;newmat=old+'_paired_hue';texture.set('name',newtex);texture.set('file',str(dest));mat.set('name',newmat);mat.set('texture',newtex)
   asset.append(texture);asset.append(mat);done[old]=newmat
   rows.append(dict(original_material=old,cloned_material=newmat,original_texture=oldtex,source_asset=str(image),source_asset_sha256=sha(image),
    changed_asset=str(dest),changed_asset_sha256=sha(dest),low_saturation_fraction=float((cv2.cvtColor(rgb.astype('float32')/255,cv2.COLOR_RGB2HSV)[...,1]<.05).mean()),
    mean_texture_rgb_change=float(np.abs(changed.astype(float)-rgb).mean()),original_material_users=[e.get('name') for e in ET.parse(src/'runtime_model.xml').getroot().iter('geom') if e.get('material')==old]))
  geom.set('material',done[old])
 tree.write(out/'background.xml',encoding='unicode')
 clean=mujoco.MjModel.from_xml_path(str(src/'runtime_model.xml'));aug=mujoco.MjModel.from_xml_path(str(out/'background.xml'))
 for field in PHYS:assert np.array_equal(getattr(clean,field),getattr(aug,field)),field
 changed_geoms=[mujoco.mj_id2name(clean,mujoco.mjtObj.mjOBJ_GEOM,i) for i in range(clean.ngeom) if clean.geom_matid[i]!=aug.geom_matid[i]]
 assert set(changed_geoms)=={e.get('name') for e in selected}
 dump(out/'background_whitelist.json',dict(geom_whitelist=WHITELIST,changed_geoms=changed_geoms,material_texture_changes=rows,delta_h=delta,key=key,
   original_XML=str(src/'runtime_model.xml'),original_XML_sha256=sha(src/'runtime_model.xml'),augmented_XML_sha256=sha(out/'background.xml'),
   physical_fields_exact=PHYS,shared_references_isolated=True,lights_unchanged=True,reflection_not_disabled=True,shadows_not_disabled=True,
   state_restores=0,render_calls=0,augmentation_admission='NOT_GRANTED_BY_ASSET_PREPARATION'))
 return out
def render(suite):
 src=RUN/'pilot_replay'/suite;prep=RUN/'paired_prepare'/suite
 result=json.loads((src/'result.json').read_text());assert result['numeric_gate'] and result['pixel_gate'],'BLOCKED Clean reproduction'
 review=json.loads((src/'visual_admission.json').read_text());assert review['complete_sequence_pass'] is True
 out=RUN/'paired_rendered'/suite;out.mkdir(parents=True,exist_ok=False)
 arr=np.load(src/'candidate_replay.npz');states=arr['states'];m=mujoco.MjModel.from_xml_path(str(src/'runtime_model.xml'));d=mujoco.MjData(m)
 renderer=mujoco.Renderer(m,256,256);opt=mujoco.MjvOption();opt.geomgroup[0]=0;opt.geomgroup[1]=1;opt.sitegroup[:]=0
 # Two compiled engines preserve reflection/shadows and share identical states.
 ma=mujoco.MjModel.from_xml_path(str(prep/'background.xml'));da=mujoco.MjData(ma);ra=mujoco.Renderer(ma,256,256)
 for field in PHYS:assert np.array_equal(getattr(m,field),getattr(ma,field))
 recordings={'clean':[],'background':[]};metadata=[];counts=dict(state_restores=0,render_calls=0,simulation_steps=0);started=time.time();checks=[];camera_records=[]
 try:
  for t,s in enumerate(states):
   labs=[]
   for label,model,data,r in [('clean',m,d,renderer),('background',ma,da,ra)]:
    data.time=s[0];data.qpos[:]=s[1:1+model.nq];data.qvel[:]=s[1+model.nq:];mujoco.mj_forward(model,data);counts['state_restores']+=1
    before=np.r_[data.time,data.qpos,data.qvel];views=[]
    for cam in ['agentview','robot0_eye_in_hand']:
     r.update_scene(data,camera=cam,scene_option=opt);views.append(r.render().copy()[:,::-1]);counts['render_calls']+=1
    assert np.array_equal(before,np.r_[data.time,data.qpos,data.qvel]);recordings[label].append(np.stack(views))
    labs.append(np.r_[data.cam_xpos.flatten(),data.cam_xmat.flatten(),data.xpos.flatten()])
   assert np.array_equal(labs[0],labs[1]);metadata.append(dict(timestep=t,source_action_index=int(arr['source_action_indices'][t]),state_time=float(s[0]),state_sha256=hashlib.sha256(s.tobytes()).hexdigest(),split='train',trajectory_id='000010',suite=suite))
   delta=np.abs(recordings['clean'][-1].astype('int16')-arr['rgb'][t].astype('int16'))
   checks.append(dict(timestep=t,front_mae=float(delta[0].mean()),wrist_mae=float(delta[1].mean()),front_p99=float(np.quantile(delta[0],.99)),wrist_p99=float(np.quantile(delta[1],.99)),
    front_color_change=float(np.abs(recordings['clean'][-1][0].astype(float)-recordings['background'][-1][0]).mean()),
    wrist_color_change=float(np.abs(recordings['clean'][-1][1].astype(float)-recordings['background'][-1][1]).mean())))
   camera_records.append(dict(c2w_raw_rotation=d.cam_xmat.copy(),c2w_raw_position=d.cam_xpos.copy()))
   if t%50==0:print('same-state pair',suite,t,len(states),flush=True)
  for name,images in recordings.items():np.savez_compressed(out/f'{name}.npz',rgb=np.stack(images))
  np.savez_compressed(out/'camera_poses.npz',rotation=np.stack([r['c2w_raw_rotation'] for r in camera_records]),position=np.stack([r['c2w_raw_position'] for r in camera_records]))
  csvout(out/'pairing_manifest.csv',metadata);csvout(out/'same_state_render_checks.csv',checks)
  clean_recheck=all(x['front_mae']<=8 and x['wrist_mae']<=8 and x['front_p99']<=40 and x['wrist_p99']<=40 for x in checks)
  dump(out/'result.json',dict(status='PAIRED_PENDING_LOADER_CHECK' if clean_recheck else 'BLOCKED_RENDERER_RECHECK',**counts,frames=len(states),input_is_replay_candidate=True,seconds=time.time()-started,
   same_state_camera_and_body_geometry_exact=True,restored_clean_vs_replay_gate=clean_recheck,AE_forwards=0,DINO_forwards=0,policy_forwards=0))
 except BaseException as e:
  dump(out/'failure.json',dict(error=repr(e),**counts,completed_frames=len(metadata)));raise
 finally:renderer.close();ra.close()
def main():
 p=argparse.ArgumentParser();p.add_argument('--suite',required=True);p.add_argument('--render',action='store_true');a=p.parse_args()
 if a.render:render(a.suite)
 else:prepare(a.suite)
if __name__=='__main__':main()
