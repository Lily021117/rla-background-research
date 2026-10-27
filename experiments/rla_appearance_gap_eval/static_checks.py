"""CPU geometry, intervention and visibility checks, no policy actions."""
import argparse,hashlib,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_appearance_gap_eval.common import *

def execute(out):
 protocol=read(out/'development_protocol/evaluation_protocol.json');records=[];render_rows=[];snapshots=[];mapping=[]
 (out/'static_previews').mkdir()
 model_fields=['geom_pos','geom_quat','geom_size','geom_rgba','geom_matid','geom_contype','geom_conaffinity','body_pos','body_quat','cam_pos','cam_quat','cam_fovy','light_pos','light_dir','light_diffuse','light_ambient','light_specular','mat_rgba','mat_specular','mat_shininess','mat_reflectance','mat_texrepeat','mat_texid']
 data_fields=['qpos','qvel','xpos','xquat','cam_xpos','cam_xmat']
 for ordinal,s in enumerate(protocol['new_development_states']):
  state=np.load(s['state_file']);assert hashlib.sha256(state.tobytes()).hexdigest()==s['state_sha256'];folder=Path(s['state_file']).parent;baseline=None;clean_data=None;original_rgb=None;original_seg=None;original_depth=None;tiles=[]
  for c in s['conditions']:
   name=c['condition'];xml=folder/(name+'.xml');assert sha(xml)==c['XML_sha256'];m=mujoco.MjModel.from_xml_path(str(xml));d=mujoco.MjData(m);assert len(state)==1+m.nq+m.nv
   d.time=state[0];d.qpos[:]=state[1:1+m.nq];d.qvel[:]=state[1+m.nq:];mujoco.mj_forward(m,d)
   snapshot={k:np.array(getattr(m,k)).copy() for k in model_fields};current={k:np.array(getattr(d,k)).copy() for k in data_fields}
   if baseline is None:baseline=snapshot;clean_data=current
   else:
    for k in model_fields:np.testing.assert_array_equal(snapshot[k],baseline[k],err_msg=k)
    for k in data_fields:np.testing.assert_array_equal(current[k],clean_data[k],err_msg=k)
   assert m.vis.quality.offsamples==4
   # Material references remain restricted to visible table/wall GEOMs.
   tids={m.texture(n).id for n in ['tex-table','tex-wall']};background_geoms=[]
   for gid,mid in enumerate(m.geom_matid):
    if mid>=0 and any(int(t) in tids for t in np.asarray(m.mat_texid[mid]).reshape(-1)):
     geom_name=m.geom(gid).name;assert geom_name=='table_visual' or geom_name.startswith('wall_'),geom_name;background_geoms.append(gid)
     if ordinal==0 and name=='clean':mapping.append(dict(geom_id=gid,geom=geom_name,material=m.material(mid).name,category='permitted table/wall visual GEOM'))
   assert len(background_geoms)==7
   if ordinal in (0,10):
    renderer=mujoco.Renderer(m,224,224);opt=mujoco.MjvOption();opt.geomgroup[0]=0;opt.geomgroup[1]=1;opt.sitegroup[:]=0;rgb=[];segments=[];depths=[]
    try:
     for camera in ['agentview','robot0_eye_in_hand']:
      renderer.update_scene(d,camera=camera,scene_option=opt);renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=1
      rgb.append(renderer.render()[:,::-1].copy());renderer.enable_segmentation_rendering();segments.append(renderer.render()[:,::-1].copy());renderer.disable_segmentation_rendering();renderer.enable_depth_rendering();depths.append(renderer.render()[:,::-1].copy());renderer.disable_depth_rendering()
    finally:renderer.close()
    rgb=np.stack(rgb);segments=np.stack(segments);depths=np.stack(depths)
    if original_rgb is None:original_rgb=rgb;original_seg=segments;original_depth=depths
    else:np.testing.assert_array_equal(segments,original_seg);np.testing.assert_array_equal(depths,original_depth)
    for cam,camera in enumerate(['agentview','robot0_eye_in_hand']):
     geom_mask=segments[cam,:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM);bg=geom_mask & np.isin(segments[cam,:,:,0],background_geoms)
     bowl_ids=[g for g in range(m.ngeom) if 'akita_black_bowl' in (m.geom(g).name or '')];plate_ids=[g for g in range(m.ngeom) if 'plate' in (m.geom(g).name or '')]
     bowl=geom_mask & np.isin(segments[cam,:,:,0],bowl_ids);plate=geom_mask & np.isin(segments[cam,:,:,0],plate_ids)
     if cam==0:assert bowl.sum()>0 and plate.sum()>0,'task not visible in fixed agentview'
     diff=abs(rgb[cam].astype(float)-original_rgb[cam]);assert np.isfinite(depths[cam]).all()
     if name!='clean' and bg.sum()>0:assert np.any(diff[bg]>0),'background intervention ineffective'
     render_rows.append(dict(state=s['seed'],condition=name,camera=camera,background_pixels=int(bg.sum()),background_MAE=float(diff[bg].mean()) if bg.any() else None,whole_RGB_MAE=float(diff.mean()),max_channel_difference=int(diff.max()),bowl_visible_pixels=int(bowl.sum()),plate_visible_pixels=int(plate.sum()),segmentation_exact=True,depth_exact=True,task_geometry_visibility_unchanged=True,reflection_effects_allowed=True))
    tile=Image.fromarray(np.concatenate(rgb,axis=1));ImageDraw.Draw(tile).text((4,4),name,fill=(255,0,255));tiles.append(tile);tile.save(out/'static_previews'/f'{s["seed"]}_{name}.png')
   assert d.time==state[0];np.testing.assert_array_equal(d.qpos,state[1:1+m.nq]);np.testing.assert_array_equal(d.qvel,state[1+m.nq:])
   records.append(dict(state=s['seed'],condition=name,model_nontexture_fields_exact=True,state_camera_pose_exact=True,physical_time_not_advanced=True,MSAA=4))
   del d,m
  if tiles:
   canvas=Image.new('RGB',(448*len(tiles),224))
   for j,tile in enumerate(tiles):canvas.paste(tile,(448*j,0))
   canvas.save(out/'static_previews'/f'{s["seed"]}_all_conditions.png')
  print('STATIC_STATE',ordinal+1,s['seed'],'PASS',flush=True)
 table(out/'static_geometry_checks.csv',records);table(out/'static_render_checks.csv',render_rows);table(out/'background_geom_mapping.csv',mapping)
 save(out/'static_acceptance.json',dict(status='PASS',states_checked=20,conditions_checked=5,rendered_state_ordinals=[0,10],camera_order=['agentview','robot0_eye_in_hand'],model_geometric_material_light_camera_fields_exact=True,segmentation_and_depth_exact=True,only_target_texture_files_changed=True,task_geometry_visible=True,scope='same saved physical states, static CPU OSMesa render; not a policy rollout',policy_actions=0,optimizer_updates=0,backward=0))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);execute(p.parse_args().out)
