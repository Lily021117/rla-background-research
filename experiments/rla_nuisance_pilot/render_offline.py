"""Saved development states only; forward kinematics/render, never mj_step."""
import sys
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_nuisance_pilot.common import *
def main():
 anchors=read(ROOT/'offline_anchors.json');conditions=[dict(id='clean')]+read(ROOT/'conditions.json');records=[]
 for episode in sorted({a['episode'] for a in anchors}):
  es=[a for a in anchors if a['episode']==episode];source=Path(es[0]['root']);lab=np.load(source/'trajectory_labels.npz');prov=read(source/'trajectory_provenance.json');xml=next(Path(p) for p in prov['source_hashes'] if p.endswith('/clean.xml'));assert sha(xml)==prov['source_hashes'][str(xml)]
  text=xml.read_text();reference={}
  for condition in conditions:
   m=mujoco.MjModel.from_xml_string(text if condition['id']=='clean' else xml_for(text,condition));d=mujoco.MjData(m);rr=mujoco.Renderer(m,224,224);opt=mujoco.MjvOption();opt.geomgroup[0]=0;opt.geomgroup[1]=1;opt.sitegroup[:]=0
   folder=ROOT/'offline_images'/episode/condition['id'];folder.mkdir(parents=True)
   try:
    for a in es:
     t=a['t'];assert lab['valid_horizon_mask'][t].all();images=[]
     for j in range(t,t+9):
      state=lab['states'][j];assert len(state)==1+m.nq+m.nv;d.time=state[0];d.qpos[:]=state[1:1+m.nq];d.qvel[:]=state[1+m.nq:];mujoco.mj_forward(m,d)
      p=np.r_[d.site_xpos[m.site('gripper0_grip_site').id],Rotation.from_quat(np.roll(d.xquat[m.body('robot0_right_hand').id],-1)).as_rotvec(),[d.qpos[m.joint('gripper0_finger_joint'+str(i)).qposadr[0]] for i in (1,2)]].astype(np.float32)
      np.testing.assert_allclose(p,lab['proprio'][j],rtol=0,atol=1e-6)
      geometry=np.concatenate([d.xpos.ravel(),d.xquat.ravel(),d.cam_xpos.ravel(),d.cam_xmat.ravel()])
      if condition['id']=='clean':reference[j]=geometry.copy()
      else:np.testing.assert_array_equal(geometry,reference[j])
      views=[]
      for camera in ['agentview','robot0_eye_in_hand']:
       rr.update_scene(d,camera=camera,scene_option=opt);rr.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=1;views.append(rr.render()[:,::-1].copy().transpose(2,0,1))
      images.append(views);assert d.time==state[0];np.testing.assert_array_equal(d.qpos,state[1:1+m.nq])
     path=folder/f'{t:04}.npz';np.savez_compressed(path,rgb=np.array(images),K=lab['K'][t:t+9],c2w=lab['c2w'][t:t+9],proprio=lab['proprio'][t],actions=lab['actions'][t:t+8],mask=lab['valid_horizon_mask'][t])
     rec=dict(episode=episode,t=t,appearance=condition['id'],path=str(path),sha256=sha(path),state_proprio_geometry_exact=True,physical_steps=0)
     if condition['id']=='clean':
      original=np.load(source/'frames'/f'{t:04}.npz')['clean'];rec['clean_render_vs_cached_RGB_MAE']=float(np.abs(original.astype(float)-np.array(images)[0]).mean())
     records.append(rec)
   finally:rr.close()
   print('OFFLINE_RENDER',episode,condition['id'],len(es),flush=True)
 save(ROOT/'offline_render_inventory.json',records);save(ROOT/'offline_render_acceptance.json',dict(status='PASS',anchors=100,conditions=7,paired_current_and_future=True,physical_steps=0,policy_forward=0,backward=0))
if __name__=='__main__':main()
