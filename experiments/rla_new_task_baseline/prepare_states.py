"""Twenty task-specific reset states, no actions or policy scoring."""
import argparse,json,random,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,mujoco
from PIL import Image
from libero.libero.envs.env_wrapper import ControlEnv
from audit_source import sha,save

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--bddl',type=Path,required=True);a=p.parse_args();o=a.dataset/'development_protocol';o.mkdir(exist_ok=False)
 protocol=json.loads((a.dataset/'experiment_protocol.json').read_text());bg=json.loads((a.dataset/'background_protocol.json').read_text());records=[]
 assert 'Open wooden_cabinet_1_middle_region' in a.bddl.read_text()
 configs={'clean':{},'training_background':bg['training'],**bg['development']}
 save(o/'plan.json',dict(seeds=protocol['development_seeds'],source='current task placement reset with initialization_noise=None, not source/sealed saved initial states',conditions=list(configs),selection='fixed seeds, reject initial success or invalid environment only; failure stops without replacement',max_steps=300,control_freq=20,chunk=8,policy_calls=0,training_collection=False))
 for seed in protocol['development_seeds']:
  folder=o/f'seed_{seed}';folder.mkdir();random.seed(seed);np.random.seed(seed)
  env=ControlEnv(bddl_file_name=str(a.bddl),use_camera_obs=False,has_offscreen_renderer=False,has_renderer=False,control_freq=20,horizon=1000,initialization_noise=None)
  try:
   env.seed(seed);env.reset();env.sim.forward();assert not env.check_success()
   s=env.get_sim_state().copy();np.save(folder/'state.npy',s);xml=env.sim.model.get_xml();checks=[];baseline=None
   for condition,files in configs.items():
    tree=ET.fromstring(xml)
    for name,path in files.items():
     nodes=[x for x in tree.iter('texture') if x.get('name')==name];assert len(nodes)==1;nodes[0].set('file',path);assert sha(path)==bg['assets'][path]
    text=ET.tostring(tree,encoding='unicode');(folder/(condition+'.xml')).write_text(text);m=mujoco.MjModel.from_xml_string(text);d=mujoco.MjData(m);d.time=s[0];d.qpos[:]=s[1:1+m.nq];d.qvel[:]=s[1+m.nq:];mujoco.mj_forward(m,d)
    assert m.vis.quality.offsamples==4
    q=float(d.qpos[m.joint('wooden_cabinet_1_middle_level').qposadr[0]]);assert q>=-.14
    renderer=mujoco.Renderer(m,224,224);opt=mujoco.MjvOption();opt.geomgroup[0]=0;opt.geomgroup[1]=1;opt.sitegroup[:]=0;rgb=[];seg=[];depth=[]
    for cam in ['agentview','robot0_eye_in_hand']:
     renderer.update_scene(d,camera=cam,scene_option=opt);renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=1;rgb.append(renderer.render()[:,::-1].copy());renderer.enable_segmentation_rendering();seg.append(renderer.render()[:,::-1].copy());renderer.disable_segmentation_rendering();renderer.enable_depth_rendering();depth.append(renderer.render()[:,::-1].copy());renderer.disable_depth_rendering()
    renderer.close();rgb=np.stack(rgb);seg=np.stack(seg);depth=np.stack(depth);geom={k:getattr(d,k).copy() for k in ['qpos','qvel','xpos','xquat','cam_xpos','cam_xmat']}
    if baseline is None:baseline=(rgb,seg,depth,geom)
    else:
     np.testing.assert_array_equal(seg,baseline[1]);np.testing.assert_array_equal(depth,baseline[2])
     for k in geom:np.testing.assert_array_equal(geom[k],baseline[3][k])
     assert np.any(rgb!=baseline[0]),'intervention invisible in entire static dual view'
    Image.fromarray(np.concatenate(rgb,axis=1)).save(folder/(condition+'.png'))
    checks.append(dict(condition=condition,XML_sha256=sha(folder/(condition+'.xml')),depth_segmentation_geometry_equal=True,RGB_MAE=float(abs(rgb.astype(float)-baseline[0]).mean()),middle_joint=q))
   records.append(dict(seed=seed,state_file=str(folder/'state.npy'),state_sha256=hashlib.sha256(s.tobytes()).hexdigest(),initial_success=False,conditions=checks))
   print('STATIC_STATE_PASS',seed,flush=True)
  finally:env.close()
 save(o/'evaluation_protocol.json',dict(status='FROZEN_STATIC_CHECKED',new_development_states=records,conditions=['clean',*bg['development']],max_steps=300,control_freq=20,execution_chunk=8,success='env.check_success(): Open wooden_cabinet_1_middle_region, middle joint qpos < -0.14',bddl=str(a.bddl),bddl_sha256=sha(a.bddl),language='open the middle drawer of the cabinet',reset_count=20,physics_steps=0,policy_rollouts=0,independence='development reset states, not sealed states; not proof of distributional independence',stage_proxy='official Open; middle drawer joint progress; EE-handle distance if handle site identity validated; no bowl stages'))
if __name__=='__main__':main()
