"""Task-reset-generated development states; static checks only, no policy."""
import argparse,json,hashlib,xml.etree.ElementTree as ET,os,random
from pathlib import Path
import numpy as np
from PIL import Image
import mujoco
from libero.libero.envs.env_wrapper import ControlEnv
from libero.libero import get_libero_path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);a=p.parse_args();out=a.run/'development_protocol';out.mkdir(exist_ok=False)
def save(n,x):
    with (out/n).open('x') as f:json.dump(x,f,indent=2)
def sha(b):return hashlib.sha256(b).hexdigest()
data=json.loads((a.dataset/'frozen_dataset_manifest.json').read_text());selection=json.loads((a.dataset/'dataset_selection.json').read_text());seen=selection['background_texture_files'];seeds=list(range(2026102300,2026102320));colors={'tex-table':[178,132,82],'tex-wall':[84,127,132]}
save('protocol_plan.json',dict(frozen_before_render=True,seeds=seeds,conditions=['clean','seen_background','unseen_warm_teal'],unseen_RGB=colors,intervention='only tex-table/tex-wall file resources, fixed solid-color textures at same dimensions',reflection=1,MSAA=4,light_camera_robot_initialization_changes=False,policy_rollouts=0,training_collection=False,selection='all20 fixed seeds, only legality checks, no policy scoring; failures retained not replaced',historical5='regression development, not independent test',relationship_to_sealed='no sealed contents read; overlap or statistical independence from sealed states cannot be certified'))
asset=out/'assets';asset.mkdir();unseen={}
for name,rgb in colors.items():
    with Image.open(seen[name]) as im:size=im.size
    path=asset/(name+'.png');Image.new('RGB',size,tuple(rgb)).save(path);unseen[name]=str(path)
known=[]
for e in data['episodes']:
    states=np.load(Path(e['root'])/'trajectory_labels.npz')['states'];known.extend(states)
known=np.stack(known);records=[];robot_ref=None
for seed in seeds:
    folder=out/f'seed_{seed}';folder.mkdir();env=ControlEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/'libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate.bddl'),use_camera_obs=False,has_offscreen_renderer=False,has_renderer=False,control_freq=20,horizon=1000,initialization_noise=None)
    try:
        random.seed(seed);np.random.seed(seed);env.seed(seed);env.reset();env.sim.forward();assert not env.check_success();state=env.get_sim_state().copy();m,d=env.sim.model._model,env.sim.data._data;assert np.isfinite(state).all();xml=env.sim.model.get_xml();root=ET.fromstring(xml)
        names=['robot0_joint'+str(i) for i in range(1,8)]+['gripper0_finger_joint1','gripper0_finger_joint2'];q=d.qpos[[m.joint(n).qposadr[0] for n in names]].copy()
        if robot_ref is None:robot_ref=q
        else:np.testing.assert_allclose(q,robot_ref,atol=1e-8,rtol=0)
        np.save(folder/'state.npy',state);baseline=None;clean_rgb=None;checks=[]
        for condition,files in [('clean',{}),('seen_background',seen),('unseen_warm_teal',unseen)]:
            tree=ET.fromstring(ET.tostring(root));changes=[]
            for name,path in files.items():
                nodes=[x for x in tree.iter('texture') if x.get('name')==name];assert len(nodes)==1;changes.append(dict(texture=name,old=nodes[0].get('file'),new=path));nodes[0].set('file',path)
            text=ET.tostring(tree,encoding='unicode');(folder/(condition+'.xml')).write_text(text);model=mujoco.MjModel.from_xml_string(text);dat=mujoco.MjData(model);dat.time=state[0];dat.qpos[:]=state[1:1+model.nq];dat.qvel[:]=state[1+model.nq:];mujoco.mj_forward(model,dat)
            geom={k:np.array(getattr(dat,k)).copy() for k in ('qpos','qvel','xpos','xquat','cam_xpos','cam_xmat')}
            if baseline is None:baseline=geom
            else:
                for k in geom:np.testing.assert_allclose(geom[k],baseline[k],atol=1e-8,rtol=0)
            renderer=mujoco.Renderer(model,224,224);opt=mujoco.MjvOption();opt.geomgroup[0]=0;opt.geomgroup[1]=1;opt.sitegroup[:]=0;images=[];segments=[];depths=[]
            for camera in ('agentview','robot0_eye_in_hand'):
                renderer.update_scene(dat,camera=camera,scene_option=opt);renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=1;assert model.vis.quality.offsamples==4;images.append(renderer.render()[:,::-1].copy());renderer.enable_segmentation_rendering();segments.append(renderer.render()[:,::-1].copy());renderer.disable_segmentation_rendering();renderer.enable_depth_rendering();depths.append(renderer.render()[:,::-1].copy());renderer.disable_depth_rendering()
            renderer.close();images=np.stack(images);segments=np.stack(segments);depths=np.stack(depths)
            if clean_rgb is None:clean_rgb=images;clean_seg=segments;clean_depth=depths
            else:np.testing.assert_array_equal(segments,clean_seg);np.testing.assert_array_equal(depths,clean_depth);assert np.any(images!=clean_rgb)
            Image.fromarray(np.concatenate(images,axis=1)).save(folder/(condition+'.png'));np.savez_compressed(folder/(condition+'_checks.npz'),segmentation=segments,depth=depths)
            checks.append(dict(condition=condition,RGB_MAE_vs_clean=float(abs(images.astype(float)-clean_rgb).mean()),geometry_match=True,segmentation_match=True,depth_match=True,XML_sha256=sha(text.encode()),changes=changes))
        comparable=known.shape[1]==len(state);exact=int(np.sum(np.all(known==state,axis=1))) if comparable else None
        records.append(dict(seed=seed,state_file=str(folder/'state.npy'),state_sha256=sha(state.tobytes()),initial_success=False,robot_pose_fixed=True,exact_matches_allowed_source_states=exact,source='official task reset/placement sampler; initialization_noise=None',conditions=checks,status='PASS_STATIC_NOT_POLICY',policy_actions=0))
        print('STATE',seed,'PASS',flush=True)
    finally:env.close()
save('evaluation_protocol.json',dict(status='FROZEN_STATIC_CHECKED',new_development_states=records,historical_development=[e['episode'] for e in data['episodes'] if e['split']=='validation'],models_future=['B20','C','D'],conditions=['clean','seen_background','unseen_warm_teal'],future_trials=3*20*3,seed_rule='environment seed equals state seed; reset saved exact state before each rollout',control_freq=20,max_steps=220,execution_chunk=8,success='env.check_success()',retry='invalid stops and is reported; no automatic rerun',independence='new reset seeds, no exact equality to allowed stored states if checked; not proof of independent distribution or no sealed-state overlap',policy_rollouts_this_run=0))
