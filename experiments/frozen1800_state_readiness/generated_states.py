"""Route B: frozen native placement sampler, CPU technical acceptance only."""
import argparse, hashlib, json, os, random, subprocess, time, traceback
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

P=Path(os.environ['RLA_PROJECT'])
L=Path(os.environ['LIBERO_ROOT'])
SOURCE=P/'experiments/rla_low_lr_continuation/run_20261030_01'
OLD=P/'experiments/rla_appearance_gap_eval/run_20261027_01'
DATA=P/'experiments/official_libero_rla_consistency/run_20261023_10/stove'
TASK='pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate'
BODIES=['akita_black_bowl_1_main','akita_black_bowl_2_main','plate_1_main','cookies_1_main','glazed_rim_porcelain_ramekin_1_main','wooden_cabinet_1_main','flat_stove_1_main']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2)
def prepare(root):
 bddl=L/f'libero/libero/bddl_files/libero_spatial/{TASK}.bddl'
 files=[bddl,L/'libero/libero/envs/bddl_base_domain.py',L/'libero/libero/envs/env_wrapper.py',L/'libero/libero/envs/regions/base_region_sampler.py',L/'libero/libero/envs/regions/workspace_region_sampler.py',L/'libero/libero/envs/objects/site_object.py',L/'libero/libero/envs/problems/libero_tabletop_manipulation.py']
 assets=read(OLD/'asset_manifest.json')
 for x in assets['assets']:assert sha(x['output'])==x['output_sha256']
 prior=sorted(root.parent.glob('**/generated_states/seed_*'))
 start=2026111000+len(prior)
 plan=dict(version=3,route='B',name='按冻结初始化分布新生成的仿真状态验证',created=time.time(),
  environment_commit=subprocess.check_output(['git','-C',str(L),'rev-parse','HEAD'],text=True).strip(),
  source_files={str(p):sha(p) for p in files},bddl=str(bddl),seed_sequence=list(range(start,2026111500)),
  candidate_budget=2026111500-start,prior_candidate_attempts=len(prior),total_candidate_cap_across_revisions=500,target_N=50,minimum_N=20,initialization_noise=None,
  settling=dict(steps=10,action=[0,0,0,0,0,0,-1],frequency_Hz=20,source='official run_libero_eval num_steps_wait10 and get_libero_dummy_action; no policy'),
  distribution='Unmodified BDDL placement regions and native _reset_internal placement/conditional samplers. Fixed robot reset. No noise added to historical states.',
  reset='seed Python/NumPy before construction; env.seed(seed); call underlying env.reset once (no wrapper unbounded RandomizationError retry); every outer candidate retained',
  validity=['finite state','native sampler succeeds','official goal false','target bowl on stove and other bowl on cabinet predicates true; plate/cookies/ramekin contact table_collision',
            'passive stability on separate MjData: 0.2 seconds, movable objects maximum displacement <=0.03m and last0.05s displacement <=0.002m; accepted state remains original reset snapshot',
            'not duplicate against known25 train/development starts,20 prior reset states,earlier accepted candidates'],
  duplicate_rule=dict(bodies=BODIES,position_max_abs_m=0.0001,rotation_matrix_max_abs=0.0001,robot_joint_max_abs=0.00001,time_excluded=True,
   note='Conservative whole-layout comparison with named bodies and robot joints, not file hash. Also reject matching qpos against official50 init entries; qpos non-match alone never certifies old-source independence.'),
  conditions=['clean','new_background_1','new_background_2','new_background_3','new_background_4'],
  asset_manifest=str(OLD/'asset_manifest.json'),asset_manifest_sha256=sha(OLD/'asset_manifest.json'),
  intervention='Only tex-table and tex-wall file attributes; reflection1/MSAA4; original textures reused',
  scoring=False,optimizer_updates=0,backward=0,policy_forward=0,
  revisions='Before any policy scoring: v1 tested contacts before settling; v2 additionally required every object center inside narrow BDDL named region. Actual native sampler boundary correction uses object radius and can put centers outside the narrow site despite stable legal table contact (e.g plate,cookies). v3 keeps native distribution unchanged, checks required bowl supports and physical table contact, logs all predicates but does not require tiny table sites. Preserve all previous candidates and count them against500; remaining seeds only. No policy result used.',
  failures='Preserve candidates/reasons; no replacement beyond fixed500seed list; infra failure stops, does not relabel policy failures')
 save(root/'generation_protocol.json',plan)
 print('GENERATION_PROTOCOL_FROZEN',sha(root/'generation_protocol.json'))

def generate(root):
 import mujoco
 from PIL import Image
 from libero.libero.envs.env_wrapper import ControlEnv
 from robosuite.utils.errors import RandomizationError
 plan=read(root/'generation_protocol.json')
 for p,h in plan['source_files'].items():assert sha(p)==h
 assert sha(plan['asset_manifest'])==plan['asset_manifest_sha256']
 asset=read(plan['asset_manifest']);configs=[dict(id='clean',textures={})]+asset['configurations']
 for a in asset['assets']:assert sha(a['output'])==a['output_sha256']
 def data_for(m,state):
  d=mujoco.MjData(m);assert len(state)==1+m.nq+m.nv;d.time=state[0];d.qpos[:]=state[1:1+m.nq];d.qvel[:]=state[1+m.nq:];mujoco.mj_forward(m,d);return d
 def signature(m,d):
  ids=[m.body(n).id for n in BODIES];joints=['robot0_joint'+str(i) for i in range(1,8)]+['gripper0_finger_joint1','gripper0_finger_joint2']
  return dict(position=d.xpos[ids].copy(),rotation=d.xmat[ids].copy(),robot=d.qpos[[m.joint(n).qposadr[0] for n in joints]].copy())
 def duplicate(a,b):return (np.max(abs(a['position']-b['position']))<=1e-4 and np.max(abs(a['rotation']-b['rotation']))<=1e-4 and np.max(abs(a['robot']-b['robot']))<=1e-5)
 known=[];known_sources=[]
 sel=read(DATA/'dataset_selection.json');ds=read(DATA/'frozen_dataset_manifest.json')
 for e in sel['episodes']:
  record=next(x for x in ds['episodes'] if x['episode']==e['episode']);state=np.load(Path(record['root'])/'trajectory_labels.npz')['states'][0]
  m=mujoco.MjModel.from_xml_path(str(Path(e['model_dir'])/'clean.xml'));d=data_for(m,state)
  known.append(signature(m,d));known_sources.append(e['source_demo'])
 oldstates=read(OLD/'development_protocol/evaluation_protocol.json')['new_development_states']
 for e in oldstates:
  state=np.load(e['state_file']);m=mujoco.MjModel.from_xml_path(str(Path(e['state_file']).parent/'clean.xml'));d=data_for(m,state)
  known.append(signature(m,d));known_sources.append('development_reset_'+str(e['seed']))
 import torch
 official=torch.load(L/f'libero/libero/init_files/libero_spatial/{TASK}.pruned_init',map_location='cpu',weights_only=False)
 save(root/'generation_content_reads.json',dict(sealed_contents_read=True,scope='Official init container50 qpos for conservative duplicate exclusion only; current25 train/dev first states and20 historical development reset states',known_source_ids=known_sources,policy_scoring=False))
 folder=root/'generated_states';folder.mkdir();accepted=[];allrows=[]
 for ordinal,seed in enumerate(plan['seed_sequence']):
  dest=folder/f'seed_{seed}';dest.mkdir();row=dict(seed=seed,ordinal=ordinal,status='REJECTED',reason=None,policy_actions=0);env=None
  try:
   random.seed(seed);np.random.seed(seed)
   env=ControlEnv(bddl_file_name=plan['bddl'],use_camera_obs=False,has_offscreen_renderer=False,has_renderer=False,control_freq=20,horizon=1000,initialization_noise=None,ignore_done=False)
   env.seed(seed);env.env.reset();env.sim.forward()
   np.save(dest/'raw_reset_state.npy',env.get_sim_state().copy())
   for _ in range(plan['settling']['steps']):env.step(plan['settling']['action'])
   env.sim.forward();state=env.get_sim_state().copy();m,d=env.sim.model._model,env.sim.data._data
   np.save(dest/'state.npy',state);text=env.sim.model.get_xml();(dest/'clean.xml').write_text(text)
   row['state_sha256']=hashlib.sha256(state.tobytes()).hexdigest();row['state_file']=str(dest/'state.npy');row['clean_xml_sha256']=sha(dest/'clean.xml')
   sig=signature(m,d);np.savez(dest/'physical_signature.npz',**sig)
   finite=bool(np.isfinite(state).all());goal=bool(env.check_success());pred=[dict(predicate=x,valid=bool(env.env._eval_predicate(x))) for x in env.env.parsed_problem['initial_state']]
   row.update(finite=finite,initial_success=goal,initial_predicates=pred)
   supports={}
   for name in ('plate_1','cookies_1','glazed_rim_porcelain_ramekin_1'):
    supports[name]=any((m.geom(c.geom1).name=='table_collision' and str(m.geom(c.geom2).name).startswith(name+'_')) or (m.geom(c.geom2).name=='table_collision' and str(m.geom(c.geom1).name).startswith(name+'_')) for c in d.contact)
   bowl_support=all(x['valid'] for x in pred if x['predicate'][1] in ('akita_black_bowl_1','akita_black_bowl_2'))
   row['physical_support_contacts']=supports
   if not finite or goal or not bowl_support or not all(supports.values()):row['reason']='invalid_support_or_goal_or_nonfinite'
   probe=data_for(m,state);ids=[m.body(n).id for n in BODIES[:5]];start=probe.xpos[ids].copy();positions=[]
   for _ in range(round(.2/m.opt.timestep)):
    mujoco.mj_step(m,probe);positions.append(probe.xpos[ids].copy())
   positions=np.asarray(positions);drift=float(np.max(np.linalg.norm(positions-start,axis=-1)));last=float(np.max(np.linalg.norm(positions[-1]-positions[-max(1,round(.05/m.opt.timestep))],axis=-1)))
   row.update(stability_max_displacement_m=drift,stability_last_displacement_m=last,stability_steps=len(positions))
   if not np.isfinite(positions).all() or drift>.03 or last>.002:row['reason']=row['reason'] or 'passive_stability_failed'
   matches=[known_sources[i] for i,k in enumerate(known) if duplicate(sig,k)]
   qmatch=[i for i,s in enumerate(official) if np.shape(s)==state.shape and np.max(abs(np.asarray(s)[1:1+m.nq]-state[1:1+m.nq]))<=1e-5]
   row.update(known_physical_duplicates=matches,official_qpos_duplicates=qmatch)
   if matches or qmatch:row['reason']=row['reason'] or 'physical_duplicate'
   # All checks above are technical, before any policy loading.
   if row['reason'] is None:
    checks=[];base=None;cleanimg=None
    for config in configs:
     tree=ET.fromstring(text);changes=[]
     for name,path in config['textures'].items():
      nodes=[n for n in tree.iter('texture') if n.get('name')==name];assert len(nodes)==1
      changes.append(dict(texture=name,old=nodes[0].get('file'),new=path));nodes[0].set('file',path)
     rendered_xml=ET.tostring(tree,encoding='unicode') if changes else text;xp=dest/(config['id']+'.xml')
     if config['id']!='clean':xp.write_text(rendered_xml)
     for change in changes:next(n for n in tree.iter('texture') if n.get('name')==change['texture']).set('file',change['old'])
     assert ET.tostring(tree)==ET.tostring(ET.fromstring(text))
     mm=mujoco.MjModel.from_xml_string(rendered_xml);dd=data_for(mm,state);geom={k:np.array(getattr(dd,k)).copy() for k in ('qpos','qvel','xpos','xmat','cam_xpos','cam_xmat')}
     if base is None:base=geom
     else:
      for k in geom:np.testing.assert_allclose(geom[k],base[k],rtol=0,atol=1e-8)
     assert mm.vis.quality.offsamples==4
     renderer=mujoco.Renderer(mm,224,224);opt=mujoco.MjvOption();opt.geomgroup[0]=0;opt.geomgroup[1]=1;opt.sitegroup[:]=0;images=[];depth=[];seg=[]
     try:
      for cam in ('agentview','robot0_eye_in_hand'):
       renderer.update_scene(dd,camera=cam,scene_option=opt);renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=1;images.append(renderer.render()[:,::-1].copy())
       renderer.enable_depth_rendering();depth.append(renderer.render().copy());renderer.disable_depth_rendering()
       renderer.enable_segmentation_rendering();seg.append(renderer.render().copy());renderer.disable_segmentation_rendering()
     finally:renderer.close()
     images=np.stack(images);depth=np.stack(depth);seg=np.stack(seg)
     if cleanimg is None:cleanimg=images;cleandepth=depth;cleanseg=seg
     else:np.testing.assert_array_equal(depth,cleandepth);np.testing.assert_array_equal(seg,cleanseg);assert np.any(images!=cleanimg)
     Image.fromarray(np.concatenate(images,axis=1)).save(dest/(config['id']+'.png'))
     checks.append(dict(condition=config['id'],XML_sha256=sha(xp),changes=changes,geometry_match=True,segmentation_match=True,depth_match=True,RGB_MAE_vs_clean=float(abs(images.astype(float)-cleanimg).mean())))
    row.update(status='ACCEPTED',reason='all_frozen_technical_checks_pass',conditions=checks,source='native BDDL reset placement; new generated state',source_group=str(seed))
    accepted.append(row);known.append(sig);known_sources.append('new_'+str(seed))
   save(dest/'acceptance.json',row);allrows.append(row);print(seed,row['status'],row['reason'],len(accepted),flush=True)
  except RandomizationError:
   row['reason']='native_sampler_RandomizationError';row['error']=traceback.format_exc();save(dest/'acceptance.json',row);allrows.append(row)
  except Exception:
   row.update(status='INFRASTRUCTURE_STOP',error=traceback.format_exc());save(dest/'technical_error.json',row);raise
  finally:
   if env:env.close()
  if len(accepted)==plan['target_N']:break
 save(root/'candidate_results.json',allrows)
 save(root/'state_protocol.json',dict(status='READY_FOR_PROTOCOL_COMMIT' if len(accepted)>=20 else 'BLOCKED_LT20',new_development_states=accepted,role=plan['name'],N=len(accepted),attempted_candidates=len(allrows),policy_rollouts=0,source_groups=[x['source_group'] for x in accepted],generation_protocol_sha256=sha(root/'generation_protocol.json')))
 print('FINAL_N',len(accepted),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','generate']);p.add_argument('--root',type=Path,required=True);a=p.parse_args();globals()[a.stage](a.root)
