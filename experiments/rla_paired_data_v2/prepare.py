"""Bounded source admission. No model inference, RGB source reads or simulation step."""
import sys,copy,importlib.util,xml.etree.ElementTree as ET,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_paired_data_v2.common import *
import h5py,mujoco
from PIL import Image

def relocate(text):
 tree=ET.fromstring(text);assets=[];robo=Path(importlib.util.find_spec('robosuite').origin).parent
 for node in tree.iter():
  v=node.get('file')
  if not v:continue
  if '/chiliocosm/assets/' in v:q=LIB/'libero/libero/assets'/v.split('/chiliocosm/assets/',1)[1]
  elif '/libero/assets/' in v:q=LIB/'libero/libero/assets'/v.split('/libero/assets/',1)[1]
  elif 'robosuite' in Path(v).parts:
   parts=Path(v).parts;i=max(i for i,n in enumerate(parts) if n=='robosuite');q=robo.joinpath(*parts[i+1:])
  else:q=Path(v)
  q=q.resolve();assert q.is_file(),q;node.set('file',str(q));assets.append(dict(local=str(q),sha256=sha(q)))
 return tree,assets

def appearance(tree,task,variant,out):
 root=copy.deepcopy(tree);changes=[]
 if variant.startswith('object'):
  bname='akita_black_bowl_1_main' if task=='stove' else 'wooden_cabinet_1_cabinet_middle'
  body=next(n for n in root.iter('body') if n.get('name')==bname)
  geoms=[g for g in body.iter('geom') if g.get('group')=='1' and g.get('material')];assert geoms
  asset=root.find('asset');cloned={}
  for g in geoms:
   old=g.get('material')
   if old not in cloned:
    mat=copy.deepcopy(next(n for n in root.iter('material') if n.get('name')==old));mat.set('name',old+'_v2_'+variant)
    mat.set('specular','0' if variant=='object_1' else '0.7');mat.set('shininess','0' if variant=='object_1' else '0.8');mat.set('reflectance','0' if variant=='object_1' else '0.5')
    oldtex=mat.get('texture');tex=copy.deepcopy(next(n for n in root.iter('texture') if n.get('name')==oldtex));tex.set('name',oldtex+'_v2_'+variant)
    image=np.asarray(Image.open(tex.get('file')).convert('RGB'),dtype=np.float32);yy,xx=np.indices(image.shape[:2]);pattern=.9+.1*np.sin(2*np.pi*xx/32)
    if task=='stove':
     # Retain a dark, neutral black bowl; this is surface albedo, not sensor noise.
     gray=image.mean(2,keepdims=True);image=np.repeat(gray,3,axis=2)
     image=image*.85 if variant=='object_1' else image*.95*pattern[:,:,None]
    else:image=image*.65 if variant=='object_1' else image*1.15*pattern[:,:,None]
    path=out/(oldtex+'_'+variant+'.png');Image.fromarray(np.clip(image,0,255).astype(np.uint8)).save(path)
    tex.set('file',str(path));mat.set('texture',tex.get('name'));asset.append(tex);asset.append(mat);cloned[old]=mat.get('name')
    changes.append(dict(original_material=old,new_material=mat.attrib,texture=tex.attrib,texture_sha256=sha(path)))
   g.set('material',cloned[old])
  # No edits to body/geom physics or existing materials. Reversibility checks XML whitelist.
  restored=copy.deepcopy(root)
  for g in restored.iter('geom'):
   for old,new in cloned.items():
    if g.get('material')==new:g.set('material',old)
  for n in list(restored.find('asset')):
   if n.get('name','').endswith('_v2_'+variant):restored.find('asset').remove(n)
  assert ET.tostring(restored)==ET.tostring(tree)
 elif variant.startswith('light'):
  for light in root.iter('light'):
   before=dict(light.attrib)
   if variant=='light_intensity':
    for key,default in [('diffuse','0.7 0.7 0.7'),('specular','0.3 0.3 0.3'),('ambient','0 0 0')]:light.set(key,' '.join(format(.65*float(v),'.12g') for v in light.get(key,default).split()))
   else:
    direction=np.fromstring(light.get('dir','0 0 -1'),sep=' ');a=np.deg2rad(30);R=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]]);light.set('dir',' '.join(format(v,'.12g') for v in R@direction))
   changes.append(dict(name=light.get('name'),before=before,after=dict(light.attrib)))
  restored=copy.deepcopy(root)
  for n,c in zip(restored.iter('light'),changes):n.attrib.clear();n.attrib.update(c['before'])
  assert ET.tostring(restored)==ET.tostring(tree)
 return ET.tostring(root,encoding='unicode'),changes

def main():
 ROOT.mkdir(exist_ok=True)
 rules=dict(version='training-prototype-v2',tasks=['stove_bowl_to_plate','open_middle_drawer'],max_candidates_per_task=20,max_sources=10,max_complete_render_versions=50,max_open_loop_replays=50,planned_open_loop_replays=0,selection='numeric unused legal source ID first; then numeric historical train fallback; never development/protected; stop when5 pass',partition='training-prototype only, not train/dev/test split',conditions=VARIANTS,camera_order=CAMS,resolution=[224,224],control_hz=20,reflection=1,MSAA=4,small_sample='first selected source/task; indices floor(linspace(0,N-1,5))',geometry_tolerances=dict(state=0,compiled_physics=0,FK_camera=0,depth_m=1e-6,segmentation_pixels=0),image_gate='fixed sample: image mean in(3,252),std>3; if Clean ROI>=4px and mean>3, variant ROI mean>=max(2,.35*Clean) and<250; legitimate Clean occlusion is missing visibility not variant failure; changed RGB>0 somewhere; no policy selection',model_forwards=0,training_updates=0,backward=0,policy_rollouts=0,simulation_integration_steps=0,light_intensity='multiply diffuse/specular/ambient by0.65; preserve hue and direction',light_direction='rotate each existing light direction30deg about world Y; no position/intensity change',object_whitelist=['clone only target visual materials and textures','target visual geom material reference','material specular/shininess/reflectance','texture file, same geometry/UV'],forbidden=['mass','friction','collision','geometry','camera','dynamics','task goal','source actions','source states'],runtime=dict(mujoco=mujoco.__version__,libero_commit=subprocess.check_output(['git','-C',str(LIB),'rev-parse','HEAD'],text=True).strip()),source_code=dict(path=str(LIB/'scripts/create_dataset.py'),sha256=sha(LIB/'scripts/create_dataset.py'),semantics='env.step(actions[j]) then obs; writes states[valid_index] with same actions index; regenerate at saved pre-action states; no settling added to state before render'))
 save(ROOT/'prototype_protocol.json',rules);entries=[];candidates=[];inventories={};configs={};assets_all=[]
 for task in ['stove','drawer']:
  manifest=read(frozen_source(task));prior=manifest['episodes'];selection=read((STOVE if task=='stove' else DRAWER)/'dataset_selection.json');source=Path(prior[0]['source_hdf5']);used={e['source_demo']:e['split'] for e in prior};protected={x['source_demo'] for x in selection['protected_sources'] if Path(x['source_hdf5'])==source}
  old_hashes={e['source_actions_sha256'] for e in prior};accepted=[];seen=[]
  with h5py.File(source,'r') as f:
   ids=sorted(f['data'],key=lambda x:int(x.split('_')[-1]));new=[i for i in ids if i not in used and i not in protected];fallback=sorted([i for i in ids if used.get(i)=='train'],key=lambda x:int(x.split('_')[-1]));pool=(new[:15]+fallback)[:20]
   inventories[task]=dict(source=str(source),source_sha256=sha(source),total_source_episodes=len(ids),old_train=sum(x=='train' for x in used.values()),old_development=sum(x=='validation' for x in used.values()),protected_count=len(protected),unassigned_unprotected_ids=new,unassigned_unprotected_count=len(new),candidate_pool=pool,known_history_manifest=str(frozen_source(task)),history_manifest_sha256=sha(frozen_source(task)),relationship='same original HDF5+demo identity; no Cosmos ID join; prior source audit may have read content but not necessarily trained/scored; unexamined candidates NOT certified independent additions',global_metadata={k:str(v) for k,v in f['data'].attrs.items()})
   for demo in pool:
    if len(accepted)==5:break
    row=dict(task=task,source_demo=demo,historical_usage=used.get(demo,'unassigned_unprotected'),status='PENDING',reason='')
    try:
     d=f['data'][demo];raw=d.attrs['model_file'];raw=raw.decode() if isinstance(raw,bytes) else str(raw);states=d['states'][:];actions=d['actions'][:];N=len(states)
     assert N>8 and actions.shape==(N,7) and np.isfinite(states).all() and np.isfinite(actions).all()
     assert np.max(abs(np.diff(states[:,0])-.05))<1e-9,'time discontinuity'
     tree,assets=relocate(raw);model=mujoco.MjModel.from_xml_string(ET.tostring(tree,encoding='unicode'));assert model.na==0 and states.shape[1]==1+model.nq+model.nv
     arm=[model.joint('robot0_joint'+str(i)).qposadr[0] for i in range(1,8)];fingers=[model.joint('gripper0_finger_joint'+str(i)).qposadr[0] for i in [1,2]]
     joint_error=float(np.max(abs(states[1:,1+np.array(arm)]-d['obs/joint_states'][:-1])));finger_error=float(np.max(abs(states[1:,1+np.array(fingers)]-d['obs/gripper_states'][:-1])));assert joint_error<=1e-8 and finger_error<=1e-8,'stored-state / post-action observation mismatch'
     assert all(d['obs/'+k].shape[0]==N for k in ['agentview_rgb','eye_in_hand_rgb','ee_states'])
     assert all(model.cam(c).id>=0 for c in CAMS) and model.vis.quality.offsamples==4
     ah=digest(actions.astype(np.float32));assert ah not in seen
     if demo not in used:assert ah not in old_hashes,'same action sequence as historical source; ancestor ambiguity'
     seen.append(ah);ep=f'v2_{task}_{demo}';folder=ROOT/'sources'/ep;folder.mkdir(parents=True);(folder/'source_model.xml').write_text(raw)
     np.savez_compressed(folder/'source_arrays.npz',states=states,actions=actions)
     variant_assets=folder/'assets';variant_assets.mkdir();variant_configs={}
     for variant in VARIANTS:
      text,changes=appearance(tree,task,variant,variant_assets);p=folder/(variant+'.xml');p.write_text(text);variant_configs[variant]=dict(xml=str(p),xml_sha256=sha(p),changes=changes)
     configs[ep]=variant_configs;assets_all+=assets
     entry=dict(episode=ep,source_id=f'{source.name}::{demo}',source_hdf5=str(source),source_hdf5_sha256=inventories[task]['source_sha256'],source_demo=demo,task=task,language=manifest['episodes'][0]['task'],source_states_sha256=digest(states),source_actions_sha256=ah,source_xml_sha256=hashlib.sha256(raw.encode()).hexdigest(),historical_usage=row['historical_usage'],identity='new_to_training_pool' if demo not in used else 'reused_training',split='training-prototype',frames=N,valid_chunks=N-8,source_RGB_used=False,source_obs_resolution=list(d['obs/agentview_rgb'].shape[1:]),state_obs_joint_max_error=joint_error,state_obs_finger_max_error=finger_error,start_time=float(states[0,0]),end_time=float(states[-1,0]),state_sampling_seconds=.05,stable_wait_added=0,root=str(folder),time_semantics=rules['source_code']['semantics'])
     accepted.append(entry);entries.append(entry);row.update(status='ACCEPTED',reason='source XML/layout, temporal interval and next-state/post-action proprio verified; no source RGB reused')
    except Exception as ex:row.update(status='REJECTED',reason=str(ex))
    candidates.append(row)
   assert len(accepted)==5,(task,'cannot fill prototype within20 candidates')
  inventories[task]['selected_new']=sum(e['identity']=='new_to_training_pool' for e in accepted);inventories[task]['selected_reused']=sum(e['identity']=='reused_training' for e in accepted);inventories[task]['remaining_unexamined_potential']=len(new)-sum(r['task']==task and r['historical_usage']=='unassigned_unprotected' for r in candidates)
 save(ROOT/'source_inventory.json',inventories);save(ROOT/'source_manifest.json',entries);table(ROOT/'source_manifest.csv',entries);table(ROOT/'candidate_admission.csv',candidates);save(ROOT/'appearance_configs.json',configs)
 save(ROOT/'asset_inventory.json',list({r['local']:r for r in assets_all}.values()));save(ROOT/'pre_render_acceptance.json',dict(status='PASS',sources=10,independent_source_keys=10,candidates_checked=len(candidates),new=sum(e['identity']=='new_to_training_pool' for e in entries),reused=sum(e['identity']=='reused_training' for e in entries),sealed_contents_read=False,physics_steps=0,model_forwards=0))
 print('PREPARED',[(e['episode'],e['frames'],e['identity']) for e in entries],flush=True)
if __name__=='__main__':main()
