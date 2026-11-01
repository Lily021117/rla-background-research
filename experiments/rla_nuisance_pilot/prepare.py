"""Outcome-blind metadata selection and CPU rendering only; no policy loading."""
import sys,os,copy,subprocess,collections
from pathlib import Path
import numpy as np
from PIL import Image
import mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_nuisance_pilot.common import *
def render(text,state):
 m=mujoco.MjModel.from_xml_string(text);d=mujoco.MjData(m);assert len(state)==1+m.nq+m.nv
 d.time=state[0];d.qpos[:]=state[1:1+m.nq];d.qvel[:]=state[1+m.nq:];mujoco.mj_forward(m,d)
 rr=mujoco.Renderer(m,224,224);opt=mujoco.MjvOption();opt.geomgroup[0]=0;opt.geomgroup[1]=1;opt.sitegroup[:]=0;rgb=[];depth=[];seg=[]
 try:
  for camera in ['agentview','robot0_eye_in_hand']:
   rr.update_scene(d,camera=camera,scene_option=opt);rr.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=1;rgb.append(rr.render()[:,::-1].copy());rr.enable_depth_rendering();depth.append(rr.render()[:,::-1].copy());rr.disable_depth_rendering();rr.enable_segmentation_rendering();seg.append(rr.render()[:,::-1].copy());rr.disable_segmentation_rendering()
 finally:rr.close()
 assert d.time==state[0];np.testing.assert_array_equal(d.qpos,state[1:1+m.nq]);np.testing.assert_array_equal(d.qvel,state[1+m.nq:])
 geom={k:np.array(getattr(m,k)).copy() for k in ['body_pos','body_quat','geom_pos','geom_quat','geom_size','geom_contype','geom_conaffinity','dof_damping','body_mass','body_inertia','cam_pos','cam_quat','cam_fovy']}
 return np.array(rgb),np.array(depth),np.array(seg),geom,m
def main():
 oldp=read(OLD/'evaluation/evaluation_protocol.json');allstates=read(oldp['frozen_state_protocol'])['new_development_states'];assert len(allstates)==50
 selected=sorted(allstates,key=lambda s:rank(s['seed']))[:25]
 save(ROOT/'state_selection.json',dict(rule='SHA256(UTF8("nuisance_pilot_v1:" + decimal state_id)), ascending first25; no outcome read',ranked=[dict(state_id=s['seed'],hash=rank(s['seed'])) for s in sorted(allstates,key=lambda s:rank(s['seed']))],selected=[s['seed'] for s in selected],role='existing evaluated states, new-appearance exploratory evaluation'))
 first=selected[0];cleantext=(Path(first['state_file']).parent/'clean.xml').read_text();initial=np.load(first['state_file']);tree=ET.fromstring(cleantext)
 save(ROOT/'preselection_rules.json',dict(background_candidates='generic texture *_COL*.png, nonbackup, first file per family; exclude every PNG source found in bounded historical protocol/manifest metadata; pair consecutive eligible families, all pairs considered',lighting_candidates='local tabletop_light_sync_modified_*.xml; import named light1/light2 attributes only; deduplicate canonical parameter lists',ranking='SHA256 nuisance_pilot_v1: + canonical candidate ID; first3 technically valid in each family',validity='finite RGB/depth, identical depth/segmentation/physical-camera arrays, agentview target bowl/plate pixels>0, image mean5..250 std>5, target-pixel mean>2, nonzero rendered change; no policy outcome',candidate_budget='all finite local metadata candidates; one selected-state static check each; no policy scoring',offline_anchors='20 evenly spaced complete anchors per each of5 existing development demos;100 total; original h1..8',operator_updates=0))
 used=set();sources=[]
 def collect(v):
  if isinstance(v,dict):
   for x in v.values():collect(x)
  elif isinstance(v,list):
   for x in v:collect(x)
  elif isinstance(v,str) and v.lower().endswith('.png') and Path(v).is_file():used.add(v)
 for parent,dirs,files in os.walk(P/'experiments'):
  dirs[:]=[d for d in dirs if d not in ['code','frames','closed_loop','episodes','features','targets','sidecars','downloads','cache','.git','projection_overlays','static_previews','assets','generated_states','source_models'] and 'sealed' not in d.lower() and 'rla_frozen_nuisance_pilot' not in d]
  for name in files:
   if name.endswith('.json') and any(k in name for k in ['protocol','manifest']) and 'sealed' not in name:
    p=Path(parent)/name
    if p.stat().st_size>5_000_000:continue
    try:collect(read(p));sources.append(str(p))
    except (ValueError,OSError):pass
 for node in tree.iter('texture'):
  if node.get('file'):used.add(node.get('file'))
 excluded={sha(p) for p in used};families={};assetroot=LIB/'libero/libero/assets/textures'
 for p in sorted(assetroot.glob('*_COL*.png')):
  if 'backup' not in p.name.lower():families.setdefault(p.name.split('_COL')[0],p)
 eligible=[(f,p,sha(p)) for f,p in sorted(families.items()) if sha(p) not in excluded]
 save(ROOT/'asset_exclusion_audit.json',dict(metadata_sources=sources,excluded=[dict(path=p,sha256=sha(p)) for p in sorted(used)],no_sealed_contents=True,scope='known project protocol/manifest references; absence of unrecorded manual use cannot be certified'))
 assets=ROOT/'assets';assets.mkdir();sizes={n.get('name'):Image.open(n.get('file')).size for n in tree.iter('texture') if n.get('name') in ['tex-table','tex-wall']};candidates=[]
 for i in range(0,len(eligible)-1,2):
  pair=eligible[i:i+2];ident='texture:'+':'.join(x[2] for x in pair);c=dict(candidate_id=ident,kind='background',textures={},sources=[])
  for name,(family,p,h) in zip(['tex-table','tex-wall'],pair):
   dest=assets/(rank(ident)+'_'+name+'.png');Image.open(p).convert('RGB').resize(sizes[name],Image.Resampling.LANCZOS).save(dest);assert sha(dest) not in excluded
   c['textures'][name]=str(dest);c['sources'].append(dict(family=family,path=str(p),sha256=h,resized_sha256=sha(dest),size=sizes[name]))
  candidates.append(c)
 seen=set()
 for p in sorted((LIB/'libero/libero/assets/scenes/lights').glob('tabletop_light_sync_modified_*.xml')):
  ls=[dict(n.attrib) for n in ET.parse(p).iter('light') if n.get('name') in ['light1','light2']]
  if len(ls)!=2:continue
  canonical=json.dumps(sorted(ls,key=lambda x:x['name']),sort_keys=True);h=hashlib.sha256(canonical.encode()).hexdigest()
  if h in seen:continue
  seen.add(h);candidates.append(dict(candidate_id='light:'+h,kind='lighting',lights=ls,source=str(p),source_sha256=sha(p)))
 save(ROOT/'candidate_metadata_inventory.json',candidates)
 candidates=[c for kind in ['background','lighting'] for c in sorted([x for x in candidates if x['kind']==kind],key=lambda x:rank(x['candidate_id']))[:12]]
 save(ROOT/'candidate_pool.json',dict(rule='before any rendering or scoring: first12 unique metadata candidates by fixed hash in each category; no extension if insufficient valid candidates',libero_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=LIB,text=True).strip(),candidates=candidates))
 ref,rd,rs,rg,rm=render(cleantext,initial);bowl=[i for i in range(rm.ngeom) if 'akita_black_bowl_1' in (rm.geom(i).name or '')];plate=[i for i in range(rm.ngeom) if 'plate_1' in (rm.geom(i).name or '')];target=(rs[0,:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))&np.isin(rs[0,:,:,0],bowl+plate);assert target.any()
 checks=[];valid=[];(ROOT/'previews').mkdir()
 for c in sorted(candidates,key=lambda x:rank(x['candidate_id'])):
  row=dict(candidate_id=c['candidate_id'],kind=c['kind'],rank=rank(c['candidate_id']),valid=False)
  try:
   image,depth,seg,geo,m=render(xml_for(cleantext,c),initial)
   for k in geo:np.testing.assert_array_equal(geo[k],rg[k],err_msg=k)
   np.testing.assert_array_equal(depth,rd);np.testing.assert_array_equal(seg,rs);assert np.isfinite(depth).all()
   assert 5<float(image.mean())<250 and image.std()>5 and float(image[0][target].mean())>2
   mae=float(np.abs(image.astype(float)-ref).mean());assert mae>0
   row.update(valid=True,RGB_MAE=mae,mean=float(image.mean()),std=float(image.std()),geometry_depth_segmentation_exact=True);valid.append(c)
   Image.fromarray(np.concatenate(image,axis=1)).save(ROOT/'previews'/(rank(c['candidate_id'])+'.png'))
  except Exception as e:row['reason']=repr(e)
  checks.append(row)
 save(ROOT/'candidate_technical_checks.json',checks)
 chosen=[]
 for kind in ['background','lighting']:
  pool=[c for c in valid if c['kind']==kind];assert len(pool)>=3,(kind,len(pool))
  for i,c in enumerate(pool[:3],1):chosen.append(dict(c,id=f'{kind}_{i}'))
 save(ROOT/'conditions.json',chosen)
 stateout=ROOT/'states';stateout.mkdir();new=[]
 for s in selected:
  folder=stateout/str(s['seed']);folder.mkdir();arr=np.load(s['state_file']);assert hashlib.sha256(arr.tobytes()).hexdigest()==s['state_sha256'];np.save(folder/'state.npy',arr)
  text=(Path(s['state_file']).parent/'clean.xml').read_text();assert hashlib.sha256(text.encode()).hexdigest()==s['clean_xml_sha256'];(folder/'clean.xml').write_text(text)
  cs=[dict(condition='clean',XML_sha256=sha(folder/'clean.xml'))]
  for c in chosen:
   p=folder/(c['id']+'.xml');p.write_text(xml_for(text,c));cs.append(dict(condition=c['id'],XML_sha256=sha(p)))
  new.append(dict(seed=s['seed'],state_file=str(folder/'state.npy'),state_sha256=s['state_sha256'],conditions=cs,original_state_file=s['state_file'],source_group=str(s['seed'])))
 save(ROOT/'state_protocol.json',dict(new_development_states=new,status='FROZEN',role='existing states, new appearance exploration'))
 registry=read(OLD/'evaluation/policy_registry.json');inventory=read(OLD/'evaluation/evaluation/formal/checkpoint_inventory.json')
 for k,p in registry.items():assert sha(p)==inventory[k]['sha256'] and inventory[k]['global_step']==1800
 save(ROOT/'checkpoint_inventory.json',inventory);save(ROOT/'evaluation/policy_registry.json',registry);plan=read(OLD/'evaluation/run_plan.json');save(ROOT/'evaluation/run_plan.json',plan)
 ep=dict(oldp,states=[s['seed'] for s in new],conditions=[c['id'] for c in chosen],trials=900,frozen_state_protocol=str(ROOT/'state_protocol.json'),role='existing states new appearance exploratory evaluation',generated_independent_states=False,nuisance_pilot=True)
 save(ROOT/'evaluation/evaluation_protocol.json',ep)
 # Only now read results, after outcome-blind state and appearance lists are fixed.
 oldrows=read(OLD/'evaluation/per_trial_results.json');reuse=[r for r in oldrows if r['condition']=='clean' and r['seed'] in ep['states']];assert len(reuse)==150
 by={s['seed']:s for s in new}
 for r in reuse:
  assert r['status']=='COMPLETE' and r['checkpoint_sha256']==inventory[r['model']]['sha256'] and r['state_sha256']==by[r['seed']]['state_sha256'] and r['xml_sha256']==by[r['seed']]['conditions'][0]['XML_sha256'] and r['language']==ep['task']
 save(ROOT/'clean_reuse.json',dict(source=str(OLD/'evaluation/per_trial_results.json'),source_sha256=sha(OLD/'evaluation/per_trial_results.json'),source_protocol_sha256=sha(OLD/'evaluation/evaluation_protocol.json'),same_deployment=True,device='same GPU0 UUID',rows=reuse))
 manifest=read(plan['inputs']['data_manifest']);dev=sorted([e for e in manifest['episodes'] if e['split']=='validation'],key=lambda e:e['episode']);assert len(dev)==5
 anchors=[]
 for e in dev:
  for t in np.linspace(0,e['valid_chunks']-1,20,dtype=int):anchors.append(dict(episode=e['episode'],t=int(t),root=e['root'],task=e['task'],source_demo=e['source_demo']))
 assert len(anchors)==100;save(ROOT/'offline_anchors.json',anchors)
 enc=read(BASE/'encoder_inventory.json')
 for a in ['C','D']:assert sha(enc[a]['path'])==enc[a]['sha256']
 save(ROOT/'encoder_inventory.json',enc);save(ROOT/'normalization_inventory.json',{a:dict(path=str(BASE/f'targets/{a}/statistics.json'),sha256=sha(BASE/f'targets/{a}/statistics.json')) for a in ['C','D']})
 save(ROOT/'protocol.json',dict(name='frozen C/D new appearance pilot',state_selection=read(ROOT/'state_selection.json'),condition_ids=ep['conditions'],primary='equal3background/equal3seed D-low minus C-low',secondary='equal3staticlighting/equal3seed D-low minus C-low; cross-intervention transfer, not trained lighting invariance',valid_new_rollouts=900,reused_clean=150,maximum_new_clean=0,offline_anchors=100,horizons=list(range(1,9)),offline='same saved pre-action state; Clean+6conditions; future only AE labels; no environment step; paired actions and per-demo summary; normalized and raw latent distances, normalized actions + decoded physical components',chunk='8 predicted and8 executed per replan; current step separately; no hypothetical1-step execution change',statistics='20000 paired physical-state resamples seed31001 preserve all models/seeds/conditions; excludes training-seed AE condition-sampling task uncertainty; offline aggregate by5source demos',selection_no_outcomes=True,optimizer_updates=0,AE_updates=0,DINO_updates=0,backward=0,historical_negative_results_unchanged=True,no_sealed_reads=True,stop='hardware/numerical error stop; preserve invalid; no silent retries or valid failure reruns'))
 print('PREPARED',len(candidates),'candidates;900new trials;100anchors',flush=True)
if __name__=='__main__':main()
