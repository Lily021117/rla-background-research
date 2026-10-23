"""Select by source completeness and manipulation context, never policy scores."""
import argparse,json,csv,hashlib,xml.etree.ElementTree as ET,importlib.util
from pathlib import Path
import numpy as np,h5py,mujoco
p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--libero',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
def save(path,x):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(x,indent=2))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
base=a.project/'experiments/official_libero_rla_consistency';old=base/'run_20261023_09';splits=json.loads((a.project/'experiments/rla_background_invariance/step2_training/split_manifest.json').read_text())['splits']
released={(str(Path(x['hdf5']).resolve()),x['demo']) for x in json.loads((a.project/'artifacts/action_conditioned_wm/TZ_three_seed_500/independent_data/selection.json').read_text())['demonstrations']}
sealed={(str(Path(x['raw_hdf5']).resolve()),x['raw_demo']) for x in json.loads((a.project/'artifacts/action_prediction_evidence/AE500_robustness_and_census/data_census_v2/sealed_evaluation_candidates.json').read_text())['demonstrations'] if x.get('raw_hdf5') and (str(Path(x['raw_hdf5']).resolve()),x['raw_demo']) not in released}
protected=sealed|{(str(Path(x['source_hdf5']).resolve()),x['source_demo']) for x in splits['test']}
previous={(str(Path(x['source_hdf5']).resolve()),x['source_demo']):s for s in ('train','validation') for x in splits[s]}
sources=sorted((a.project/'data/libero_original_minimal/libero_spatial').glob('*.hdf5'));inventory=[];candidates=[]
for path in sources:
 context='confined_open_drawer_retrieval' if 'top_drawer' in path.name else 'stovetop_fixture_retrieval' if 'stove' in path.name else 'elevated_ramekin_support' if 'ramekin' in path.name else 'historical_cookie_box'
 with h5py.File(path,'r') as f:
  for demo in sorted(f['data'],key=lambda x:int(x.split('_')[-1])):
   key=(str(path.resolve()),demo);d=f['data'][demo];guard=key in protected
   complete=all(x in d for x in ('states','actions','obs/agentview_rgb','obs/eye_in_hand_rgb','obs/ee_states','obs/gripper_states','obs/joint_states')) and 'model_file' in d.attrs
   row=dict(source=str(path),demo=demo,context=context,protected=guard,metadata_complete=complete,frames=int(d['actions'].shape[0]) if 'actions' in d else 0,status='PROTECTED_METADATA_ONLY' if guard else 'METADATA_COMPLETE' if complete else 'MISSING_FIELDS',contents_read=False,previous_split=previous.get(key,'unassigned'))
   inventory.append(row)
  eligible=sum(r['source']==str(path) and not r['protected'] and r['metadata_complete'] for r in inventory)
  candidates.append(dict(source=str(path),context=context,eligible_by_metadata=eligible,total=len(f['data'])))
priority={'confined_open_drawer_retrieval':0,'stovetop_fixture_retrieval':1,'elevated_ramekin_support':2,'historical_cookie_box':99}
chosen=sorted((x for x in candidates if x['context']!='historical_cookie_box' and x['eligible_by_metadata']>=25),key=lambda x:priority[x['context']])[:2]
assert len(chosen)==2,'Not enough distinct complete nonsealed tasks; no model-based replacement'
seen=json.loads((base/'run_20261023_04/dataset_selection.json').read_text())['background_texture_files'];assets=a.libero/'libero/libero/assets';robo=Path(importlib.util.find_spec('robosuite').origin).parent
protocol=dict(frozen_before_any_new_model_scoring=True,primary='D-C per task and task-equal average',secondary=['C-B','D-B'],historical_task=str(old/'report.md'),historical_states_role='retain prior20 as regression result; no rerun or tuning',selection_rule='metadata completeness>=25 nonprotected demos; predeclared context priority drawer then stove then ramekin; no model outcomes',task_candidates=candidates,selected_contexts=[x['context'] for x in chosen],limits='All available original tasks are bowl pick-and-place; differences are confined retrieval and support context, not new action primitives',seed=20261023,train_demos=20,development_demos=5,AE_updates=600,policy_updates=600,lambda_cons=.009127798272072366,lambda_aux=.22408536585365854,AE_initial='official LIBERO step90000',policy_initial='original B20 step0; never trained B20',normalization='per-task train-only; shared B/C/D action and proprio; common AE-C train-only RLA normalization for C/D',AE_plan=str(base/'run_20261023_08/training_plan.json'),policy_plan=str(old/'run_plan.json'),conditions=['clean','seen_background','unseen_background'],unseen_palette=[{'id':'warm_teal','tex-table':[178,132,82],'tex-wall':[84,127,132]},{'id':'cool_navy','tex-table':[87,117,154],'tex-wall':[143,151,164]},{'id':'muted_rose','tex-table':[149,105,117],'tex-wall':[115,142,121]}],palette_assignment='state ordinal modulo3; frozen, no policy score selection',texture_changes_only=['tex-table','tex-wall'],reflection=1,MSAA=4,max_control_steps=220,chunk_execution=8,valid_trials_per_task=180,total_new_valid_trials=360,invalid_rule='retain original infrastructure failure and stop for diagnosis; retries only after identified infrastructure repair, same model/state/condition, never task failures',heldout_read=False,task_equal_statistics='three fixed tasks descriptive; state clusters within task, task heterogeneity explicit; no claim across all LIBERO tasks',future_seeds=[20261023,20261024,20261025],future_seed_scope='retrain both AE and policies with matching seeds; policy-only seed variation is conditional on fixed AE')
save(a.output/'extension_protocol.json',protocol)
for i,info in enumerate(chosen):
 slug='drawer' if 'drawer' in info['context'] else 'stove' if 'stove' in info['context'] else 'ramekin';out=a.output/slug;out.mkdir();pool=out/'source_models';pool.mkdir();available=[];seen_actions=set();source=Path(info['source']);task=source.name.removesuffix('_demo.hdf5').replace('_',' ')
 with h5py.File(source,'r') as f:
  for row in [x for x in inventory if x['source']==str(source) and x['status']=='METADATA_COMPLETE']:
   d=f['data'][row['demo']]
   try:
    s=d['states'][:];ac=d['actions'][:].astype(np.float32);row['contents_read']=True;N=len(s);assert ac.shape==(N,7) and N>8 and np.isfinite(s).all() and np.isfinite(ac).all();assert np.allclose(np.diff(s[:,0]),.05,rtol=0,atol=1e-9)
    assert d['obs/agentview_rgb'].shape==d['obs/eye_in_hand_rgb'].shape==(N,128,128,3)
    ah=digest(ac);assert ah not in seen_actions,'duplicate action sequence';seen_actions.add(ah);raw=str(d.attrs['model_file']);xml=ET.fromstring(raw);relocations=[]
    for n in xml.iter():
     value=n.get('file')
     if not value:continue
     dest=value
     if '/chiliocosm/assets/' in value:dest=str(assets/value.split('/chiliocosm/assets/',1)[1])
     elif 'robosuite' in Path(value).parts:
      parts=Path(value).parts;idx=max(j for j,x in enumerate(parts) if x=='robosuite');dest=str(robo.joinpath(*parts[idx+1:]))
     assert Path(dest).is_file(),'missing asset '+dest
     if dest!=value:n.set('file',dest);relocations.append(dict(original=value,local=dest))
    clean=ET.tostring(xml,encoding='unicode');bg=ET.fromstring(clean)
    for name,value in seen.items():next(n for n in bg.iter('texture') if n.get('name')==name).set('file',value)
    modeldir=pool/row['demo'];modeldir.mkdir();(modeldir/'clean.xml').write_text(clean);(modeldir/'background.xml').write_text(ET.tostring(bg,encoding='unicode'));m=mujoco.MjModel.from_xml_path(str(modeldir/'clean.xml'));assert m.na==0 and s.shape[1]==1+m.nq+m.nv and m.vis.quality.offsamples==4;assert m.cam('agentview').id>=0 and m.cam('robot0_eye_in_hand').id>=0
    e=dict(episode=f'derived_{slug}_{row["demo"]}_v1',episode_id=f'derived_{slug}_{row["demo"]}_v1',source_hdf5=str(source),source_demo=row['demo'],task=task,frames=N,complete_h8_chunks=N-8,source_actions_sha256=ah,source_states_sha256=digest(s),source_xml_sha256=hashlib.sha256(raw.encode()).hexdigest(),model_dir=str(modeldir),existing_frames=None,previous_split=row['previous_split'],dataset_version=f'{slug}_dual224_fk_v1',RLDS_identity_required=False,relocations=relocations)
    available.append(e);row['status']='AVAILABLE'
   except Exception as ex:row.update(status='INVALID',reason=str(ex))
 existing=[e for e in available if e['previous_split'] in ('train','validation')];new=sorted((e for e in available if e not in existing),key=lambda e:hashlib.sha256(('20261023|'+e['source_demo']).encode()).hexdigest())
 for e in existing:e['split']=e['previous_split']
 nd=max(0,5-sum(e['split']=='validation' for e in existing));nt=max(0,20-sum(e['split']=='train' for e in existing));selected=existing.copy()
 for e in new[:nd]:e['split']='validation';selected.append(e)
 for e in new[nd:nd+nt]:e['split']='train';selected.append(e)
 assert sum(e['split']=='train' for e in selected)==20 and sum(e['split']=='validation' for e in selected)==5,'insufficient valid25; preserve evidence, do not duplicate'
 selection=dict(version=f'{slug}_dual224_fk_v1',task_id=slug,task=task,episodes=selected,protected_sources=[dict(source_hdf5=x,source_demo=y) for x,y in protected],background_texture_files=seen,selection_rule='preserve existing permitted train/validation identities; SHA256(seed|demo) for remaining identities; first missing validation then train; no model scoring',source_available=len(available),frozen_before_render=True)
 save(out/'dataset_selection.json',selection);save(out/'task_protocol.json',dict(task_id=slug,task=task,bddl=source.name.removesuffix('_demo.hdf5')+'.bddl',development_seeds=list(range(2026102400+i*100,2026102420+i*100)),unseen_palette_assignment=[j%3 for j in range(20)],initialization_noise=None,training_seed=20261023,max_steps=220,source=str(source),original_cases_not_read=True))
with (a.output/'trajectory_availability.csv').open('x') as f:
 keys=sorted(set().union(*(r.keys() for r in inventory)));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(inventory)
print(json.dumps(dict(selected=chosen,output=str(a.output)),indent=2))
