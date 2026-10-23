"""Admit nonsealed HDF5 demos by metadata and state/action invariants only."""
import argparse,csv,hashlib,json,os,xml.etree.ElementTree as ET,importlib.util
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import h5py,numpy as np,mujoco

def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2)
def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--libero',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    splitpath=a.project/'experiments/rla_background_invariance/step2_training/split_manifest.json';s=json.loads(splitpath.read_text())['splits'];task=s['train'][0]['task'];source=s['train'][0]['source_hdf5']
    sealpath=a.project/'artifacts/action_prediction_evidence/AE500_robustness_and_census/data_census_v2/sealed_evaluation_candidates.json';original_seal=json.loads(sealpath.read_text())['demonstrations']
    releasedpath=a.project/'artifacts/action_conditioned_wm/TZ_three_seed_500/independent_data/selection.json';released={(x['hdf5'],x['demo']) for x in json.loads(releasedpath.read_text())['demonstrations']}
    sealed={(x.get('raw_hdf5'),x.get('raw_demo')) for x in original_seal if (x.get('raw_hdf5'),x.get('raw_demo')) not in released};test={(x['source_hdf5'],x['source_demo']) for x in s['test']}
    old=json.loads((a.previous/'episode_manifest.json').read_text());known={(x['source_hdf5'],x['source_demo']):x for x in old}
    assets=a.libero/'libero/libero/assets';robosuite=Path(importlib.util.find_spec('robosuite').origin).parent
    reference=ET.parse(Path(old[0]['geometry_path']).parent/'background.xml').getroot();background_files={e.get('name'):e.get('file') for e in reference.iter('texture') if e.get('name') in ('tex-table','tex-wall')};assert len(background_files)==2
    inventory=[];admitted=[];seen={};pool=a.out/'source_models';pool.mkdir()
    with h5py.File(source,'r') as f:
        globalmeta={k:str(v) for k,v in f['data'].attrs.items()}
        for demo in sorted(f['data'],key=lambda n:int(n.rsplit('_',1)[-1])):
            d=f['data'][demo];pair=(source,demo);protected=pair in sealed or pair in test
            row=dict(source_demo=demo,source_hdf5=source,task=task,previous_split=known.get(pair,{}).get('split','unassigned'),status='PROTECTED_SEALED' if pair in sealed else 'PROTECTED_FROZEN_TEST' if pair in test else 'PENDING',reason='protected identity metadata only' if protected else '',length=int(d['actions'].shape[0]) if 'actions' in d else 0,has_states='states' in d,has_actions='actions' in d,has_agentview='obs/agentview_rgb' in d,has_wrist='obs/eye_in_hand_rgb' in d,has_XML='model_file' in d.attrs,content_read=False,action_sha256='',states_sha256='',last_reward='',last_done='',source_recorded_success='',prior_script_selected=pair in known)
            if protected:inventory.append(row);continue
            try:
                assert all(row[k] for k in ('has_states','has_actions','has_agentview','has_wrist','has_XML')),'missing_required_field'
                states=d['states'][:];actions=d['actions'][:].astype(np.float32);row['content_read']=True;N=len(states)
                assert states.ndim==2 and actions.shape==(N,7) and N>8,'invalid_shape_or_length'
                assert np.isfinite(states).all() and np.isfinite(actions).all(),'nonfinite'
                assert np.allclose(np.diff(states[:,0]),.05,rtol=0,atol=1e-9),'control_time_not_20Hz'
                assert all(d['obs/'+k].shape==(N,128,128,3) and d['obs/'+k].dtype==np.uint8 for k in ('agentview_rgb','eye_in_hand_rgb')),'camera_schema_mismatch'
                ah=digest(actions);sh=digest(states);row.update(action_sha256=ah,states_sha256=sh)
                assert ah not in seen,'duplicate_full_action_identity_of_'+seen.get(ah,'');seen[ah]=demo
                raw=str(d.attrs['model_file']);root=ET.fromstring(raw);relocations=[]
                for e in root.iter():
                    original=e.get('file')
                    if not original:continue
                    target=original
                    if '/chiliocosm/assets/' in target:target=str(assets/target.split('/chiliocosm/assets/',1)[1])
                    elif 'robosuite' in Path(target).parts:
                        parts=Path(target).parts;idx=max(i for i,v in enumerate(parts) if v=='robosuite');target=str(robosuite.joinpath(*parts[idx+1:]))
                    assert Path(target).is_file(),'missing_asset:'+target
                    if target!=original:e.set('file',target);relocations.append(dict(original=original,local=target))
                clean=ET.tostring(root,encoding='unicode');background=ET.fromstring(clean)
                for name,path in background_files.items():
                    node=next(e for e in background.iter('texture') if e.get('name')==name);node.set('file',path)
                modeldir=pool/demo;modeldir.mkdir();(modeldir/'clean.xml').write_text(clean);(modeldir/'background.xml').write_text(ET.tostring(background,encoding='unicode'))
                m=mujoco.MjModel.from_xml_path(str(modeldir/'clean.xml'));assert m.na==0 and states.shape[1]==1+m.nq+m.nv
                assert m.cam('agentview').id>=0 and m.cam('robot0_eye_in_hand').id>=0
                assert m.vis.quality.offsamples==4
                if 'rewards' in d:
                    reward=d['rewards'][:];row['last_reward']=float(reward[-1]);row['source_recorded_success']=bool(np.any(reward>0))
                if 'dones' in d:row['last_done']=bool(d['dones'][-1])
                identity=known[pair]['episode'] if pair in known else f"derived_cookie_{demo}_v4"
                entry=dict(episode=identity,episode_id=identity,source_hdf5=source,source_demo=demo,task=task,frames=N,complete_h8_chunks=N-8,source_actions_sha256=ah,source_states_sha256=sh,source_xml_sha256=hashlib.sha256(raw.encode()).hexdigest(),model_dir=str(modeldir),existing_frames=str(a.previous/'episodes'/identity) if pair in known else None,previous_split=row['previous_split'],dataset_version='cookie_dual224_fk_v4',RLDS_identity_required=False,relocations=relocations)
                row.update(status='AVAILABLE',reason='complete raw state/action/two-camera/XML; no RLDS join required');admitted.append(entry)
            except Exception as ex:row.update(status='INVALID',reason=str(ex))
            inventory.append(row)
    selected=[e for e in admitted if e['existing_frames']];new=[e for e in admitted if not e['existing_frames']];new.sort(key=lambda e:hashlib.sha256(('20261023|'+e['source_demo']).encode()).hexdigest())
    for e in selected:e['split']=e['previous_split']
    need_dev=max(0,5-sum(e['split']=='validation' for e in selected));need_train=max(0,20-sum(e['split']=='train' for e in selected))
    for e in new[:need_dev]:e['split']='validation';selected.append(e)
    for e in new[need_dev:need_dev+need_train]:e['split']='train';selected.append(e)
    selected.sort(key=lambda e:(e['split'],int(e['source_demo'].rsplit('_',1)[-1])))
    chosen={(e['source_hdf5'],e['source_demo']):e['split'] for e in selected}
    for row in inventory:
        row['new_split']=chosen.get((row['source_hdf5'],row['source_demo']),'not_selected')
        if row['status']=='AVAILABLE' and not row['prior_script_selected']:row['prior_nonuse_reason']='old script iterated only Step2 fixed split/cache; no demonstrated source-data deficiency'
        else:row['prior_nonuse_reason']=''
    with (a.out/'trajectory_availability.csv').open('x') as f:w=csv.DictWriter(f,fieldnames=list(inventory[0]));w.writeheader();w.writerows(inventory)
    manifest=dict(version='cookie_dual224_fk_v4',selection_rule='preserve old4 train+1 dev; eligible new IDs ascending SHA256(20261023|source_demo), first4 dev then16 train; no outcome/model scoring used',episodes=selected,protected_sources=[dict(source_hdf5=x,source_demo=y) for x,y in sealed|test if x is not None],source_metadata_hashes={str(v):sha(v) for v in (splitpath,sealpath,releasedpath)},background_texture_files=background_files,source_global_metadata=globalmeta)
    save(a.out/'dataset_selection.json',manifest)
    summary=dict(source_demos=len(inventory),protected_sealed=sum(x['status']=='PROTECTED_SEALED' for x in inventory),protected_frozen_test=sum(x['status']=='PROTECTED_FROZEN_TEST' for x in inventory),available=len(admitted),invalid=sum(x['status']=='INVALID' for x in inventory),old_train=4,old_development=1,new_train=sum(x['split']=='train' for x in selected),new_development=sum(x['split']=='validation' for x in selected),new_independent_episodes=sum(not x['existing_frames'] for x in selected),unused_old_manifest_only=sum(x['status']=='AVAILABLE' and not x['prior_script_selected'] for x in inventory),optimizer_updates=0,backward=0,new_closed_loop=0,protected_episode_contents_read=False)
    save(a.out/'availability_summary.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
