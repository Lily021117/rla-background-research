"""Pinned original HDF5 -> explicit source XML and pre-action derived identities."""
import argparse,hashlib,json,importlib.util,xml.etree.ElementTree as ET
from pathlib import Path
import h5py,numpy as np,mujoco
from audit_source import sha,save

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--download-record',type=Path);p.add_argument('--project',type=Path,required=True);p.add_argument('--libero',type=Path,required=True);a=p.parse_args();o=a.out;o.mkdir(exist_ok=True)
 download=json.loads((a.download_record or o/'download_provenance.json').read_text());assert download['status']=='PASS';source=Path(download['path']);assert sha(source)==download['expected_sha256']
 assets=a.libero/'libero/libero/assets';robo=Path(importlib.util.find_spec('robosuite').origin).parent
 old=a.project/'experiments/official_libero_rla_consistency/run_20261023_10/stove/dataset_selection.json'
 seen=json.loads(old.read_text())['background_texture_files']
 gaps=json.loads((a.project/'experiments/rla_appearance_gap_eval/run_20261027_01/asset_manifest.json').read_text())
 bg={f'new_background_{i}':{} for i in range(1,5)}
 for x in gaps['assets']:
  assert sha(x['output'])==x['output_sha256'];bg[x['condition']][x['texture']]=x['output']
 assert all(set(x)=={'tex-table','tex-wall'} for x in bg.values())
 save(o/'background_protocol.json',dict(training=seen,development=bg,rule='reuse all four pre-existing generic texture combinations, no task model scoring; only tex-table/tex-wall files',assets={p:sha(p) for p in [*seen.values(),*(p for b in bg.values() for p in b.values())]},reflection=1,MSAA=4,visibility_rule='report each frame/view; require intervention nonzero somewhere in each episode and both camera streams; legitimate occlusion is not a different physical state'))
 save(o/'experiment_protocol.json',dict(status='FROZEN_BEFORE_TRAINING',task='libero_goal/open_the_middle_drawer_of_the_cabinet',methods=['BC-standard','BC-matched','C-frozen-RLA'],orders=[20261101,20261102],updates_each=600,total_updates=3600,lambda_aux=.22408536585365854,policy_lr=.0002,scheduler='MultiStepLR milestone600 gamma0.1 AFTER optimizer',microbatch=8,accumulation=2,appearance_batch=16,AE='original official LIBERO step90000, frozen',AE_updates=0,initialization='untrained original common policy origin, no stove trained weights',checkpoint_steps=[0,100,300,600],Clean_trials=120,background_trials_max=480,gate='one method must achieve >=12/20 Clean in BOTH orders; then all six policies x four backgrounds',development_seeds=list(range(2026111000,2026111020)),max_steps=300,control_freq=20,chunk=8,retry='infrastructure failure stops and retained; no success-selected task-failure reruns',stop='failed source alignment, nonfinite, CUDA/ECC or resource threshold; never change loss/config',sealed_reads=False))
 records=[];excluded=[];audit=[];modelroot=o/'source_models';modelroot.mkdir()
 with h5py.File(source,'r') as f:
  g=f['data'];ids=sorted(g,key=lambda x:int(x.rsplit('_',1)[1]));metadata={k:str(v) for k,v in g.attrs.items()};env=json.loads(g.attrs['env_args']);assert env['env_kwargs']['control_freq']==20 and env['env_kwargs']['controller_configs']['control_delta']
  assert 'open_the_middle_drawer_of_the_cabinet.bddl' in g.attrs['bddl_file_name']
  save(o/'source_structure.json',dict(data_attributes=metadata,episode_ids=ids,source_change='official pinned original file, NOT a Cosmos ID join',selection='ascending numeric ID; first20 complete then5 complete; exclusions only missing fields, invalid state/time/model; no policy result filtering',window_rule='full contiguous h1..8 within one episode; no invalid tail repeated as real',observation_semantics='scripts/create_dataset.py steps action[j], records obs, then saves states[valid_index] and actions[valid_index]; derived input uses state[t], not obs[t]'))
  for demo in ids:
   if len(records)==25:break
   d=g[demo];row=dict(demo=demo,status='PENDING')
   try:
    assert 'model_file' in d.attrs
    s=d['states'][:];ac=d['actions'][:];N=len(s);assert ac.shape==(N,7) and N>8 and np.isfinite(s).all() and np.isfinite(ac).all()
    assert all(d['obs/'+k].shape[0]==N for k in ['agentview_rgb','eye_in_hand_rgb','joint_states','gripper_states','ee_states'])
    breaks=np.flatnonzero(abs(np.diff(s[:,0])-.05)>1e-9);assert not len(breaks),'non20Hz discontinuities: '+str(breaks.tolist())
    raw=str(d.attrs['model_file']);tree=ET.fromstring(raw);relocations=[]
    for e in tree.iter():
     v=e.get('file')
     if not v:continue
     if '/chiliocosm/assets/' in v:q=assets/v.split('/chiliocosm/assets/',1)[1]
     elif '/libero/assets/' in v:q=assets/v.split('/libero/assets/',1)[1]
     elif 'robosuite' in Path(v).parts:
      parts=Path(v).parts;i=max(i for i,n in enumerate(parts) if n=='robosuite');q=robo.joinpath(*parts[i+1:])
     else:q=Path(v)
     q=q.resolve();assert q.is_file(),str(q);e.set('file',str(q));relocations.append(dict(original=v,local=str(q),sha256=sha(q)))
    folder=modelroot/demo;folder.mkdir();text=ET.tostring(tree,encoding='unicode');(folder/'clean.xml').write_text(text);back=ET.fromstring(text)
    for name,path in seen.items():nodes=[e for e in back.iter('texture') if e.get('name')==name];assert len(nodes)==1;nodes[0].set('file',path)
    (folder/'background.xml').write_text(ET.tostring(back,encoding='unicode'))
    m=mujoco.MjModel.from_xml_path(str(folder/'clean.xml'));assert m.na==0 and s.shape[1]==1+m.nq+m.nv
    arm=[m.joint('robot0_joint'+str(i)).qposadr[0] for i in range(1,8)];fg=[m.joint('gripper0_finger_joint'+str(i)).qposadr[0] for i in [1,2]]
    np.testing.assert_array_equal(s[1:,1:1+m.nq][:,arm],d['obs/joint_states'][:-1]);np.testing.assert_array_equal(s[1:,1:1+m.nq][:,fg],d['obs/gripper_states'][:-1])
    assert all(m.cam(c).id>=0 for c in ['agentview','robot0_eye_in_hand']) and m.vis.quality.offsamples==4
    assert m.joint('wooden_cabinet_1_middle_level').id>=0
    ep='official_goal_middle_'+demo+'_v1';records.append(dict(episode=ep,episode_id=ep,source_hdf5=str(source),source_demo=demo,task='open the middle drawer of the cabinet',frames=N,complete_h8_chunks=N-8,split='train' if len(records)<20 else 'validation',source_actions_sha256=hashlib.sha256(ac.astype(np.float32).tobytes()).hexdigest(),source_states_sha256=hashlib.sha256(s.tobytes()).hexdigest(),source_xml_sha256=hashlib.sha256(raw.encode()).hexdigest(),model_dir=str(folder),existing_frames=None,relocations=relocations))
    row.update(status='PASS_PRE_RENDER',frames=N,non20Hz_intervals=0,state_t_plus_1_vs_obs_t_joints_exact=True,fingers_exact=True)
   except Exception as ex:row.update(status='EXCLUDED',reason=str(ex));excluded.append(row)
   audit.append(row)
 save(o/'admission_audit.json',dict(rows=audit,excluded=excluded,complete_candidates=len(records),physics_steps=0,source_hdf5_sha256=sha(source),status='PASS_PRE_RENDER' if len(records)==25 else 'BLOCKED'))
 assert len(records)==25,'fewer than20+5 complete candidates'
 save(o/'dataset_selection.json',dict(version='official_goal_middle_dual224_fk_v1',task_id='libero_goal/open_the_middle_drawer_of_the_cabinet',episodes=records,protected_sources=[],background_texture_files=seen,source_download=download,source_global_metadata=metadata,source_change='do not identify official demo IDs with Cosmos IDs',source_code=str(a.libero/'scripts/create_dataset.py'),source_code_sha256=sha(a.libero/'scripts/create_dataset.py')))
 print('ADMISSION_PRE_RENDER_PASS',len(records),sum(e['frames'] for e in records),flush=True)
if __name__=='__main__':main()
