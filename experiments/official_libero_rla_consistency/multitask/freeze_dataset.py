"""Strict full-sequence pairing acceptance; train-only normalization."""
import argparse,json,csv,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);a=p.parse_args();selection=json.loads((a.task/'dataset_selection.json').read_text());progress={x['episode']:x for x in json.loads((a.task/'render_results.json').read_text())};accepted=[];failures=[];hashes=[];values={'action':[],'proprio':[]};seen=set()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for e in selection['episodes']:
 try:
  assert progress[e['episode']]['status']=='PASS','render_failed';root=Path(progress[e['episode']]['root']);lab=np.load(root/'trajectory_labels.npz');N=e['frames'];metrics=list(csv.DictReader((root/'regional_sequence_metrics.csv').open()));assert len(metrics)==N*2
  assert all(x['seg_equal']=='True' and x['depth_equal']=='True' for x in metrics),'geometry_mismatch'
  assert all(int(x['changed_pixels'])>0 for x in metrics),'background_invisible_camera_frame'
  assert sha(root/'camera_geometry_audit.json') and json.loads((root/'camera_geometry_audit.json').read_text())['status']=='PASS_RASTER_PROJECTION'
  assert hashlib.sha256(lab['actions'].astype(np.float32).tobytes()).hexdigest()==e['source_actions_sha256'];assert hashlib.sha256(lab['states'].tobytes()).hexdigest()==e['source_states_sha256'];assert lab['valid_horizon_mask'][:N-8].all()
  key=(e['source_hdf5'],e['source_demo']);assert key not in seen;seen.add(key)
  for t in range(N):
   path=root/'frames'/f'{t:04}.npz'
   with np.load(path) as d:
    assert d['clean'].shape==d['background'].shape==(2,3,224,224) and d['clean'].dtype==d['background'].dtype==np.uint8
    assert all(np.isfinite(d[k]).all() for k in d.files)
    for k,field in [('state','states'),('action','actions'),('proprio','proprio'),('K','K'),('c2w','c2w')]:np.testing.assert_array_equal(d[k],lab[field][t])
    assert d['K'].shape==(2,3,3) and np.all(d['K'][:,0,0]<0) and np.all(d['K'][:,1,1]>0)
   hashes.append(dict(episode=e['episode'],timestep=t,file=str(path),sha256=sha(path)))
  act=lab['actions'].astype(np.float32).copy();act[:,6]=1-np.clip(act[:,6],0,1)
  if e['split']=='train':values['action'].append(act);values['proprio'].append(lab['proprio'])
  accepted.append(dict(episode=e['episode'],split=e['split'],source_hdf5=e['source_hdf5'],source_demo=e['source_demo'],source_actions_sha256=e['source_actions_sha256'],source_states_sha256=e['source_states_sha256'],task=e['task'],frames=N,valid_chunks=N-8,root=str(root),dataset_version=selection['version'],storage_format='dual224_fk_v2',physical_episode_independent_unit=True,appearance_variants=2))
 except Exception as ex:failures.append(dict(episode=e['episode'],split=e['split'],reason=str(ex)))
audit=dict(accepted=len(accepted),train=sum(e['split']=='train' for e in accepted),development=sum(e['split']=='validation' for e in accepted),failures=failures,criterion='unchanged strict full sequence: every view/frame intervention nonzero, exact geometry/depth, same-state FK, source and camera projections verified',model_updates=0)
(a.task/'data_acceptance_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2),flush=True)
assert not failures,'Data failures retained: do not start models; no relaxed quality threshold'
stats={}
for field,arr in values.items():
 x=np.concatenate(arr);stats[field]={k:np.asarray(v).tolist() for k,v in dict(mean=x.mean(0),std=x.std(0),min=x.min(0),max=x.max(0),q01=np.quantile(x,.01,axis=0),q99=np.quantile(x,.99,axis=0)).items()}
stats['action']['mask']=[True]*6+[False]
train={(e['source_hdf5'],e['source_demo']) for e in accepted if e['split']=='train'};dev={(e['source_hdf5'],e['source_demo']) for e in accepted if e['split']=='validation'};protected={(e['source_hdf5'],e['source_demo']) for e in selection['protected_sources']};assert not train&dev and not (train|dev)&protected
def save(name,x):
 with (a.task/name).open('x') as f:json.dump(x,f,indent=2)
save('normalization_statistics_expanded_train_only.json',stats);save('frozen_dataset_manifest.json',dict(status='PASS',dataset_version=selection['version'],task_id=selection['task_id'],episodes=accepted,camera_order=['agentview','robot0_eye_in_hand'],reflection=1,MSAA=4,training_episodes=len(train),development_episodes=len(dev),total_frames=sum(e['frames'] for e in accepted),train_chunks=sum(e['valid_chunks'] for e in accepted if e['split']=='train'),development_chunks=sum(e['valid_chunks'] for e in accepted if e['split']=='validation'),selection_sha256=sha(a.task/'dataset_selection.json'),proprio='same-recorded-state FK with joint/finger agreement and robot-only FK check',geometry='exact pair segmentation/depth/state/body/cameras',visibility='nonzero RGB intervention each frame and both views',tail='h1..8 anchored targets; incomplete windows excluded from policy'))
save('frame_source_manifest.json',hashes);save('split_integrity.json',dict(train_development_overlap=0,protected_overlap=0,appearances_follow_episode=True,protected_contents_read=False))
