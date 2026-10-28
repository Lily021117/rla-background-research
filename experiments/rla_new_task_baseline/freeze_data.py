"""Accept all selected episodes; train-only statistics and new-task balanced indices."""
import argparse,csv,json,hashlib,collections
from pathlib import Path
import numpy as np
from audit_source import save,sha

def balanced(lengths,seed):
 rng=np.random.RandomState(seed);episodes=list(lengths);pending={e:[] for e in episodes};order=[]
 for cycle in range(240):
  for j in rng.permutation(len(episodes)):
   e=episodes[j]
   if not pending[e]:pending[e]=list(map(int,rng.permutation(lengths[e])))
   order.append([e,pending[e].pop()])
 assert len(order)==4800 and set(collections.Counter(e for e,t in order).values())=={240}
 return order

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);a=p.parse_args();o=a.dataset;selection=json.loads((o/'dataset_selection.json').read_text());results=json.loads((o/'render_results.json').read_text());assert all(r['status']=='PASS' for r in results)
 byid={r['episode']:r for r in results};entries=[];vals={'action':[],'proprio':[]};frames=[];visibility=[]
 for e in selection['episodes']:
  root=Path(byid[e['episode']]['root']);lab=np.load(root/'trajectory_labels.npz');N=e['frames'];assert N==len(lab['states'])
  assert hashlib.sha256(lab['states'].tobytes()).hexdigest()==e['source_states_sha256'];assert hashlib.sha256(lab['actions'].astype(np.float32).tobytes()).hexdigest()==e['source_actions_sha256']
  assert np.allclose(np.diff(lab['states'][:,0]),.05,atol=1e-9,rtol=0) and lab['valid_horizon_mask'][:N-8].all()
  metrics=list(csv.DictReader((root/'regional_sequence_metrics.csv').open()));assert len(metrics)==N*2 and all(r['seg_equal']=='True' and r['depth_equal']=='True' for r in metrics)
  for cam in ['agentview','robot0_eye_in_hand']:
   rs=[r for r in metrics if r['camera']==cam];visible=sum(int(r['changed_pixels'])>0 for r in rs);assert visible>0
   visibility.append(dict(episode=e['episode'],camera=cam,frames=N,background_changed_frames=visible,zero_changed_frames=N-visible))
  assert json.loads((root/'camera_geometry_audit.json').read_text())['status']=='PASS_RASTER_PROJECTION'
  for t in range(N):
   f=root/'frames'/f'{t:04}.npz'
   with np.load(f) as d:
    assert d['clean'].shape==d['background'].shape==(2,3,224,224) and d['clean'].dtype==d['background'].dtype==np.uint8
    assert all(np.isfinite(d[k]).all() for k in d.files)
    for k,field in [('state','states'),('action','actions'),('proprio','proprio'),('K','K'),('c2w','c2w')]:np.testing.assert_array_equal(d[k],lab[field][t])
   frames.append(dict(episode=e['episode'],timestep=t,sha256=sha(f)))
  if e['split']=='train':
   act=lab['actions'].astype(np.float32).copy();act[:,6]=1-np.clip(act[:,6],0,1);vals['action'].append(act);vals['proprio'].append(lab['proprio'])
  entries.append(dict(episode=e['episode'],split=e['split'],source_hdf5=e['source_hdf5'],source_demo=e['source_demo'],source_actions_sha256=e['source_actions_sha256'],source_states_sha256=e['source_states_sha256'],task=e['task'],frames=N,valid_chunks=N-8,root=str(root),dataset_version=selection['version']))
 train=[e for e in entries if e['split']=='train'];dev=[e for e in entries if e['split']=='validation'];assert len(train)==20 and len(dev)==5 and not {e['source_demo'] for e in train}&{e['source_demo'] for e in dev}
 stats={}
 for field,vs in vals.items():
  x=np.concatenate(vs);stats[field]={k:np.asarray(v).tolist() for k,v in dict(mean=x.mean(0),std=x.std(0),min=x.min(0),max=x.max(0),q01=np.quantile(x,.01,axis=0),q99=np.quantile(x,.99,axis=0)).items()}
 stats['action']['mask']=[True]*6+[False]
 save(o/'normalization_statistics_expanded_train_only.json',stats)
 save(o/'frozen_dataset_manifest.json',dict(status='PASS',task_id=selection['task_id'],dataset_version=selection['version'],episodes=entries,training_episodes=20,development_episodes=5,total_frames=sum(e['frames'] for e in entries),train_chunks=sum(e['valid_chunks'] for e in train),development_chunks=sum(e['valid_chunks'] for e in dev),camera_order=['agentview','robot0_eye_in_hand'],reflection=1,MSAA=4,tail='exclude incomplete anchors; no repeated final frames counted as real',source='pinned official HDF5; no Cosmos mapping',derived=True))
 save(o/'frame_source_manifest.json',frames);save(o/'background_visibility.json',visibility)
 orders={}
 for seed in [20261101,20261102]:
  order=balanced({e['episode']:e['valid_chunks'] for e in train},seed);path=o/f'sampling_order_{seed}.json';save(path,order);orders[str(seed)]=dict(path=str(path),sha256=sha(path),physical_exposures=4800,appearance_exposures=9600,per_episode_exposures=240)
 save(o/'sampling_audit.json',dict(orders=orders,sampler='240 shuffled episode cycles; per-episode valid windows shuffled and exhausted before refill; independent NumPy RandomState',data_weights='uniform episode, not uniform window; no task-phase upweighting',development_used=False))
 save(o/'data_acceptance_audit.json',dict(status='PASS',train=20,development=5,frames=sum(e['frames'] for e in entries),train_windows=sum(e['valid_chunks'] for e in train),development_windows=sum(e['valid_chunks'] for e in dev),appearance_variants=2,source_hash_verified=True,exact_paired_state_camera_depth_segmentation=True,FK_robot_only_checked=True,raw_observation_phase='post action, not used directly',physics_steps=0,policy_calls=0))
 print('FULL_DATA_ACCEPTANCE_PASS',flush=True)
if __name__=='__main__':main()
