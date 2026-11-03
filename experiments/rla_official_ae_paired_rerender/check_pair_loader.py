"""Read actual pair using pinned official loader; never construct a model."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,sys,json,shutil
import numpy as np
from omegaconf import OmegaConf
from prepare import RUN,OFFICIAL,dump
def main():
 p=argparse.ArgumentParser();p.add_argument('--suite',required=True);a=p.parse_args();src=RUN/'paired_rendered'/a.suite
 sys.path.insert(0,str(OFFICIAL));from src.datasets.trajectory_dataset import TrajectoryDataset
 cfg=OmegaConf.to_container(OmegaConf.load(OFFICIAL/'rebuttal/exp2_vla_adapter/configs/rla/16x64_libero.yaml'),resolve=True)
 datasets={};checks=[]
 for label in ['clean','background']:
  args=dict(cfg['dataset']['args']);args.update(root=str(src/'official_loader'/label),configs={a.suite:args['configs'][a.suite]},
   use_mp_for_indexing=False,trajectory_info_cache_file=str(src/f'{label}_loader_index.json'))
  datasets[label]=TrajectoryDataset(**args);assert len(datasets[label].trajectory_info)==1
 n=int(json.loads((src/'result.json').read_text())['frames'])
 for start,h in [(0,2),(n//2,10),(n-65,64)]:
  assert start+h<n, 'Official sampler reserves the final frame; do not create an unreachable handle'
  samples={}
  for label,ds in datasets.items():
   t=ds.trajectory_info[0];handle=dict(version=1,dataset_idx=t['dataset_idx'],group_name=a.suite,traj_id='000010',frame_id=start,
    sampled_horizon=h,feasible_horizon=h,camera_keys=['agentview_camera','wrist_camera'],rgb_variant='base',interaction_frame_indices=None,dataset_root=ds.datasets[t['dataset_idx']].root)
   samples[label]=ds.getitem_from_handle(handle,apply_augmentation=False)
  c,b=samples['clean'],samples['background']
  for key in ['intrinsics','w2c','qpos','target_qpos','frame_id']:
   assert np.array_equal(np.asarray(c[key]),np.asarray(b[key])),key
  assert list(c['rgbs'].shape)==[2,2,3,224,224]
  assert np.array_equal(np.asarray(c['frame_id']),[start,start+h-1])
  assert c['task_description']==b['task_description']
  # Official field w2c carries cam2world, as in the pinned converter; do not invert it.
  checks.append(dict(source_frames=[start,start+h-1],horizon=h,rgb_shape=list(c['rgbs'].shape),camera_order=handle['camera_keys'],metadata_equal=True,
   rgb_mean_abs_delta=float(np.abs(np.asarray(c['rgbs']).astype(float)-np.asarray(b['rgbs'])).mean()),returned_frame_id=np.asarray(c['frame_id']).tolist()))
 if (src/'loader_audit.json').exists() and not (src/'loader_audit_initial_boundary.json').exists():
  shutil.copy2(src/'loader_audit.json',src/'loader_audit_initial_boundary.json')
 dump(src/'loader_audit.json',dict(status='PASS',official_commit='0fc6e97212442ebccffa06f1d42f5481e5f22950',checks=checks,
  full_dataset_training_authorized=False,scope='one admitted pilot; not reduced-scope training',AE_forwards=0,DINO_forwards=0,policy_forwards=0))
 print('official pair loader PASS',a.suite,flush=True)
if __name__=='__main__':main()
