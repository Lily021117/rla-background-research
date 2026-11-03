"""Pinned official loader and paired interface; no encoder/DINO construction."""
from common import *
import sys, argparse
from omegaconf import OmegaConf

def index():
 for appearance in ['clean','background']:
  for suite in SUITES:
   dest=RUN/'official_loader'/appearance/suite
   if not dest.exists():continue
   original=json.loads((ROOT/'data/libero_converted'/suite/'index.json').read_text())
   trajectories={}
   for t in sorted(dest.glob('traj_*')):
    tid=t.name[5:];result=RUN/'trajectories'/suite/t.name/'result.json'
    if not result.exists():continue
    entry=original['trajectories'][tid].copy();entry['path']=str(t);trajectories[tid]=entry
   dump(dest/'index.json',dict(trajectories=trajectories,trajectory_list=list(trajectories)))

class PairedInterface:
 """Same handle for all arms. Base=(Clean,Clean); Aug/Cons=(Clean,Background).

 Actual elapsed time is explicit; no uniform-duration claim for filtered frames.
 Training is deliberately NOT implemented or launched in this data-only task.
 """
 def __init__(self,clean,background):self.clean=clean;self.background=background
 def read(self,handle,mode):
  assert mode in ['Base','Aug','Aug-Cons']
  a=self.clean.getitem_from_handle(handle,apply_augmentation=False)
  other=self.clean if mode=='Base' else self.background
  paired_handle=dict(handle,dataset_root=other.datasets[handle['dataset_idx']].root)
  b=other.getitem_from_handle(paired_handle,apply_augmentation=False)
  src=RUN/'trajectories'/handle['group_name']/f"traj_{handle['traj_id']}"/'camera_time.json';meta=json.loads(src.read_text());ids=np.array(a['frame_id'],dtype=int)
  info=dict(source_indices=[meta['source_indices'][i] for i in ids],source_times_s=[meta['source_times'][i] for i in ids],horizon_count=handle['sampled_horizon'])
  info['elapsed_time_s']=info['source_times_s'][-1]-info['source_times_s'][0]
  return a,b,info

def check(pilots):
 index();sys.path.insert(0,str(OFFICIAL));from src.datasets.trajectory_dataset import TrajectoryDataset
 cfg=OmegaConf.to_container(OmegaConf.load(OFFICIAL/'rebuttal/exp2_vla_adapter/configs/rla/16x64_libero.yaml'),resolve=True);checks=[]
 for suite in SUITES:
  for split,key in [('train','dataset'),('val','val_dataset')]:
   rows=[x for x in csv.DictReader(open(RUN/'source_manifest.csv')) if x['suite']==suite and x['split']==split and (RUN/'trajectories'/suite/f"traj_{x['trajectory_id']}"/'result.json').exists() and (not pilots or x['trajectory_id']=='000010')]
   if not rows:continue
   datasets={}
   for app in ['clean','background']:
    args=dict(cfg[key]['args']);args.update(root=str(RUN/'official_loader'/app),configs={suite:args['configs'][suite]},use_mp_for_indexing=False,trajectory_info_cache_file=str(RUN/f'{suite}_{split}_{app}_loader_index.json'))
    # Cache content must reflect current admitted render set, not a previous partial snapshot.
    cache=Path(args['trajectory_info_cache_file'])
    if cache.exists():args['trajectory_info_cache_file']=str(RUN/f'{suite}_{split}_{app}_loader_index_{len(rows)}.json')
    datasets[app]=TrajectoryDataset(**args)
   pair=PairedInterface(datasets['clean'],datasets['background'])
   for row in rows:
    tid=row['trajectory_id'];n=int(row['frames']);t=next(t for t in datasets['clean'].trajectory_info if str(t['traj_id']).zfill(6)==tid)
    entry=[]
    long_h=min(64,n-1)
    for start,h in [(0,2 if split=='train' else 10),(n//2,min(10,n-n//2-1)),(n-long_h-1,long_h)]:
     assert start>=0 and start+h<n
     handle=dict(version=1,dataset_idx=t['dataset_idx'],group_name=suite,traj_id=tid,frame_id=start,sampled_horizon=h,feasible_horizon=h,camera_keys=STREAMS,rgb_variant='base',interaction_frame_indices=None,dataset_root=datasets['clean'].datasets[t['dataset_idx']].root)
     a,b,info=pair.read(handle,'Aug')
     assert list(a['rgbs'].shape)==[2,2,3,224,224]
     for k in ['intrinsics','w2c','qpos','target_qpos','frame_id','root_poses']:assert np.array_equal(np.asarray(a[k]),np.asarray(b[k])),k
     assert a['task_description']==b['task_description'] and np.array_equal(a['frame_id'],[start,start+h-1])
     c,d,_=pair.read(handle,'Base');assert np.array_equal(c['rgbs'],d['rgbs'])
     entry.append(dict(handle=handle,physical_time=info,paired_metadata_equal=True,base_duplicate_exact=True,aug_rgb_mae=float(np.abs(np.asarray(a['rgbs']).astype(float)-np.asarray(b['rgbs'])).mean())))
    report=dict(suite=suite,trajectory_id=tid,status='PASS',official_commit='0fc6e97212442ebccffa06f1d42f5481e5f22950',checks=entry,model_forwards=0,training_enabled=False)
    dump(RUN/'trajectories'/suite/f'traj_{tid}'/'loader_audit.json',report);checks.append(report);print('loader',suite,tid,'PASS',flush=True)
 dump(RUN/'loader_audit.json',dict(trajectories=len(checks),all_pass=True,rows=checks,training_not_authorized=True,coverage_not_full=True,learned_model_forwards=0))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--pilots',action='store_true');a=p.parse_args();check(a.pilots)
