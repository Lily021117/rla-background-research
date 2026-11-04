"""CPU-only frozen subset/official sampling preparation. No model forward."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import json,csv,hashlib,sys,copy,subprocess
import numpy as np
P=Path('/data/lyc/projects/rla-wm')
R=P/'experiments/rla_official_subset_ae_startup/run_20261104_01'
S=P/'experiments/rla_official_demo_source_completion/run_20261103_01'
O=P/'worktrees/rla-official-0fc6e97'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def table(p,rows):
 with p.open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 from omegaconf import OmegaConf
 sys.path.insert(0,str(O))
 from src.datasets.trajectory_dataset import TrajectoryDataset
 cfg=OmegaConf.to_container(OmegaConf.load(O/'rebuttal/exp2_vla_adapter/configs/rla/16x64_libero.yaml'),resolve=True)
 original=list(csv.DictReader(open(S/'delivery/frozen_data_manifest.csv')))
 rows=[x for x in original if x['source_status']=='MULTI_EVIDENCE_JOIN' and x['paired_complete']=='True' and x['loader_status']=='PASS' and x['split_conflict']=='False']
 assert len(rows)==1323
 totals={s:dict(trajectories=sum(x['split']==s for x in rows),tasks=len({x['task'] for x in rows if x['split']==s}),frames=sum(int(x['frames']) for x in rows if x['split']==s)) for s in ['train','val']}
 assert (totals['train']['trajectories'],totals['val']['trajectories'],totals['train']['tasks'],totals['val']['tasks'])==(1296,27,38,19)
 table(R/'subset_manifest.csv',rows)
 save(R/'data_identity.json',dict(name='官方LIBERO示范统一重渲染子集',counts=totals,parent_manifest_sha256=sha(S/'delivery/frozen_data_manifest.csv'),parent_commit='382766d2078edb3ae421483bfb754af93200fc01',excluded_unresolved=287,excluded_missing=83,full_author_data_reproduction=False,no_download_or_rerender=True))
 timing={}
 for x in rows:
  p=S/'trajectories'/x['suite']/f"traj_{x['trajectory_id']}"/'camera_time.json'
  timing[x['suite'],x['trajectory_id']]=json.loads(p.read_text())
 plan=dict(status='FROZEN_CPU_PLAN_GPU_STARTUP_PENDING',init_seed=20261104,model_train_seed=20261105,sampler_seed=20261106,calibration_sampler_seed=20261107,validation_sampler_seed=20261108,augmentation_rng='none: fixed per-trajectory H+S; never consumes sampler/model RNG',official_commit='0fc6e97212442ebccffa06f1d42f5481e5f22950',effective_physical_batch=128,target_microbatch_physical=1,accumulation_steps=128,microbatch_status='pending actual full-effective-batch GPU test, identical for all arms',calibration_updates=100,calibration_nodes=[0,10,30,60,100],startup_updates_each=200,max_total_updates=700,checkpoints=[0,50,100,200],formal_training_authorized=False,formal_init='reload same newly constructed untrained step0; discard calibration and startup endpoints',optimizer=dict(name='AdamW',lr=1e-4,weight_decay=0,betas=[.9,.999],eps=1e-8),scheduler=None,precision='official mixed FP16 torso / FP32 masters; fixed loss scale256; fail on nonfinite, no silent scaler/precision changes',backend='Math SDPA, preflight required',load_dir=None,resume=None,auto_resume=False,AE_pretrained_loading=False,DINO=dict(model='facebook/dinov3-vitl16-pretrain-lvd1689m',freeze=True,eval=True,patch_size=16,channels=1024,patches_per_view=196,silent_dinov2_fallback=False,source_identity_status='local weight path/revision/hash must be verified before GPU execution'),image_decoder='not constructed; not feature decoder',loss=dict(reconstruction='0.5*(mean L1+mean MSE branch0 + mean L1+mean MSE branch1)',consistency='mean MSE(z_clean,z_bg) over effective B,16,64',feature_reduction='mean over effective B,2 cameras,196 patches,1024 channels'),calibration_rule='100 rec-only paired updates on train, separate sampler. Same fixed 128 train pairs at10/30/60/100. lambda=min(median(0.1*g_rec/g_cons), min(0.1*L_rec/L_cons)) over those four nodes. Require all poststartup unscaled encoder norms finite and >1e-12; otherwise STOP, no alternate calibration/search. step0 measured/excluded due zero output layer. Freeze before all three 200-update runs; AE-cons applies from update1.',latent_normalization='At each evaluation node compute per-position mean and global RMS scale on same fixed 128 training pairs, pooled both appearances; no validation fitting; report raw/unit/standardized distances, raw norms variance rank; near-zero scale marked invalid, not hidden.',shuffle_z='fixed derangement among validation samples in same task when possible; no self pairing; record task/horizon differences; unavailable groups explicitly missing',exposures=dict(each_arm_physical=25600,each_arm_reconstruction_branches=51200,calibration_physical=12800),stops=['successful update budgets','nonfinite loss/gradient/parameters','CUDA or new ECC anomaly','OOM','data/hash/paired-index mismatch'],policy_updates=0,rollouts=0)
 save(R/'startup_plan.json',plan)
 for name,branches,cons in [('AE-clean',['clean','clean'],False),('AE-aug',['clean','background'],False),('AE-cons',['clean','background'],True)]:
  a=copy.deepcopy(plan);a.update(arm=name,branches=branches,consistency_enabled=cons,lambda_cons='pending_train_calibration' if cons else 0,models={k:cfg['models'][k] for k in ['encoder','decoder']})
  save(R/'configs'/f'{name}.json',a)
 datasets={}
 for split,key in [('train','dataset'),('val','val_dataset')]:
  args=copy.deepcopy(cfg[key]['args']);args.update(root=str(S/'official_loader/clean'),use_mp_for_indexing=False,trajectory_info_cache_file=str(R/f'{split}_official_index.json'))
  ds=TrajectoryDataset(**args);datasets[split]=ds
  actual={(v['group_name'],str(v['traj_id']).zfill(6)) for v in ds.trajectory_info};expected={(x['suite'],x['trajectory_id']) for x in rows if x['split']==split};assert actual==expected
 def sample(ds,rg):
  state=np.random.get_state();np.random.set_state(rg.get_state())
  try:f=ds.sample_frame_index(-1);rg.set_state(np.random.get_state())
  finally:np.random.set_state(state)
  info=next(v for v in ds.trajectory_info if v['dataset_idx']==f['dataset_idx'] and v['traj_id']==f['traj_id'])
  suite=info['group_name'];tid=str(f['traj_id']).zfill(6);t=int(f['frame_id']);h=int(f['sampled_horizon']);j=t+h-1;meta=timing[suite,tid]
  assert j<len(meta['source_times']) and t+h<len(meta['source_times'])
  return dict(suite=suite,trajectory_id=tid,dataset_idx=int(f['dataset_idx']),frame_id=t,sampled_horizon=h,feasible_horizon=int(f['feasible_horizon']),future_frame=j,source_index=meta['source_indices'][t],future_source_index=meta['source_indices'][j],time_s=meta['source_times'][t],future_time_s=meta['source_times'][j],elapsed_time_s=meta['source_times'][j]-meta['source_times'][t])
 for name,n,seed in [('startup',25600,20261106),('calibration',12800,20261107),('fixed_train',128,20261109)]:
  rg=np.random.RandomState(seed);samples=[dict(occurrence=i,**sample(datasets['train'],rg)) for i in range(n)];table(R/f'{name}_samples.csv',samples)
 rg=np.random.RandomState(20261108);samples=[];seen={}
 while len(samples)<27*3:
  x=sample(datasets['val'],rg);key=(x['suite'],x['trajectory_id'])
  if seen.get(key,0)>=3:continue
  seen[key]=seen.get(key,0)+1;samples.append(dict(occurrence=len(samples),**x))
 table(R/'validation_samples.csv',samples)
 save(R/'sampling_audit.json',dict(status='PASS_METADATA_ONLY',official_random_sampler_called=True,global_numpy_rng_restored_each_call=True,train_horizon=[2,64],val_horizon=[10,64],future_index='t+h-1',physical_elapsed_time_recorded=True,train_samples=25600,calibration_samples=12800,fixed_train=128,fixed_validation=81,hashes={p.name:sha(p) for p in R.glob('*samples.csv')},model_forwards=0,updates=0))
 print('CPU subset and sampling preparation PASS',totals,flush=True)

if __name__=='__main__':main()
