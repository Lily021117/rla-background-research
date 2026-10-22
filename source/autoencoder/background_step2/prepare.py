from .common import *
def main():
 assert shell(['git','-C',str(WT),'rev-parse','HEAD']).strip()=='0fc6e97212442ebccffa06f1d42f5481e5f22950'
 protected={}
 for base,inventory in ((STEP1,read(STEP1/'final_delivery_sha256.json')['files']),(READY,read(READY/'artifact_sha256.json'))):
  for n,h in inventory.items():assert sha(base/n)==h;protected[str(base/n)]=h
 for f in read(STEP1/'verified_downloads.json')['files']:assert sha(f['path'])==f['sha256'];protected[f['path']]=f['sha256']
 assert sha(DINO/'model.safetensors')=='dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179'
 for n in shell(['git','-C',str(ROOT),'ls-files','-z']).split('\0'):
  if n and (ROOT/n).is_file():protected[str(ROOT/n)]=sha(ROOT/n)
 protected[str(ROOT/'.git/config')]=sha(ROOT/'.git/config')
 write(OUT/'protection_before.json',protected)
 write(OUT/'git_before.json',dict(branch=shell(['git','-C',str(ROOT),'branch','--show-current']),head=shell(['git','-C',str(ROOT),'rev-parse','HEAD']),status=shell(['git','-C',str(ROOT),'status','--short']),worktrees=shell(['git','-C',str(ROOT),'worktree','list'])))
 write(OUT/'gpu_ecc_baseline.json',ecc())
 write(OUT/'resource_before.json',dict(gpus=shell(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu','--format=csv']),processes=shell(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv']),disk=shell(['df','-h','/','${DATA_ROOT}']),mount=shell(['findmnt','-T','${DATA_ROOT}']),memory=shell(['free','-h'])))
 manifest=read(READY/'episode_split_manifest.json')
 assert [manifest['counts'][s]['episodes'] for s in ('train','validation','test')]==[19,4,20]
 write(OUT/'split_manifest.json',manifest)
 write(OUT/'protocol_before_execution.json',dict(seed=20261020,updates_each=200,physical_batch=4,num_workers=0,
  camera_handling='unchanged single-view 32x64 AE; two cameras independently sampled, same order for B/C',
  effective_view_pairs_per_update=4,branch_samples_per_update=8,
  precision='original inflat_all, initial log_scale=8 for conservative finite-gradient smoke, growth .001',
  reconstruction='SUM of original clean/background L1+MSE, mean over batch/camera units',
  previous_readiness_change='Readiness proposed branch mean; latest user formula uses branch sum, applied identically to B/C',
  optimizer=dict(name='AdamW',lr=1e-4,weight_decay=0),consistency='mean squared difference over complete 32x64 latent; no projection head',
  lambda_rule='train-only probe: min(0.1,0.1*norm(grad_reconstruction_on_encoder_out_layer)/norm(grad_consistency_on_same_parameter)); freeze before any update; no test or validation search',
  smoke_accounting='B/C first successful update is retained as step1 of 200; restore check resumes from this step, not an extra discarded test update',
  reserved_test='20 entire Step1 episodes excluded; no test inference or tuning this run',
  authorization='latest user explicitly requests measured batch1 smoke at 27.9GiB free on UUID1; no other GPU allowed',
  fail_policy='stop on OOM, nonfinite values, unexpected missing gradient, ECC increment; no automated device switch',
  initialization='official step100000 model weights; fresh AdamW/step0, not optimizer resume',
  checkpoints=[0,1,50,100,200],code_commit=shell(['git','-C',str(WT),'rev-parse','HEAD']).strip()))
 print('PROTECTION_AND_PROTOCOL_FROZEN',flush=True)
if __name__=='__main__':main()
