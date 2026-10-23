"""Official frozen AE extraction for one verified derived training episode."""
import argparse,importlib.util,json,hashlib,sys,subprocess,os
from pathlib import Path
import numpy as np,torch
p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();o=a.out;w=a.project/'worktrees/rla-official-0fc6e97'
def js(n,v):
 with (o/n).open('x') as f:json.dump(v,f,indent=2)
uuid=os.environ['RLA_GPU_UUID']
assert str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')==uuid.removeprefix('GPU-')
assert torch.cuda.mem_get_info()[0]>=26*1024**3
torch.cuda.set_per_process_memory_fraction(.50);torch.set_num_threads(4);torch.manual_seed(20261023)
before=subprocess.check_output(['nvidia-smi','-i',uuid,'-q','-d','ECC'],text=True);(o/'ae_ecc_before.txt').write_text(before)
sys.path.insert(0,str(w));path=w/'rebuttal/exp2_vla_adapter/sidecar/extract_rla_sidecar.py';spec=importlib.util.spec_from_file_location('official_extractor',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
dino=a.project/'downloads/dinov3-vitl16-pretrain-lvd1689m';mod.get_dinov3_model_for_channels=lambda c:str(dino)
run=a.project/'experiments/official_libero_rla_consistency/run_20261023_01/downloads/extracted/runs/16x64_libero/20260725_00-39-52'
ext=mod.RlaSidecarExtractor(str(run/'config.yaml'),str(run),step='90000',need_decoder=True,need_image_decoder=False,dino_compile=False)
assert 'dinov3' in ext.dino_extractor.config.model_type and ext.dino_channels==1024
labels=np.load(o/'trajectory_labels.npz');K=labels['K'].astype(np.float32);C=labels['c2w'].astype(np.float32);states=labels['proprio'];N=len(states);key=hashlib.sha1(np.ascontiguousarray(states,dtype=np.float32).tobytes()).hexdigest();mask=labels['valid_horizon_mask'];checks=[];recon=[]
with torch.inference_mode():
 for appearance in ['clean','background']:
  rgb=np.stack([np.load(o/'frames'/f'{t:04}.npz')[appearance] for t in range(N)])
  data=ext._reference_batch(rgb,K,C,np.array([0]),np.array([8]));out=ext.inference_batch(ext.models,data,training=False);repeat=ext.inference_batch(ext.models,data,training=False)
  assert torch.isfinite(out['enc_tokens']).all() and torch.equal(out['enc_tokens'],repeat['enc_tokens'])
  recon.append(dict(appearance=appearance,input_shape=list(data['rgbs'].shape),latent_shape=list(out['enc_tokens'].shape),decoder_shape=list(out['x1_pred'].shape),repeat_max=0.,latent_norm=float(out['enc_tokens'].float().norm()),latent_variance=float(out['enc_tokens'].float().var()),per_camera_L1=[float((out['x1_pred'][:,i]-out['x1_gt'][:,i]).abs().mean()) for i in range(2)],per_camera_MSE=[float((out['x1_pred'][:,i]-out['x1_gt'][:,i]).square().mean()) for i in range(2)]))
  del out,repeat,data
  tokens,hw=ext.episode_tokens(rgb,batch_frames=2);rays=ext.episode_plucker(K,C,hw,(224,224));z=ext.encode_episode(tokens,rays,chunk=8,batch_pairs=8,pairs_mode='anchored').float().cpu().numpy();assert z.shape==(N,8,16,64) and np.isfinite(z).all()
  root=o/('sidecar_'+appearance);(root/'libero_spatial').mkdir(parents=True);np.save(root/'libero_spatial/ep_000000.npy',z.astype(np.float16));np.save(root/'valid_horizon_mask.npy',mask)
  valid=z[:N-8].reshape(-1,16,64);mean=valid.mean(0);std=valid.std(0)
  for name,value in [('manifest.json',dict(chunk=8,num_tokens=16,token_dim=64,pairs='anchored',format='npy',suites=['libero_spatial'],appearance=appearance,step=90000,img_size=224,source='same-state derived HDF5 training trajectory')),('stats.json',dict(count=len(valid),mean=mean.tolist(),std=std.tolist(),std_floor=1e-3,global_rms=float(np.sqrt((valid**2).mean())),scope='complete training chunks only')),('libero_spatial/keys.json',{key:dict(file='ep_000000.npy',episode_index=0,traj_len=N)})]:
   (root/name).write_text(json.dumps(value,indent=2))
  # Frozen deterministic selection, no search based on results.
  for t,h in [(0,1),(0,8),(N//2,4)]:
   ref=ext.inference_batch(ext.models,ext._reference_batch(rgb,K,C,np.array([t]),np.array([t+h])),training=False)['enc_tokens'].float().cpu().numpy()[0]
   saved=z[t,h-1].astype(np.float16).astype(np.float32);err=float(np.max(abs(ref-saved)));tol=max(.001,float(abs(ref).max())*.02);assert err<tol,(err,tol)
   checks.append(dict(appearance=appearance,anchor=t,horizon=h,future=t+h,max_abs_error=err,tolerance=tol))
  del tokens,rays,z,rgb
after=subprocess.check_output(['nvidia-smi','-i',uuid,'-q','-d','ECC'],text=True);(o/'ae_ecc_after.txt').write_text(after)
# Compare numeric ECC fields only (timestamp differs).
def counters(s):return [x.strip() for x in s.splitlines() if ':' in x and any(k in x for k in ['Correctable','Uncorrectable','SRAM SM','Threshold Exceeded'])]
assert counters(before)==counters(after)
js('ae_forward_results.json',dict(status='PASS_NUMERICAL_INTERFACE_ONLY',reconstruction=recon,step=90000,not_paper_step40000=True,peak_GPU_allocated=torch.cuda.max_memory_allocated(),peak_GPU_reserved=torch.cuda.max_memory_reserved(),ECC_increment=False,optimizer_updates=0,backward=0))
js('sidecar_alignment_audit.json',dict(status='PASS_EXTRACTION',shape=[N,8,16,64],valid_full_chunks=N-8,derived_state_hash=key,identity='source episode+timestep+appearance+dual224_fk_v2',appearance_separate_roots=True,tail_clamped=True,tail_is_not_real_valid_target=True,validity_file='valid_horizon_mask.npy',recompute_checks=checks,source_frame_formula='min(t+h,N-1), h=1..8',camera_order=['agentview','robot0_eye_in_hand'],both_appearances_same_frozen_AE=True))
print('AE_EXTRACT_PASS',flush=True)
