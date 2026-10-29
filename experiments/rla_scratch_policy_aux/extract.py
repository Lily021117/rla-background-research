"""Encode accepted frozen DINO/ray cache, strict new AE step2000, no AE update.

Only complete anchors are materialized: no padded/clamped terminal targets.
"""
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_scratch_policy_aux.common import *
import numpy as np
import torch
from rla_ae_from_scratch import run as ae
def main():
 before=ae.ecc();mods,cfg=ae.setup();inventory=read(ROOT/'encoder_inventory.json');es=read(ROOT/'data_manifest.json')['episodes'];rows=[];checks=[]
 save(ROOT/'extraction_configuration.json',dict(model_config=cfg['models'],input='accepted frozen DINO cache (agentview,wrist), current state Plucker',horizons=list(range(1,9)),micro_pairs=8,mode='eval + inference_mode',AE_updates=0,normalization=read(ROOT/'protocol.json')['normalization']))
 with torch.inference_mode():
  for arm in ['C','D']:
   ck=torch.load(inventory[arm]['path'],map_location='cpu',weights_only=False);assert ck['global_step']==2000 and ck['next_sample']==16000
   for k,m in mods.items():m.load_state_dict(ck['models'][k],strict=True);m.eval().requires_grad_(False)
   assert ae.digest(mods)==read(AE/arm/'completion.json')['final_parameter_sha256'];del ck
   total=np.zeros((16,64),np.float64);squares=total.copy();n=0
   for e in es:
    for appearance in ['clean','background']:
     folder=ROOT/'targets'/arm/appearance;folder.mkdir(parents=True,exist_ok=True)
     features=np.load(Path(e['feature_root'])/(appearance+'.npy'),mmap_mode='r');rays=np.load(Path(e['feature_root'])/'plucker.npy',mmap_mode='r')
     assert features.shape==(e['frames'],2,196,1024) and rays.shape==(e['frames'],392,6)
     N=e['valid_chunks'];assert N==e['frames']-8
     lab=np.load(Path(e['root'])/'trajectory_labels.npz');assert lab['valid_horizon_mask'][:N].shape==(N,8) and lab['valid_horizon_mask'][:N].all()
     z=np.empty((N,8,16,64),np.float16)
     for t in range(N):
      x=torch.from_numpy(np.array(features[t+1:t+9]-features[t],copy=True)).cuda().reshape(8,392,1024)
      r=torch.from_numpy(np.repeat(rays[t:t+1],8,axis=0).copy()).cuda()
      v=mods['encoder'](x,num_views=2,plucker=r)[0].float().cpu().numpy();assert v.shape==(8,16,64) and np.isfinite(v).all()
      z[t]=v
      if e['split']=='train':total+=v.astype(np.float64).sum(0);squares+=np.square(v.astype(np.float64)).sum(0);n+=8
     path=folder/(e['episode']+'.npy');assert not path.exists();np.save(path,z)
     for t,h in [(0,1),(N//2,4),(N-1,8)]:
      x,y,r=ae.inputs(e,t,h,appearance);ref=ae.encode(mods,x,y,r).float().cpu().numpy()[0];delta=float(abs(ref-z[t,h-1].astype(np.float32)).max());tol=max(.002,.002*float(abs(ref).max()));assert delta<=tol,(e['episode'],delta,tol)
      checks.append(dict(arm=arm,episode=e['episode'],appearance=appearance,t=t,h=h,future=t+h,max_error=delta,tolerance=tol,comparison='independent single-pair accepted cache vs stored batch8 FP16'))
     rows.append(dict(arm=arm,episode=e['episode'],split=e['split'],appearance=appearance,valid_chunks=N,shape=list(z.shape),path=str(path),sha256=sha(path),encoder_checkpoint_sha256=inventory[arm]['sha256'],feature_path=str(Path(e['feature_root'])/(appearance+'.npy')),mask_source=str(Path(e['root'])/'trajectory_labels.npz')))
     assert ae.ec(before)==ae.ec(ae.ecc());print('SIDECAR',arm,e['episode'],appearance,N,flush=True)
   mu=total/n;std=np.sqrt(np.maximum(squares/n-mu*mu,0));scale=float(np.sqrt((mu*mu+std*std).mean()-mu.mean()**2));assert np.isfinite(scale) and scale>0
   stats=dict(mean=mu.tolist(),std=std.tolist(),global_scale=scale,count=n,scope='only20 train episodes, both appearances equally, every valid physical anchor once x8 horizons; each position16x64',mean_shape=[16,64],std_floor=None,global_scale_floor=None,no_clipping=True)
   save(ROOT/'targets'/arm/'statistics.json',stats)
   # CPU normalization audit of every frame, and frozen metadata inventory.
   norms=[]
   for row in [r for r in rows if r['arm']==arm]:
    raw=np.load(row['path']);normalized=(raw.astype(np.float32)-mu.astype(np.float32))/np.float32(scale);assert normalized.shape==tuple(row['shape']) and np.isfinite(normalized).all()
    norms.append(dict(episode=row['episode'],split=row['split'],appearance=row['appearance'],raw_RMS=float(np.sqrt(np.square(raw.astype(np.float64)).mean())),normalized_RMS=float(np.sqrt(np.square(normalized.astype(np.float64)).mean()))))
   save(ROOT/'targets'/arm/'distribution.json',norms)
 save(ROOT/'target_manifest.json',rows);save(ROOT/'target_acceptance.json',dict(status='PASS',AE_updates=0,DINO_updates=0,checks=checks,episodes=25,appearances=2,AE_models=2,cumulative_horizon='t+h h1..8, complete windows only',camera_order=['agentview','robot0_eye_in_hand'],statistics_hash={a:sha(ROOT/'targets'/a/'statistics.json') for a in ['C','D']},independent_source='accepted historical DINO/ray cache, not new RGB extraction',ECC_unchanged=True))
if __name__=='__main__':main()
