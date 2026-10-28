"""Original official step90000 only; both appearances, no encoder updates."""
import argparse,json,hashlib,sys,os
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'official_libero_rla_consistency/ae_consistency'))
from common import load,ecc,counters
from audit_source import save,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 # Resource/identity checks precede model construction; failure leaves explicit logs.
 assert torch.cuda.mem_get_info()[0]>=26*1024**3
 before=ecc();ext,init=load(a.project);a.out.mkdir(exist_ok=False)
 expected={'encoder':'a22efe9311f1588675f19c83f5a2af3afb1dd221e59c8170fb64ced60037086c','decoder':'916952b1da241492db095d959ce850eed2f77693b380e9f8b9756b1c7612ff1d'}
 for k,h in expected.items():assert sha(init/'ckpts'/f'{k}_step0090000.pt')==h;ext.models[k].eval().requires_grad_(False)
 ext.dino_extractor.eval().requires_grad_(False)
 manifest=json.loads((a.dataset/'frozen_dataset_manifest.json').read_text());assert manifest['status']=='PASS'
 total=np.zeros((16,64),np.float64);total2=total.copy();count=0;keys={c:{} for c in ['clean','background']};checks=[];records=[];recons=[]
 for c in keys:(a.out/c/'libero_goal').mkdir(parents=True)
 with torch.inference_mode():
  for index,e in enumerate(manifest['episodes']):
   root=Path(e['root']);lab=np.load(root/'trajectory_labels.npz');K=lab['K'].astype(np.float32);C=lab['c2w'].astype(np.float32);N=e['frames'];valid=e['valid_chunks'];assert valid==N-8
   key=hashlib.sha1(np.ascontiguousarray(lab['proprio'],dtype=np.float32).tobytes()).hexdigest()
   for c in keys:
    rgb=np.stack([np.load(root/'frames'/f'{t:04}.npz')[c] for t in range(N)])
    tokens,hw=ext.episode_tokens(rgb,batch_frames=4);rays=ext.episode_plucker(K,C,hw,(224,224));z=ext.encode_episode(tokens,rays,chunk=8,batch_pairs=8,pairs_mode='anchored').float().cpu().numpy()
    assert z.shape==(N,8,16,64) and np.isfinite(z).all();path=a.out/c/'libero_goal'/f'ep_{index:06}.npy';np.save(path,z.astype(np.float16));mask=lab['valid_horizon_mask'];np.save(path.with_name(f'mask_{index:06}.npy'),mask)
    assert key not in keys[c];keys[c][key]=dict(file=path.name,episode_index=index,traj_len=N)
    if e['split']=='train':
     v=z[:valid].reshape(-1,16,64).astype(np.float64);total+=v.sum(0);total2+=(v*v).sum(0);count+=len(v)
    for t,h in [(0,1),(0,8),(valid//2,4)]:
     out=ext.inference_batch(ext.models,ext._reference_batch(rgb,K,C,np.array([t]),np.array([t+h])),training=False);ref=out['enc_tokens'].float().cpu().numpy()[0];err=float(abs(ref-z[t,h-1].astype(np.float16).astype(np.float32)).max());tol=max(.001,.02*float(abs(ref).max()));assert err<tol
     checks.append(dict(episode=e['episode'],appearance=c,anchor=t,horizon=h,future=t+h,max_error=err,tolerance=tol))
     if t==0 and h==8:
      repeat=ext.inference_batch(ext.models,ext._reference_batch(rgb,K,C,np.array([t]),np.array([t+h])),training=False);assert torch.equal(out['enc_tokens'],repeat['enc_tokens'])
      recons.append(dict(episode=e['episode'],appearance=c,repeat_exact=True,per_camera_L1=[float((out['x1_pred'][:,i]-out['x1_gt'][:,i]).abs().mean()) for i in [0,1]],latent_norm=float(out['enc_tokens'].float().norm())))
    records.append(dict(episode=e['episode'],source_demo=e['source_demo'],split=e['split'],appearance=c,file=str(path),sha256=sha(path),proprio_key=key,frames=N,valid_chunks=valid,encoder_sha256=expected['encoder']))
    del rgb,tokens,rays,z
   assert counters(before)==counters(ecc());print('FROZEN_SIDECAR',e['episode'],flush=True)
 mean=total/count;std=np.sqrt(np.maximum(total2/count-mean**2,0));stats=dict(count=count,mean=mean.tolist(),std=std.tolist(),std_floor=.001,scope='only training complete physical windows, both appearances, all8 valid cumulative horizons; no development')
 for c in keys:
  save(a.out/c/'libero_goal/keys.json',keys[c]);save(a.out/c/'stats.json',stats);save(a.out/c/'manifest.json',dict(chunk=8,num_tokens=16,token_dim=64,pairs='anchored',format='npy',suites=['libero_goal'],appearance=c,encoder=str(init/'ckpts/encoder_step0090000.pt'),encoder_sha256=expected['encoder'],step=90000,img_size=224))
 save(a.out/'sidecar_provenance.json',records);save(a.out/'normalization.json',dict(mean=mean.tolist(),scale=float(np.sqrt((mean**2+std**2).mean()-mean.mean()**2)),definition='official center: percoordinate training mean, scalar global standard deviation',stats_sha256=sha(a.out/'clean/stats.json')))
 save(a.out/'sidecar_alignment_audit.json',dict(status='PASS',checks=checks,reconstruction=recons,shape='N,8,16,64',formula='t -> min(t+h,N-1), h1..8',tail='official cache clamping with explicit mask; policy excludes all incomplete anchors',source_RGB_recompute=True,AE_updates=0,DINO_updates=0,policy_updates=0,backward=0,task_pretraining_exclusion='not established'))
if __name__=='__main__':main()
