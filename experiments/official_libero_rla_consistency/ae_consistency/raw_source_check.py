"""Re-extract selected source images and compare persisted sidecar targets."""
import argparse
from common import *
p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--models',nargs='+',choices=['C','D']);a=p.parse_args();ext,init=load(a.project);before=ecc();es=entries(a.run);rng=np.random.RandomState(20261023);selected=[(es[i],int(h)) for i,h in zip(rng.choice(len(es),3,replace=False),(1,4,8))];rows=[]
with torch.inference_mode():
    for arm in (a.models or ('C','D')):
        for k,m in ext.models.items():m.load_state_dict(torch.load(armdir(a.run,arm)/f'{k}_step0600.pt',map_location='cuda',weights_only=True),strict=True)
        for e,h in selected:
            t=e['valid_chunks']//2;root=Path(e['root']);labels=np.load(root/'trajectory_labels.npz');index=next(i for i,x in enumerate(es) if x['episode']==e['episode'])
            for c in ('clean','background'):
                files=[root/'frames'/f'{j:04}.npz' for j in (t,t+h)];rgb=np.stack([np.load(f)[c] for f in files]);data=ext._reference_batch(rgb,labels['K'][[t,t+h]].astype(np.float32),labels['c2w'][[t,t+h]].astype(np.float32),np.array([0]),np.array([1]));ref=ext.inference_batch(ext.models,data,training=False)['enc_tokens'].float().cpu().numpy()[0];saved=np.load(a.run/'sidecars'/arm/c/'libero_spatial'/f'ep_{index:06}.npy',mmap_mode='r')[t,h-1].astype(np.float32);error=float(abs(ref-saved).max());tol=max(.001,.02*float(abs(ref).max()));assert error<tol
                rows.append(dict(model=arm,episode=e['episode'],appearance=c,anchor=t,h=h,future=t+h,max_error=error,tolerance=tol,source_frame_files=[str(f) for f in files],source_frame_sha256=[sha(f) for f in files]))
assert counters(before)==counters(ecc());save(a.run/'raw_source_recompute_audit.json',dict(status='PASS',checks=rows,selection='seed20261023 choose3 episodes before model scoring, horizons1/4/8',same_preprocessing=True,no_new_training=True,ECC_increment=False))
