"""Quantify common-C centering's constant component, without changing targets."""
import argparse,collections
from common import *
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=Path(read(a.out/'configuration.json')['source']);ae=root/'ae_run';es=read(root/'frozen_dataset_manifest.json')['episodes'];pro=read(ae/'sidecar_provenance.json');lookup={(r['model'],r['episode'],r['appearance']):r for r in pro};st=read(ae/'sidecars/C/clean/stats.json');mu=np.array(st['mean']);std=np.array(st['std']);scale=np.sqrt(np.mean(mu**2+std**2)-mu.mean()**2);rows=[]
for arm in ('C','D'):
 for split in ('train','validation'):
  for app in ('clean','background'):
   n=0;s=np.zeros(1024);ss=np.zeros(1024);rawabs=0.;normalizedabs=0.
   for e in es:
    if e['split']!=split:continue
    r=lookup[arm,e['episode'],app];assert r['split']==split and r['source_demo']==e['source_demo'];path=Path(r['file']);keys=read(path.parent/'keys.json');expected=__import__('hashlib').sha1(np.ascontiguousarray(np.load(Path(e['root'])/'trajectory_labels.npz')['proprio'],dtype=np.float32).tobytes()).hexdigest();assert expected==r['key'] and keys[expected]['file']==path.name
    x=np.load(path,mmap_mode='r')[:e['valid_chunks']].reshape(-1,1024).astype(np.float64);z=(x-mu.reshape(1,-1))/scale;n+=len(z);s+=z.sum(0);ss+=(z*z).sum(0);normalizedabs+=abs(z).sum();rawabs+=abs(x).sum()
   mean=s/n;var=ss/n-mean**2;energy=float((ss/n).sum());constant=float((mean**2).sum());rows.append(dict(model=arm,split=split,appearance=app,latent_rows=n,common_global_std=float(scale),normalized_target_mean_norm=float(np.linalg.norm(mean)),normalized_target_RMS=float(np.sqrt(energy/1024)),centered_variance_mean=float(var.mean()),constant_mean_energy_fraction=constant/energy,dynamic_variance_energy_fraction=float(var.sum()/energy),mean_absolute_target=normalizedabs/(n*1024),raw_mean_absolute_target=rawabs/(n*1024)))
csvsave(a.out/'target_scale_and_offset.csv',rows);save(a.out/'target_join_verification.json',dict(episodes=25,models=2,appearances=2,checks=100,status='PASS_ALL_SOURCE_PROPRIO_HASH_TO_EPISODE_SIDECAR',common_statistics=True,normalization='(z - AE_C_train_per_coordinate_mean) / AE_C_train_global_std',constant_fraction='||mean(normalized target)||^2 / E||normalized target||^2; descriptive squared energy, NOT L1 loss fraction or gradient fraction',targets_modified=False))
