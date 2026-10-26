"""Train-only FP32 center, unchanged C scale; exact official reader checks."""
import argparse,json,os,sys,hashlib,csv
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();o=a.out;ae=o/'ae_run'
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
plan=read(o/'run_plan.json');H=Path(plan['historical_center_source']);old=read(H/'normalization.json')
assert read(ae/'AE_D_inv03/acceptance.json')['successful_updates']==600
assert read(ae/'raw_source_recompute_audit.json')['status']=='PASS'
stats=read(ae/'sidecars/D/clean/raw_target_statistics.json');assert stats==read(ae/'sidecars/D/background/raw_target_statistics.json')
mu=np.asarray(stats['mean'],np.float32);assert mu.shape==(16,64) and np.isfinite(mu).all()
meta=dict(mu_C=old['mu_C'],mu_D=mu.tolist(),s_C=old['s_C'],statistics_D=stats,statistic_count=stats['count'],weighting=old['weighting'],source_precision=old['source_precision'],arm='D-inv03',source=str(ae/'sidecars/D/clean/raw_target_statistics.json'),historical_scale_metadata_sha256=sha(H/'normalization.json'))
save(o/'normalization.json',meta)
R=Path(plan['inputs']['project']);sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'official_libero_rla_consistency'),str(R/'experiments/official_libero_rla_consistency/run_20261023_02/vla-official'),str(R/'worktrees/rla-official-0fc6e97/rebuttal/exp2_vla_adapter/vla_adapter_overlay')]
os.environ['CENTER_RUN']=str(o);os.environ['CENTER_ARM']='D-inv03'
from stove_centering.store import CenterStore
es=read(ae/'data_manifest.json')['episodes'];prov=read(ae/'sidecar_provenance.json');total=np.zeros((16,64),np.float64);count=0;checks=[]
for c in ('clean','background'):
 root=ae/'sidecars/D'/c;store=CenterStore(str(root),chunk=8,num_tokens=16,token_dim=64,norm='center')
 assert store.scale==np.float32(old['s_C']);np.testing.assert_array_equal(store.offset,mu)
 for e in es:
  pr=next(x for x in prov if x['episode']==e['episode'] and x['appearance']==c);assert sha(pr['file'])==pr['sha256']
  z=np.load(pr['file'],mmap_mode='r');mask=np.load(pr['mask_file']);n=e['valid_chunks'];assert z.shape==(e['frames'],8,16,64) and mask[:n].all()
  labels=np.load(Path(e['root'])/'trajectory_labels.npz');np.testing.assert_array_equal(mask,labels['valid_horizon_mask']);slot=store.slot(labels['proprio']);assert slot>=0
  for t in (0,n//2,n-1):
   actual=store.frame(slot,t);expected=((z[t].astype(np.float32)-mu)/np.float32(old['s_C'])).reshape(8,1024);np.testing.assert_array_equal(actual,expected);assert np.isfinite(actual).all()
   checks.append(dict(episode=e['episode'],appearance=c,split=e['split'],t=t,shape=list(actual.shape),reader_exact=True))
  if e['split']=='train':
   for start in range(0,n,16):
    v=z[start:min(start+16,n)].astype(np.float64).reshape(-1,16,64);assert np.isfinite(v).all();total+=v.sum(0);count+=len(v)
assert count==stats['count']==40880
err=float(abs(total/count-np.asarray(stats['mean'])).max());assert err<1e-3
save(o/'target_readiness.json',dict(status='PASS',arm='D-inv03',checks=checks,training_moment_rows=count,mean_source='FP32 encoder output, FP64 accumulation before FP16 storage; complete train windows both appearances h1..8',FP16_recomputed_mean_max_error=err,FP16_check_threshold=1e-3,s_C_unchanged=True,raw_source_recompute=read(ae/'raw_source_recompute_audit.json'),cache_alignment=read(ae/'sidecar_alignment_audit.json'),policy_updates=0,development_excluded_from_mean=True))
print('TARGET_READER_PASS',count,len(checks),err)
