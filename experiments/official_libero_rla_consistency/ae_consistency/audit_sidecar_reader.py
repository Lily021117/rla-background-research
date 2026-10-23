"""CPU official RlaStore joins every model/appearance/episode; no forward."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args();sys.path[:0]=[str(a.project/'experiments/official_libero_rla_consistency/run_20261023_02/vla-official'),str(a.project/'worktrees/rla-official-0fc6e97/rebuttal/exp2_vla_adapter/vla_adapter_overlay')]
from rla.sidecar import RlaStore
es=json.loads((a.run/'data_manifest.json').read_text())['episodes'];pro=json.loads((a.run/'sidecar_provenance.json').read_text());rows=[];stat_hashes=[]
for arm in ('C','D'):
    for c in ('clean','background'):
        root=a.run/'sidecars'/arm/c;store=RlaStore(str(root),chunk=8,num_tokens=16,token_dim=64,norm='center');stat_hashes.append(hashlib.sha256((root/'stats.json').read_bytes()).hexdigest())
        for e in es:
            states=np.load(Path(e['root'])/'trajectory_labels.npz')['proprio'];slot=store.slot(states);assert slot>=0;meta=next(x for x in pro if x['model']==arm and x['appearance']==c and x['episode']==e['episode']);block=np.load(meta['file'],mmap_mode='r')
            for t in (0,e['valid_chunks']//2,e['valid_chunks']-1):
                z=store.frame(slot,t);expected=((block[t].astype(np.float32)-store.offset)/store.scale).reshape(8,1024);np.testing.assert_array_equal(z,expected);rows.append(dict(model=arm,appearance=c,episode=e['episode'],timestep=t,slot=slot,shape=list(z.shape),finite=bool(np.isfinite(z).all()),join_exact=True))
assert len(set(stat_hashes))==1
with (a.run/'official_sidecar_reader_audit.json').open('x') as f:json.dump(dict(status='PASS',episodes_per_store=25,stores=4,checks=rows,common_stats_sha256=stat_hashes[0],policy_updates=0,GPU_forward=0),f,indent=2)
