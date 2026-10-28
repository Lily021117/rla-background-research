"""Confirm the deduplicated initial snapshots also share complete saved RNG."""
import hashlib,json,pickle,sys
from pathlib import Path
import torch
out=Path(sys.argv[1]);rows=[]
for r in json.loads((out/'checkpoint_inventory.json').read_text()):
 if r['step']!=0:continue
 ck=torch.load(r['path'],map_location='cpu',weights_only=False);h=hashlib.sha256()
 h.update(pickle.dumps((ck['python_rng'],ck['numpy_rng'])))
 for t in [ck['torch_rng']]+ck['cuda_rng']:h.update(t.cpu().numpy().tobytes())
 rows.append(dict(order=r['order'],method=r['method'],checkpoint_sha256=r['sha256'],RNG_sha256=h.hexdigest()));del ck
assert len({r['RNG_sha256'] for r in rows})==1
with (out/'initial_rng_audit.json').open('x') as f:json.dump(dict(status='PASS',rows=rows,complete_saved_RNG_equal=True),f,indent=2)
