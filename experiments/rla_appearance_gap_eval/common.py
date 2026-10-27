"""Frozen development-only appearance evaluation, no optimization."""
import csv,hashlib,json
from pathlib import Path
R=Path('/data/lyc/projects/rla-wm')
C=R/'experiments/rla_sampling_stability/run_20261027_01'
D=R/'experiments/rla_sampling_stability/run_20261027_02'
SHARED=R/'experiments/rla_shared_reference_target/run_20261027_01'
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ORDERS=(20261101,20261102)
GPU='GPU-205a3f5a-d91b-6385-c588-4e4cf3505f7e'
CONDITIONS=['clean']+[f'new_background_{i}' for i in range(1,5)]
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def table(p,rows):
 with Path(p).open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
