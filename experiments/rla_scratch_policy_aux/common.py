"""Frozen downstream experiment constants; all large outputs remain outside git."""
import csv, hashlib, json, os
from pathlib import Path
P=Path('/data/lyc/projects/rla-wm')
ROOT=Path(os.environ.get('SCRATCH_POLICY_ROOT',str(P/'experiments/rla_scratch_policy_aux/run_20261029_01')))
AE=P/'experiments/rla_ae_from_scratch/run_20261029_01'
OLD=P/'experiments/rla_sampling_stability/run_20261027_01'
GAP=P/'experiments/rla_appearance_gap_eval/run_20261027_01'
HERE=Path(__file__).resolve().parent
CODE=HERE.parents[1]
GPU='GPU-205a3f5a-d91b-6385-c588-4e4cf3505f7e'
ORDERS=[20261101,20261102]
ARMS=['C-fixed','D-fixed','D-low','D-decay']
LAMBDA=.22408536585365854
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def table(p,rows):
 with Path(p).open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def weight(arm,k):
 assert arm in ARMS and 0<=k<600
 return LAMBDA*(.5 if arm=='D-low' else 1-k/599 if arm=='D-decay' else 1.)
