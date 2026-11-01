"""Fixed drawer capability / appearance-gap protocol; no adaptive search."""
import os,json,hashlib
from pathlib import Path
P=Path(os.environ.get('RLA_PROJECT','/data/lyc/projects/rla-wm'))
HERE=Path(__file__).resolve().parent
CODE=HERE.parents[1]
ROOT=Path(os.environ.get('DRAWER_GAP_ROOT',str(CODE.parent)))
SOURCE=P/'experiments/rla_new_task_baseline/run_20261028_02'
DATA=SOURCE/'dataset_v2'
AE=P/'experiments/rla_ae_from_scratch/run_20261029_01'
GPU='GPU-205a3f5a-d91b-6385-c588-4e4cf3505f7e'
SEEDS=[31001,31002,31003]
METHODS=['BC-standard','C-low']
LAMBDA=.11204268292682927
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def replace(source,old,new):
 assert source.count(old)==1,(old,source.count(old))
 return source.replace(old,new)
