import os,json,hashlib,csv
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;CODE=HERE.parents[1];P=Path(os.environ.get('RLA_PROJECT_ROOT',str(CODE)));ROOT=Path(os.environ.get('V2_EVAL_ROOT',str(CODE.parent)))
V2=P/'experiments/rla_paired_data_v2/run_20261102_01/revision2'
STOVE=P/'experiments/rla_low_lr_continuation/run_20261030_01';DRAWER=P/'experiments/rla_drawer_baseline_gap/run_20261101_01'
DATA=P/'experiments/rla_new_task_baseline/run_20261028_02/dataset_v2'
GPU='GPU-205a3f5a-d91b-6385-c588-4e4cf3505f7e';SEEDS=[31001,31002,31003];CONDITIONS=['object_1','object_2','light_intensity','light_direction']
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def table(p,rows):
 with Path(p).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))));w.writeheader();w.writerows(rows)
