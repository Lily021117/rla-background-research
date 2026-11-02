import os,json,hashlib,csv
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;CODE=HERE.parents[1];P=Path(os.environ.get('RLA_PROJECT_ROOT',str(CODE)));ROOT=Path(os.environ.get('PAIRED_V2_ROOT',str(CODE.parent)))
LIB=Path(os.environ.get('RLA_LIBERO_ROOT',str(P.parent/'LIBERO-plus')));VARIANTS=['clean','object_1','object_2','light_intensity','light_direction'];CAMS=['agentview','robot0_eye_in_hand']
STOVE=P/'experiments/official_libero_rla_consistency/run_20261023_10/stove'
DRAWER=P/'experiments/rla_new_task_baseline/run_20261028_02/dataset_v2'
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def table(p,rows):
 with Path(p).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def frozen_source(task):return (STOVE if task=='stove' else DRAWER)/'frozen_dataset_manifest.json'
