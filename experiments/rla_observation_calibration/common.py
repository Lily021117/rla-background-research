import os,json,hashlib,csv
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;CODE=HERE.parents[1]
P=Path(os.environ['RLA_PROJECT_ROOT']);BASE=Path(os.environ['OBS_CAL_ROOT'])
OLD=P/'experiments/rla_v2_frozen_c_eval/run_20261102_01'
GPU='GPU-205a3f5a-d91b-6385-c588-4e4cf3505f7e'
CONDITIONS=['blur_1','blur_2','blur_3','noise_003','noise_007','noise_012','crop_095','crop_085','crop_075']
COMBO='blur2_noise007';SEEDS=[31001,31002,31003];CAMS=['agentview','robot0_eye_in_hand']
TASK_NAMES={'stove':'pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate','drawer':'open_the_middle_drawer_of_the_cabinet'}
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def table(p,rows):
 with Path(p).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))));w.writeheader();w.writerows(rows)
