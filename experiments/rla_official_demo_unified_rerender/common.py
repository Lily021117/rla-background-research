"""Official-demo unified rerender: CPU only; never imports learned models."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', MUJOCO_GL='osmesa', PYOPENGL_PLATFORM='osmesa', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', LP_NUM_THREADS='2')
from pathlib import Path
import csv, json, hashlib
import numpy as np

ROOT=Path('/data/lyc/projects/rla-wm')
RUN=ROOT/'experiments/rla_official_demo_unified_rerender/run_20261103_01'
OLD=ROOT/'experiments/rla_official_ae_paired_rerender/run_20261103_01'
REPAIR=ROOT/'experiments/rla_official_ae_sync_repair/run_20261103_01'
OFFICIAL=ROOT/'worktrees/rla-official-0fc6e97'
SUITES=['libero_spatial','libero_object','libero_goal','libero_10']
CAMS=['agentview','robot0_eye_in_hand']
STREAMS=['agentview_camera','wrist_camera']
WHITELIST=['floor','table_visual','wall_leftcorner_visual','wall_rightcorner_visual','wall_left_visual','wall_right_visual','wall_rear_visual','wall_front_visual']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.tmp-{os.getpid()}');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def csvout(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+f'.tmp-{os.getpid()}')
 with open(tmp,'w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 tmp.replace(p)
def filtered(actions):
 ids=[];prev=None
 for i,a in enumerate(actions):
  if np.linalg.norm(a[:-1])<1e-4 and (prev is None or a[-1]==prev[-1]):continue
  ids.append(i);prev=a
 return np.array(ids,dtype=np.int64)
