"""Shared read-only context for Step5.10. Paths are supplied by the operator."""
import os,json,csv,hashlib,subprocess
from pathlib import Path
import numpy as np
R=Path(os.environ['RLA_PROJECT_ROOT']);E=R/'experiments/rla_background_invariance'
O=Path(os.environ['RUN_DIR']);W=R/'worktrees/rla-background-causal-unified-trainer'
P=E/'step5_9_base_degradation/run_20261022_01';T=E/'step5_7_causal_training/run_20261022_01'
S=E/'step5_1_policy_training/run_20261022_02';BASE=R/'artifacts/rla_policy_closed_loop_pilot/runtime_checkpoint'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def js(n,v):
 with (O/n).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False)
def csvread(p):return list(csv.DictReader(open(p)))
def cs(n,rows):
 with (O/n).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def git(p):return {k:subprocess.check_output(['git','-C',str(p)]+a,text=True) for k,a in {'HEAD':['rev-parse','HEAD'],'branch':['branch','--show-current'],'status':['status','--short'],'remote':['remote','-v']}.items()}
