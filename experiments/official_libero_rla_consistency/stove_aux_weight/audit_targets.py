"""Verify the low-weight experiment does not alter either target family."""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']=''
cache=Path('/data/lyc/tmp/rla_background_invariance/cache')
for k,v in {'HF_HOME':cache/'huggingface','TORCH_HOME':cache/'torch','XDG_CACHE_HOME':cache/'xdg','TMPDIR':cache/'tmp'}.items():os.environ[k]=str(v)
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();plan=json.loads((a.out/'run_plan.json').read_text())
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(plan['inputs']['project'])/'worktrees/rla-official-0fc6e97/rebuttal/exp2_vla_adapter/vla_adapter_overlay'))
os.environ.update(CENTER_RUN=str(a.out),CENTER_ARM='D-center',RLA_STEPS='8',RLA_QUERIES='16',RLA_DIM='64',RLA_NORM='center')
sys.path.insert(0,str(Path(plan['inputs']['project'])/'experiments/official_libero_rla_consistency/run_20261023_02/vla-official'))
from rla.sidecar import RlaStore
from stove_centering.store import CenterStore
from stove_aux_weight.store import LowWeightStore
checks=[]
for source in ('C','D'):
 for app in ('clean','background'):
  path=str(Path(plan['previous'])/'sidecars'/source/app)
  original=(RlaStore if source=='C' else CenterStore)(path,chunk=8,num_tokens=16,token_dim=64,norm='center')
  new=LowWeightStore(path,chunk=8,num_tokens=16,token_dim=64,norm='center')
  assert np.array_equal(original.offset,new.offset) and original.scale==new.scale
  # Same implementation and data mapping; compare fixed cumulative frames.
  n=0
  for slot in range(min(2,len(new))):
   for t in (0,1,8):
    x=original.frame(slot,t);y=new.frame(slot,t);np.testing.assert_array_equal(x,y);n+=1
  checks.append(dict(source=source,appearance=app,identical_center=True,identical_scale=True,identical_targets=n))
with (a.out/'target_identity_audit.json').open('x') as f:json.dump(dict(status='PASS',policy_weight_only=True,checks=checks),f,indent=2)
