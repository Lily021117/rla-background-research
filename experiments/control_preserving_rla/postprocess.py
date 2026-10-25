"""CPU-only summaries after the complete 240-trial evaluation succeeds."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();src=Path(__file__).parent
f=a.out/'evaluate_formal_process.json'
while not f.exists():time.sleep(5)
assert json.loads(f.read_text())['returncode']==0
env=os.environ.copy();cache='/data/lyc/tmp/rla_background_invariance/cache'
env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',HF_HOME=cache+'/huggingface',TORCH_HOME=cache+'/torch',XDG_CACHE_HOME=cache+'/xdg',TMPDIR=cache+'/tmp',MPLCONFIGDIR=cache+'/matplotlib',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',LP_NUM_THREADS='2',PYTHONPATH=str(src.parent)+':/data/lyc/projects/LIBERO-plus',LIBERO_CONFIG_PATH='/data/lyc/projects/LIBERO-plus/.libero')
for name in ('analyze','finalize'):
 with (a.out/(name+'.log')).open('x') as log:
  r=subprocess.run(['/data/lyc/projects/vla-adapter-ref/.venv-cpu/bin/python','-u',str(src/(name+'.py')),'--out',str(a.out)],env=env,stdout=log,stderr=subprocess.STDOUT)
 assert r.returncode==0,(name,r.returncode)
 print(name,'COMPLETE',flush=True)
