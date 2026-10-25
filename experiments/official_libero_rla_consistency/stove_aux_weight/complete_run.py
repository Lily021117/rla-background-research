"""Bounded continuation of the already authorized run; no automatic retries."""
import argparse,json,time,subprocess,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
while True:
 checks=[a.out/f'train-{arm}_isolated_smoke_process.json' for arm in ('C-low','D-center-low')]
 for f in checks:
  if f.exists():assert json.loads(f.read_text())['returncode']==0,f
 if all(f.exists() for f in checks):break
 time.sleep(45)
for arm in ('C-low','D-center-low'):assert json.loads((a.out/arm/'isolated_smoke/training_acceptance.json').read_text())['successful_updates']==600
src=Path(__file__).parent;env=os.environ.copy();cache=Path('/data/lyc/tmp/rla_background_invariance/cache')
env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',HF_HOME=str(cache/'huggingface'),TORCH_HOME=str(cache/'torch'),XDG_CACHE_HOME=str(cache/'xdg'),TMPDIR=str(cache/'tmp'),MPLCONFIGDIR=str(cache/'matplotlib'),MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',LP_NUM_THREADS='2',LIBERO_CONFIG_PATH='/data/lyc/projects/LIBERO-plus/.libero',PYTHONPATH='/data/lyc/projects/LIBERO-plus:'+env.get('PYTHONPATH',''))
for stage in ('evaluate','diagnostics'):
 subprocess.run(['/usr/bin/python3',str(src/'launch.py'),'--out',str(a.out),'--stage',stage],env=env,check=True)
env['CUDA_VISIBLE_DEVICES']=''
for stage in ('analyze','finalize'):
 with (a.out/(stage+'.log')).open('x') as f:subprocess.run(['/data/lyc/projects/vla-adapter-ref/.venv-cpu/bin/python','-u',str(src/(stage+'.py')),'--out',str(a.out)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
print('AUTHORIZED_BOUNDED_RUN_COMPLETE',flush=True)
