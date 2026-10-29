"""Sequential bounded stages, isolated process per arm; fail closed on first error."""
import os,subprocess,json,time
from pathlib import Path
ROOT=Path('/data/lyc/projects/rla-wm/experiments/rla_ae_from_scratch/run_20261029_01')
CODE=ROOT/'code'
PY='/data/lyc/projects/rla-wm/.venv-sidecar/bin/python'
env=os.environ.copy()
for key,name in [('HF_HOME','huggingface'),('TORCH_HOME','torch'),('XDG_CACHE_HOME','xdg'),('TMPDIR','tmp'),('MPLCONFIGDIR','matplotlib')]:
 p=ROOT/'cache'/name;p.mkdir(parents=True,exist_ok=True);env[key]=str(p)
env.update(CUDA_VISIBLE_DEVICES='GPU-205a3f5a-d91b-6385-c588-4e4cf3505f7e',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
env['PYTHONPATH']='/data/lyc/projects/rla-wm/worktrees/rla-official-0fc6e97:/data/lyc/projects/rla-wm/worktrees/rla-official-0fc6e97/third_party/diffusion_policy'
stages=os.sys.argv[1:] or ['prepare','init','pilot','C','D']
for stage in stages:
 suffix='' if not (ROOT/f'{stage}.log').exists() else '_retry_'+str(time.time_ns())
 with (ROOT/f'{stage}{suffix}.log').open('x') as f:
  begin=time.time();p=subprocess.run([PY,str(CODE/'experiments/rla_ae_from_scratch/run.py'),stage],cwd=CODE,env=env,stdout=f,stderr=subprocess.STDOUT)
 with (ROOT/f'{stage}{suffix}_process.json').open('x') as f:json.dump(dict(stage=stage,returncode=p.returncode,seconds=time.time()-begin),f,indent=2)
 if p.returncode:raise SystemExit(p.returncode)
