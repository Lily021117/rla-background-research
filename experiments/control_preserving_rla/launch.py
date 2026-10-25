"""Authorized UUID0/1 only, data-volume caches, isolated resource-checked stages."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',choices=['teacher','gate','resume-precheck','E-C','E-D','E-C-keep','E-D-keep','evaluate'],required=True);p.add_argument('--attempt',default='formal');p.add_argument('--gpu',choices=['0','1'],default='0');a=p.parse_args()
cache=Path('/data/lyc/tmp/rla_background_invariance/cache');env=os.environ.copy()
for k,v in {'HF_HOME':cache/'huggingface','HUGGINGFACE_HUB_CACHE':cache/'huggingface/hub','TORCH_HOME':cache/'torch','XDG_CACHE_HOME':cache/'xdg','TMPDIR':cache/'tmp','CUDA_CACHE_PATH':cache/'cuda','TRITON_CACHE_DIR':cache/'triton','MPLCONFIGDIR':cache/'matplotlib','HF_DATASETS_CACHE':cache/'huggingface/datasets','NUMBA_CACHE_DIR':cache/'numba'}.items():v.mkdir(parents=True,exist_ok=True);env[k]=str(v)
env.update(RLA_GPU_UUID='GPU-205a3f5a-d91b-6385-c588-4e4cf3505f7e',CUDA_VISIBLE_DEVICES='GPU-205a3f5a-d91b-6385-c588-4e4cf3505f7e',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',LP_NUM_THREADS='2',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',RLA_MIN_FREE_GIB='26',HF_HUB_OFFLINE='1',LIBERO_CONFIG_PATH='/data/lyc/projects/LIBERO-plus/.libero')
if a.gpu=='1':env.update(RLA_GPU_UUID='GPU-37d3f7d4-309e-2bb8-2d51-292dbbcce419',CUDA_VISIBLE_DEVICES='GPU-37d3f7d4-309e-2bb8-2d51-292dbbcce419')
env.update(RLA_STEPS='8',RLA_QUERIES='16',RLA_DIM='64',RLA_NORM='center',LAMBDA_ACT='1',LAMBDA_RLA='0.22408536585365854',RLA_LOG_EVERY='1000',RLA_MASK='1',RLA_ACTION_TOKENS='1',RLA_INIT_FROM='',RLA_SHUFFLE_TARGETS='0')
env['PYTHONPATH']='/data/lyc/projects/LIBERO-plus:'+env.get('PYTHONPATH','')
arm,entry={'teacher':('teacher','teacher'),'gate':('preflight','gate'),'resume-precheck':('preflight','resume_precheck'),'evaluate':('evaluation','evaluation')}.get(a.stage,(a.stage,'training'))
if entry=='training':assert json.loads((a.out/'gate_acceptance.json').read_text())['status']=='PASS'
cmd=['/data/lyc/projects/vla-adapter-ref/.venv-cpu/bin/python','-u',str(Path(__file__).with_name('bootstrap.py')),'--out',str(a.out),'--arm',arm,'--entry','control_preserving_rla.'+entry,'--attempt',a.attempt]
start=time.time()
with (a.out/(a.stage+'_'+a.attempt+'.log')).open('x') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
with (a.out/(a.stage+'_'+a.attempt+'_process.json')).open('x') as f:json.dump(dict(returncode=r.returncode,seconds=time.time()-start,command=cmd),f,indent=2)
raise SystemExit(r.returncode)
