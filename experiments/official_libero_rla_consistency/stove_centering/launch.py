"""Data-disk-only environment; sequential bounded stages, stop on first failure."""
import argparse,os,subprocess,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',choices=['gate','train-old','train-center','evaluate','diagnostics'],required=True);p.add_argument('--attempt',default='isolated_smoke');a=p.parse_args()
cache=Path('/data/lyc/tmp/rla_background_invariance/cache');env=os.environ.copy()
for k,v in {'HF_HOME':cache/'huggingface','HUGGINGFACE_HUB_CACHE':cache/'huggingface/hub','TORCH_HOME':cache/'torch','XDG_CACHE_HOME':cache/'xdg','TMPDIR':cache/'tmp','CUDA_CACHE_PATH':cache/'cuda','TRITON_CACHE_DIR':cache/'triton','MPLCONFIGDIR':cache/'matplotlib'}.items():v.mkdir(parents=True,exist_ok=True);env[k]=str(v)
env.update(RLA_GPU_UUID='GPU-37d3f7d4-309e-2bb8-2d51-292dbbcce419',CUDA_VISIBLE_DEVICES='GPU-37d3f7d4-309e-2bb8-2d51-292dbbcce419',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',LP_NUM_THREADS='2',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',RLA_MIN_FREE_GIB='26',TRANSFORMERS_CACHE=str(cache/'huggingface/hub'),HF_DATASETS_CACHE=str(cache/'huggingface/datasets'),NUMBA_CACHE_DIR=str(cache/'numba'),HF_HUB_OFFLINE='1')
arm,entry={'gate':('preflight','stove_centering.gate'),'train-old':('D-old','stove_centering.training'),'train-center':('D-center','stove_centering.training'),'evaluate':('evaluation','stove_centering.evaluation'),'diagnostics':('diagnostics','stove_centering.diagnostics')}[a.stage]
env.update(RLA_STEPS='8',RLA_QUERIES='16',RLA_DIM='64',RLA_NORM='center',LAMBDA_ACT='1',LAMBDA_RLA='0.22408536585365854',RLA_LOG_EVERY='1000',RLA_MASK='1',RLA_ACTION_TOKENS='1',RLA_INIT_FROM='',RLA_SHUFFLE_TARGETS='0')
env['PYTHONPATH']='/data/lyc/projects/LIBERO-plus:'+env.get('PYTHONPATH','')
env['LIBERO_CONFIG_PATH']='/data/lyc/projects/LIBERO-plus/.libero'
if a.stage.startswith('train'):assert json.loads((a.out/'gate_acceptance.json').read_text())['status']=='PASS'
cmd=['/data/lyc/projects/vla-adapter-ref/.venv-cpu/bin/python','-u',str(Path(__file__).with_name('bootstrap.py')),'--out',str(a.out),'--arm',arm,'--entry',entry,'--attempt',a.attempt]
start=time.time()
with (a.out/(a.stage+'_'+a.attempt+'.log')).open('x') as f:result=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
with (a.out/(a.stage+'_'+a.attempt+'_process.json')).open('x') as f:json.dump(dict(returncode=result.returncode,seconds=time.time()-start,command=cmd),f,indent=2)
if a.stage=='gate' and result.returncode==0:
 with (a.out/'gate_acceptance.json').open('x') as f:json.dump(json.loads((a.out/'preflight'/a.attempt/'gate.json').read_text()),f,indent=2)
raise SystemExit(result.returncode)
