"""Bounded single stage on authorized GPU0, stop on any failure; no silent retry."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--stage',choices=['gate','train','evaluate','representation'],required=True);a=p.parse_args()
src=Path(__file__).resolve().parents[1];run=a.out/f'seed_{a.seed}';gpu='GPU-205a3f5a-d91b-6385-c588-4e4cf3505f7e'
assert int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))>=26624
assert int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True))==0
env=os.environ.copy();cache=Path('/data/lyc/tmp/rla_background_invariance/cache')
for k,v in {'HF_HOME':cache/'huggingface','HUGGINGFACE_HUB_CACHE':cache/'huggingface/hub','TORCH_HOME':cache/'torch','XDG_CACHE_HOME':cache/'xdg','TMPDIR':cache/'tmp','CUDA_CACHE_PATH':cache/'cuda','TRITON_CACHE_DIR':cache/'triton','MPLCONFIGDIR':cache/'matplotlib','HF_DATASETS_CACHE':cache/'huggingface/datasets','NUMBA_CACHE_DIR':cache/'numba'}.items():v.mkdir(parents=True,exist_ok=True);env[k]=str(v)
env.update(RLA_GPU_UUID=gpu,CUDA_VISIBLE_DEVICES=gpu,RLA_MIN_FREE_GIB='26',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',LP_NUM_THREADS='2',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',HF_HUB_OFFLINE='1',LIBERO_CONFIG_PATH='/data/lyc/projects/LIBERO-plus/.libero',RLA_STEPS='8',RLA_QUERIES='16',RLA_DIM='64',RLA_NORM='center',LAMBDA_ACT='1',LAMBDA_RLA='0.22408536585365854',RLA_LOG_EVERY='1000',RLA_MASK='1',RLA_ACTION_TOKENS='1',RLA_INIT_FROM='',RLA_SHUFFLE_TARGETS='0')
env['PYTHONPATH']='/data/lyc/projects/LIBERO-plus:'+str(src)+':'+str(src/'official_libero_rla_consistency')+':'+env.get('PYTHONPATH','')
arm,entry={'gate':('preflight','stove_centering.training'),'train':('C-shared-target','stove_centering.training'),'evaluate':('evaluation','rla_lr_budget_sweep.extend_evaluation'),'representation':('representation','rla_shared_reference_target.representation')}[a.stage]
if a.stage=='train':assert json.loads((a.out/'seed_20261101/preflight/formal/shared_reference_gate.json').read_text())['status']=='PASS'
attempt=os.environ.get('SHARED_DIAGNOSTIC_ATTEMPT','formal')
assert attempt=='formal' or a.stage=='representation','alternate attempt only for frozen diagnostic'
suffix='' if attempt=='formal' else '_'+attempt
cmd=[sys.executable,'-u',str(src/'rla_lr_budget_sweep/bootstrap.py'),'--out',str(run),'--arm',arm,'--entry',entry,'--attempt',attempt];t=time.time()
with (run/(a.stage+suffix+'.log')).open('x') as f:result=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
with (run/(a.stage+suffix+'_process.json')).open('x') as f:json.dump(dict(returncode=result.returncode,seconds=time.time()-t,command=cmd),f,indent=2)
sys.exit(result.returncode)
