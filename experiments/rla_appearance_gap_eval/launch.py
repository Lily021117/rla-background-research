"""Authorized GPU0 only, unchanged deployed model/backend; no training entry."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_appearance_gap_eval.common import GPU,HERE,REPO,save
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
assert int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))>=26624
assert int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True))==0
env=os.environ.copy();cache=Path('/data/lyc/tmp/rla_background_invariance/cache');src=HERE.parent
for k,v in {'HF_HOME':cache/'huggingface','HUGGINGFACE_HUB_CACHE':cache/'huggingface/hub','TORCH_HOME':cache/'torch','XDG_CACHE_HOME':cache/'xdg','TMPDIR':cache/'tmp','CUDA_CACHE_PATH':cache/'cuda','TRITON_CACHE_DIR':cache/'triton','MPLCONFIGDIR':cache/'matplotlib','HF_DATASETS_CACHE':cache/'huggingface/datasets','NUMBA_CACHE_DIR':cache/'numba'}.items():v.mkdir(parents=True,exist_ok=True);env[k]=str(v)
env.update(RLA_GPU_UUID=GPU,CUDA_VISIBLE_DEVICES=GPU,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',LP_NUM_THREADS='2',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',HF_HUB_OFFLINE='1',LIBERO_CONFIG_PATH='/data/lyc/projects/LIBERO-plus/.libero',RLA_STEPS='8',RLA_QUERIES='16',RLA_DIM='64',RLA_NORM='center',LAMBDA_ACT='1',LAMBDA_RLA='0.22408536585365854',RLA_LOG_EVERY='1000',RLA_MASK='1',RLA_ACTION_TOKENS='1',RLA_INIT_FROM='',RLA_SHUFFLE_TARGETS='0')
env['PYTHONPATH']='/data/lyc/projects/LIBERO-plus:'+str(src)+':'+str(src/'official_libero_rla_consistency')+':'+env.get('PYTHONPATH','')
cmd=[sys.executable,'-u',str(src/'rla_lr_budget_sweep/bootstrap.py'),'--out',str(a.out),'--arm','evaluation','--entry','rla_appearance_gap_eval.evaluate','--attempt','formal'];start=time.time()
save(a.out/'evaluation_start.json',dict(command=cmd,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),GPU=GPU,ECC_baseline=subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total','--format=csv'],text=True)))
with (a.out/'evaluation.log').open('x') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
save(a.out/'evaluation_process.json',dict(returncode=r.returncode,seconds=time.time()-start,command=cmd));sys.exit(r.returncode)
