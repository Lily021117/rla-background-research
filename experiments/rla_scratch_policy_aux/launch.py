"""One isolated stage, authorized GPU0 only, fail closed and preserve logs."""
import os,sys,subprocess,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_scratch_policy_aux.common import *
def environment(sidecar=False):
 env=os.environ.copy();cache=Path('/data/lyc/tmp/rla_background_invariance/cache')
 for k,n in {'HF_HOME':'huggingface','HUGGINGFACE_HUB_CACHE':'huggingface/hub','TORCH_HOME':'torch','XDG_CACHE_HOME':'xdg','TMPDIR':'tmp','CUDA_CACHE_PATH':'cuda','TRITON_CACHE_DIR':'triton','MPLCONFIGDIR':'matplotlib','HF_DATASETS_CACHE':'huggingface/datasets','NUMBA_CACHE_DIR':'numba'}.items():
  p=cache/n;p.mkdir(parents=True,exist_ok=True);env[k]=str(p)
 env.update(SCRATCH_POLICY_ROOT=str(ROOT),RLA_GPU_UUID=GPU,CUDA_VISIBLE_DEVICES=GPU,RLA_MIN_FREE_GIB='26',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',LP_NUM_THREADS='2',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',HF_HUB_OFFLINE='1',LIBERO_CONFIG_PATH='/data/lyc/projects/LIBERO-plus/.libero',RLA_STEPS='8',RLA_QUERIES='16',RLA_DIM='64',RLA_NORM='center',LAMBDA_ACT='1',LAMBDA_RLA=str(LAMBDA),RLA_LOG_EVERY='0',RLA_MASK='1',RLA_ACTION_TOKENS='1',RLA_INIT_FROM='',RLA_SHUFFLE_TARGETS='0')
 paths=[str(HERE.parent),str(HERE.parent/'official_libero_rla_consistency'),'/data/lyc/projects/LIBERO-plus']
 if sidecar:paths+=[str(P/'worktrees/rla-official-0fc6e97'),str(P/'worktrees/rla-official-0fc6e97/third_party/diffusion_policy')]
 env['PYTHONPATH']=':'.join(paths);return env
def run(stage,order=None,arm=None):
 free=int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True));assert free>=26624,('resource_gate',free)
 assert int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True))==0
 if stage=='extract':
  root=ROOT;label='extraction';cmd=[str(P/'.venv-sidecar/bin/python'),'-u',str(HERE/'extract.py')]
 else:
  root=ROOT/f'seed_{order}' if order else ROOT/f'evaluation_{stage}';label=arm or stage
  cmd=[sys.executable,'-u',str(HERE.parent/'rla_lr_budget_sweep/bootstrap.py'),'--out',str(root),'--arm',arm or 'evaluation','--entry','rla_scratch_policy_aux.training' if order else 'rla_scratch_policy_aux.evaluation','--attempt','formal']
 t=time.time()
 with (root/(label+'.log')).open('x') as f:r=subprocess.run(cmd,cwd=CODE,env=environment(stage=='extract'),stdout=f,stderr=subprocess.STDOUT)
 save(root/(label+'_process.json'),dict(returncode=r.returncode,seconds=time.time()-t,command=cmd,GPU=GPU))
 assert r.returncode==0,(stage,order,arm,r.returncode)
if __name__=='__main__':run(sys.argv[1],int(sys.argv[2]) if len(sys.argv)>2 else None,sys.argv[3] if len(sys.argv)>3 else None)
