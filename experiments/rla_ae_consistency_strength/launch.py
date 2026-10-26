"""Single authorized GPU0 stages; all caches on data volume, immutable outputs."""
import argparse,os,json,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',choices=['ae','representation','sidecar','rawcheck','policy','evaluate'],required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];src=root/'official_libero_rla_consistency';cache=Path('/data/lyc/tmp/rla_background_invariance/cache');env=os.environ.copy()
for k,v in {'HF_HOME':cache/'huggingface','HUGGINGFACE_HUB_CACHE':cache/'huggingface/hub','TORCH_HOME':cache/'torch','XDG_CACHE_HOME':cache/'xdg','TMPDIR':cache/'tmp','CUDA_CACHE_PATH':cache/'cuda','TRITON_CACHE_DIR':cache/'triton','MPLCONFIGDIR':cache/'matplotlib','HF_DATASETS_CACHE':cache/'huggingface/datasets','NUMBA_CACHE_DIR':cache/'numba'}.items():v.mkdir(parents=True,exist_ok=True);env[k]=str(v)
env.update(RLA_GPU_UUID='GPU-205a3f5a-d91b-6385-c588-4e4cf3505f7e',CUDA_VISIBLE_DEVICES='GPU-205a3f5a-d91b-6385-c588-4e4cf3505f7e',RLA_MIN_FREE_GIB='26',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',LP_NUM_THREADS='2',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',HF_HUB_OFFLINE='1',LIBERO_CONFIG_PATH='/data/lyc/projects/LIBERO-plus/.libero')
env['PYTHONPATH']='/data/lyc/projects/LIBERO-plus:'+str(root)+':'+str(src)+':'+env.get('PYTHONPATH','')
project='/data/lyc/projects/rla-wm';ae=a.out/'ae_run'
if a.stage in ('ae','representation','sidecar','rawcheck'):
 py=project+'/.venv-sidecar/bin/python';base=['--project',project,'--run',str(ae)]
 if a.stage=='ae':
  env['RLA_EXPECTED_AE_ORDER']=json.loads((a.out/'experiment_protocol.json').read_text())['expected_AE_order'];cmd=[py,'-u',str(src/'ae_consistency/train_ae.py'),*base,'--arm','D','--output-name','AE_D_inv03']
 elif a.stage=='rawcheck':cmd=[py,'-u',str(src/'ae_consistency/raw_source_check.py'),*base,'--models','D']
 else:cmd=[py,'-u',str(src/'ae_consistency/evaluate_and_extract.py'),*base,'--mode','evaluate' if a.stage=='representation' else 'sidecar','--models','D','--common-stats',project+'/experiments/official_libero_rla_consistency/run_20261023_10/stove/ae_run/sidecars/C/clean/stats.json']
else:
 env.update(RLA_STEPS='8',RLA_QUERIES='16',RLA_DIM='64',RLA_NORM='center',LAMBDA_ACT='1',LAMBDA_RLA='0.22408536585365854',RLA_LOG_EVERY='1000',RLA_MASK='1',RLA_ACTION_TOKENS='1',RLA_INIT_FROM='',RLA_SHUFFLE_TARGETS='0')
 arm,entry=('D-inv03','stove_centering.training') if a.stage=='policy' else ('evaluation','rla_lr_budget_sweep.extend_evaluation')
 if a.stage=='policy':assert json.loads((a.out/'target_readiness.json').read_text())['status']=='PASS'
 cmd=['/data/lyc/projects/vla-adapter-ref/.venv-cpu/bin/python','-u',str(root/'rla_lr_budget_sweep/bootstrap.py'),'--out',str(a.out),'--arm',arm,'--entry',entry,'--attempt','formal']
before=subprocess.check_output(['nvidia-smi','-i',env['RLA_GPU_UUID'],'-q','-d','ECC'],text=True)
assert int(subprocess.check_output(['nvidia-smi','-i',env['RLA_GPU_UUID'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))>=26624
with (a.out/(a.stage+'_ecc_before.txt')).open('x') as f:f.write(before)
t=time.time()
with (a.out/(a.stage+'.log')).open('x') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
with (a.out/(a.stage+'_process.json')).open('x') as f:json.dump(dict(command=cmd,returncode=r.returncode,seconds=time.time()-t),f,indent=2)
raise SystemExit(r.returncode)
