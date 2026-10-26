"""One bounded stage on authorized GPU0; no hidden retries or parameter changes."""
import argparse,json,os,subprocess,time,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--stage',choices=['gate','C','D-center','evaluate'],required=True);p.add_argument('--managed-worker',action='store_true');p.add_argument('--resume-step100',action='store_true');a=p.parse_args();root=Path(__file__).resolve().parents[1];run=a.out/f'seed_{a.seed}';protocol=json.loads((a.out/'replication_protocol.json').read_text());assert a.seed in protocol['seeds']
if (run/'externally_managed.json').exists() and a.stage in ('C','D-center') and not a.managed_worker:
 while True:
  process=run/(a.stage+'_process.json')
  if process.exists():
   result=json.loads(process.read_text())
   if result['returncode']==0:assert json.loads((run/a.stage/'formal/training_acceptance.json').read_text())['successful_updates']==600
   sys.exit(result['returncode'])
  print('WAIT_FOR_AUTHORIZED_WORKER',a.seed,a.stage,flush=True);time.sleep(30)
env=os.environ.copy();cache=Path('/data/lyc/tmp/rla_background_invariance/cache')
for k,v in {'HF_HOME':cache/'huggingface','HUGGINGFACE_HUB_CACHE':cache/'huggingface/hub','TORCH_HOME':cache/'torch','XDG_CACHE_HOME':cache/'xdg','TMPDIR':cache/'tmp','CUDA_CACHE_PATH':cache/'cuda','TRITON_CACHE_DIR':cache/'triton','MPLCONFIGDIR':cache/'matplotlib','HF_DATASETS_CACHE':cache/'huggingface/datasets','NUMBA_CACHE_DIR':cache/'numba'}.items():v.mkdir(parents=True,exist_ok=True);env[k]=str(v)
env.update(RLA_GPU_UUID='GPU-205a3f5a-d91b-6385-c588-4e4cf3505f7e',CUDA_VISIBLE_DEVICES='GPU-205a3f5a-d91b-6385-c588-4e4cf3505f7e',RLA_MIN_FREE_GIB='26',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',LP_NUM_THREADS='2',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',HF_HUB_OFFLINE='1',LIBERO_CONFIG_PATH='/data/lyc/projects/LIBERO-plus/.libero',RLA_STEPS='8',RLA_QUERIES='16',RLA_DIM='64',RLA_NORM='center',LAMBDA_ACT='1',LAMBDA_RLA='0.22408536585365854',RLA_LOG_EVERY='1000',RLA_MASK='1',RLA_ACTION_TOKENS='1',RLA_INIT_FROM='',RLA_SHUFFLE_TARGETS='0')
env['PYTHONPATH']='/data/lyc/projects/LIBERO-plus:'+str(root)+':'+str(root/'official_libero_rla_consistency')+':'+env.get('PYTHONPATH','')
if (run/'resource_assignment.json').exists():
 assignment=json.loads((run/'resource_assignment.json').read_text());gpu=assignment['GPU_UUID'];assert gpu in ('GPU-205a3f5a-d91b-6385-c588-4e4cf3505f7e','GPU-37d3f7d4-309e-2bb8-2d51-292dbbcce419');env.update(RLA_GPU_UUID=gpu,CUDA_VISIBLE_DEVICES=gpu)
if (run/'policy_registry_migration.json').exists():env['RLA_POLICY_REGISTRY']='policy_registry_migration.json'
if a.stage=='gate':env.update(RLA_BOOTSTRAP_PLAN='planned_run_plan.json',RLA_BOOTSTRAP_SEED=str(a.seed),RLA_FULL_SEED_RECIPE='1');arm,entry,attempt='initialization','rla_policy_seed_replication.gate','zero_update'
elif a.stage=='evaluate':arm,entry,attempt='evaluation','rla_lr_budget_sweep.extend_evaluation','formal'
else:
 assert json.loads((run/'initialization_gate.json').read_text())['status']=='PASS';arm,entry,attempt=a.stage,'stove_centering.training','formal'
 assert json.loads((a.out/'frozen_target_audit.json').read_text())['status']=='PASS'
label=a.stage
if a.resume_step100:
 assert a.seed==20261029 and a.stage=='D-center' and json.loads((run/'migration_boundary_audit.json').read_text())['global_step']==100
 env['POLICY_RESUME']=str(run/'D-center/formal/step0100.pt');attempt='resume_step0100';label='D-center_resume_step0100'
before=subprocess.check_output(['nvidia-smi','-i',env['RLA_GPU_UUID'],'-q','-d','ECC'],text=True);assert int(subprocess.check_output(['nvidia-smi','-i',env['RLA_GPU_UUID'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))>=26624
with (run/(label+'_ecc_before.txt')).open('x') as f:f.write(before)
cmd=['/data/lyc/projects/vla-adapter-ref/.venv-cpu/bin/python','-u',str(root/'rla_lr_budget_sweep/bootstrap.py'),'--out',str(run),'--arm',arm,'--entry',entry,'--attempt',attempt];t=time.time()
with (run/(label+'.log')).open('x') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
with (run/(label+'_process.json')).open('x') as f:json.dump(dict(command=cmd,returncode=r.returncode,seconds=time.time()-t),f,indent=2)
raise SystemExit(r.returncode)
