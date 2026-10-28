"""Frozen four-group run. Stops on failed gate, resources or process; no silent retry."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from labels import read,save,sha
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1];PROJECT=Path(os.environ['RLA_PROJECT_ROOT']);CROOT=PROJECT/'experiments/rla_sampling_stability/run_20261027_01';GAP=PROJECT/'experiments/rla_appearance_gap_eval/run_20261028_01';GPU=os.environ['RLA_GPU_UUID']
ORDERS=[20261101,20261102];ARMS=['Action-only-match','Geometry-target']
def prepare(out):
 labels=read(out/'labels/manifest.json');assert labels['status']=='PASS'
 protocol=dict(base_commit='8cb44ab99194c812837ee862f2dc33fe5c0f1de2',orders=ORDERS,methods=ARMS,updates_per_group=600,total_updates=2400,new_valid_rollouts=400,AE_updates=0,primary='Geometry-target minus Action-only-match',secondary='Geometry-target minus frozen C-reference',checkpoints=[0,1,100,300,600],validation=[0,100,300,600],lambda_aux=.22408536585365854,auxiliary_geometry='FP32 mean absolute error of first12 final independent output coordinates, valid8 horizons; all unobserved outputs excluded, no RLA loss',geometry_floor_m=.001,teacher=False,keep=False,tail_padding=False,trial_retry='stop on infrastructure error; no task-failure retry',seed='historical model RNG; independent frozen sampling orders only',GPU=GPU,geometry_vs_C_caveat='supervision meaning, dimensions and normalization differ; not a single representation-factor causal contrast',decision='retain negative results; no hyperparameter extension',sealed_read=False)
 save(out/'protocol.json',protocol);protected=read(out/'labels/protected_inputs.json')
 for order in ORDERS:
  src=CROOT/f'seed_{order}';root=out/f'seed_{order}';root.mkdir();plan=read(src/'run_plan.json');step0=src/'C/formal/step0000.pt'
  plan.update(arms=ARMS,common_step0=str(step0),common_step0_sha256=sha(step0),geometry_labels=str(out/'labels'),geometry_gate=str(root/'preflight/formal/preflight_acceptance.json'),paired_C_sample_trace=str(src/'C/formal/actual_sample_ids.jsonl'),experiment='task_geometry_supervision',target_source_by_arm={a:'C' for a in ARMS},closed_loop_trials=100)
  save(root/'run_plan.json',plan);save(root/'normalization.json',read(src/'normalization.json'))
  for p in [step0,src/'run_plan.json',src/'normalization.json',Path(plan['paired_C_sample_trace']),Path(plan['inputs']['B20_sampling_order'])]:protected[str(p)]=sha(p)
 plan=read(out/f'seed_{ORDERS[0]}/run_plan.json');save(out/'run_plan.json',plan);save(out/'normalization.json',read(out/f'seed_{ORDERS[0]}/normalization.json'))
 ep=read(GAP/'evaluation_protocol.json');models={f'{a}_order{o}_s600':str(out/f'seed_{o}'/a/'formal/step0600.pt') for o in ORDERS for a in ARMS}
 ep.update(models=list(models),conditions=['clean']+[f'new_background_{i}' for i in range(1,5)],trials=400,geometry_target_evaluation=True,order='state,appearance,order/method',checkpoint_selection='all final600 only',Clean='new for four new models; C200 historical results reused',C_background='reuse160 plus40Clean')
 save(out/'evaluation_protocol.json',ep);save(out/'policy_registry.json',models);save(out/'protected_inputs.json',protected)
 refs=[r for r in read(GAP/'reused_results.json') if r['method']=='C'];assert len(refs)==200;save(out/'C_reference_results.json',refs)
def launch(out,order,arm,entry):
 root=out if order is None else out/f'seed_{order}'
 free=int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True));assert free>=26624,free
 assert int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True))==0
 env=os.environ.copy();cache=PROJECT/'experiments/rla_task_geometry_target/cache'
 for k,n in {'HF_HOME':'huggingface','HUGGINGFACE_HUB_CACHE':'huggingface/hub','TORCH_HOME':'torch','XDG_CACHE_HOME':'xdg','TMPDIR':'tmp','CUDA_CACHE_PATH':'cuda','TRITON_CACHE_DIR':'triton','MPLCONFIGDIR':'matplotlib','HF_DATASETS_CACHE':'huggingface/datasets'}.items():
  # Reuse model caches rather than download weights into a new cache.
  p=PROJECT.parents[1]/'tmp/rla_background_invariance/cache'/n;p.mkdir(parents=True,exist_ok=True);env[k]=str(p)
 env.update(RLA_GPU_UUID=GPU,CUDA_VISIBLE_DEVICES=GPU,RLA_MIN_FREE_GIB='26',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',LP_NUM_THREADS='2',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',HF_HUB_OFFLINE='1',LIBERO_CONFIG_PATH=str(PROJECT.parent/'LIBERO-plus/.libero'),RLA_STEPS='8',RLA_QUERIES='16',RLA_DIM='64',RLA_NORM='center',LAMBDA_ACT='1',LAMBDA_RLA='0.22408536585365854',RLA_LOG_EVERY='1000',RLA_MASK='1',RLA_ACTION_TOKENS='1',RLA_INIT_FROM='',RLA_SHUFFLE_TARGETS='0')
 env['PYTHONPATH']=str(PROJECT.parent/'LIBERO-plus')+':'+str(HERE.parent)+':'+str(HERE.parent/'official_libero_rla_consistency')+':'+env.get('PYTHONPATH','')
 cmd=[sys.executable,'-u',str(HERE.parent/'rla_lr_budget_sweep/bootstrap.py'),'--out',str(root),'--arm',arm,'--entry',entry,'--attempt','formal'];start=time.time()
 with (root/(arm+'.log')).open('x') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,cwd=REPO)
 save(root/(arm+'_process.json'),dict(returncode=r.returncode,seconds=time.time()-start,command=cmd,GPU=GPU));assert r.returncode==0,(order,arm,r.returncode)
def run(out):
 for order in ORDERS:
  launch(out,order,'preflight','rla_task_geometry_target.training')
  for arm in ARMS:launch(out,order,arm,'rla_task_geometry_target.training')
 launch(out,None,'evaluation','rla_lr_budget_sweep.extend_evaluation')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['prepare','run'],required=True);a=p.parse_args();(prepare if a.mode=='prepare' else run)(a.out)
