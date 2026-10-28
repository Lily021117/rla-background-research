"""Bounded serial GPU0 pipeline; fail closed, never alter a failed configuration."""
import argparse,fcntl,json,os,subprocess,time,datetime
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--extraction-attempt',default='gpu0_queue_retry2');a=p.parse_args();root=Path(__file__).resolve().parents[1];project=Path('/data/lyc/projects/rla-wm');out=a.run/'pipeline';out.mkdir(exist_ok=False)
 lock=(out/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 env=os.environ.copy();uuid='GPU-205a3f5a-d91b-6385-c588-4e4cf3505f7e';cache=a.run/'cache'
 for key,suffix in {'HF_HOME':'huggingface','HF_HUB_CACHE':'huggingface/hub','HUGGINGFACE_HUB_CACHE':'huggingface/hub','TORCH_HOME':'torch','XDG_CACHE_HOME':'xdg','TMPDIR':'tmp','CUDA_CACHE_PATH':'cuda','TRITON_CACHE_DIR':'triton','MPLCONFIGDIR':'matplotlib','NUMBA_CACHE_DIR':'numba'}.items():
  path=cache/suffix;path.mkdir(parents=True,exist_ok=True);env[key]=str(path)
 env.update(RLA_GPU_UUID=uuid,CUDA_VISIBLE_DEVICES=uuid,RLA_MIN_FREE_GIB='26',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',LP_NUM_THREADS='2',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',HF_HUB_OFFLINE='1',LIBERO_CONFIG_PATH='/data/lyc/projects/LIBERO-plus/.libero',RLA_STEPS='8',RLA_QUERIES='16',RLA_DIM='64',RLA_NORM='center',LAMBDA_ACT='1',LAMBDA_RLA='0.22408536585365854',RLA_LOG_EVERY='1000',RLA_MASK='1',RLA_ACTION_TOKENS='1',RLA_INIT_FROM='',RLA_SHUFFLE_TARGETS='0',NEW_TASK_RUN=str(a.run))
 env['PYTHONPATH']='/data/lyc/projects/LIBERO-plus:'+str(root)+':'+str(root/'official_libero_rla_consistency')
 py='/data/lyc/projects/vla-adapter-ref/.venv-cpu/bin/python'
 def record(state,**kw):
  row=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),state=state,**kw)
  with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  temp=out/'status.tmp';temp.write_text(json.dumps(row,indent=2));temp.replace(out/'status.json');print(json.dumps(row),flush=True)
 baseline=None
 def gate(stage):
  nonlocal baseline
  stable=0
  while stable<3:
   free=int(subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True));ecc=subprocess.check_output(['nvidia-smi','-i',uuid,'-q','-d','ECC'],text=True)
   counters=[s.strip() for s in ecc.splitlines() if ':' in s and any(k in s for k in ('Correctable','Uncorrectable','SRAM SM','Threshold Exceeded'))]
   assert all(int(s.rsplit(':',1)[1])==0 for s in counters if 'Uncorrectable' in s)
   assert all(s.rsplit(':',1)[1].strip()=='No' for s in counters if 'Threshold Exceeded' in s)
   if baseline is None:baseline=counters
   assert counters==baseline,'ECC increment'
   stable=stable+1 if free>=26624 else 0;record('RESOURCE_WAIT',stage=stage,free_MiB=free,stable_checks=stable)
   if stable<3:time.sleep(30)
 def launch(seed,arm,entry):
  stage=f'{seed}_{arm}';gate(stage);command=[py,'-u',str(root/'rla_lr_budget_sweep/bootstrap.py'),'--out',str(a.run/f'seed_{seed}'),'--arm',arm,'--entry',entry,'--attempt','formal'];record('RUNNING',stage=stage,command=command);start=time.time()
  with (out/f'{stage}.log').open('x') as f:proc=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT)
  record('STAGE_FINISHED',stage=stage,returncode=proc.returncode,seconds=time.time()-start)
  assert proc.returncode==0,f'{stage} failed; preserve state, no silent retry'
 try:
  while True:
   status=a.run/a.extraction_attempt/'status.json';s=json.loads(status.read_text())
   if s['state'].startswith('BLOCKED'):raise RuntimeError('Frozen encoder stage blocked: '+str(status))
   if s['state']=='PREPARATION_COMPLETE':break
   record('WAITING_EXTRACTION',source=str(status),upstream_state=s['state']);time.sleep(30)
  assert json.loads((a.run/'frozen_sidecars/sidecar_alignment_audit.json').read_text())['status']=='PASS'
  for arm in ['preflight-standard','preflight-rla']:launch(20261101,arm,'rla_new_task_baseline.training')
  for seed in [20261101,20261102]:
   for arm in ['BC-standard','BC-matched','C-frozen-RLA']:launch(seed,arm,'rla_new_task_baseline.training')
  launch(20261101,'evaluation','rla_new_task_baseline.evaluation')
  proc=subprocess.run([py,str(root/'rla_new_task_baseline/summarize.py'),'--run',str(a.run)],env=env)
  assert proc.returncode==0
  record('COMPLETED',successful_policy_updates=3600,AE_updates=0,heldout_read=False,next='Stop; report/publication review only')
 except Exception as exc:
  record('BLOCKED',error=repr(exc));raise

if __name__=='__main__':main()
