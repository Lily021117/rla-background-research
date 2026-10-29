"""Own only two bounded AE workers on the authorized GPU; never launch policies."""
import os,subprocess,time,json
from pathlib import Path
R=Path('/data/lyc/projects/rla-wm/experiments/rla_ae_from_scratch/run_20261029_01')
C=R/'code';HERE=C/'experiments/rla_ae_from_scratch'
env=os.environ.copy()
for key,name in [('HF_HOME','huggingface'),('TORCH_HOME','torch'),('XDG_CACHE_HOME','xdg'),('TMPDIR','tmp'),('MPLCONFIGDIR','matplotlib')]:
 p=R/'cache'/name;p.mkdir(parents=True,exist_ok=True);env[key]=str(p)
env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
with (R/'formal_execution.json').open('x') as f:json.dump(dict(mode='two isolated concurrent processes on same authorized GPU0',no_configuration_change=True,policy_updates=0,updates_each=2000,stop_on_error=True,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=C,text=True).strip()),f,indent=2)
workers={};handles=[]
for arm in ['C','D']:
 f=(R/(arm+'_launcher.log')).open('x');handles.append(f)
 workers[arm]=subprocess.Popen(['python',str(HERE/'launch.py'),arm],cwd=C,env=env,stdout=f,stderr=subprocess.STDOUT)
 time.sleep(15) # GPU gate is rechecked by each worker; same inputs and RNG, not shared state
failed=False
while any(p.poll() is None for p in workers.values()):
 if not failed and any(p.poll() not in [None,0] for p in workers.values()):
  failed=True
  with (R/'STOP_FORMAL.json').open('x') as f:json.dump(dict(reason='one formal worker failed; save sibling at next successful-update boundary',statuses={a:p.poll() for a,p in workers.items()}),f)
 time.sleep(15)
for f in handles:f.close()
codes={a:p.returncode for a,p in workers.items()}
with (R/'formal_process_completion.json').open('x') as f:json.dump(codes,f)
if any(codes.values()):raise SystemExit('Formal training stopped; preserve evidence, no auto retry')
with (R/'summarize.log').open('x') as f:
 p=subprocess.run(['python',str(HERE/'summarize.py')],cwd=C,env=env,stdout=f,stderr=subprocess.STDOUT)
if p.returncode:raise SystemExit('Summarization failed; training remains complete')
with (R/'resource_release.json').open('x') as f:json.dump(dict(own_workers_exited=True,worker_returncodes=codes,policy_updates=0,rollouts=0,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),f,indent=2)
