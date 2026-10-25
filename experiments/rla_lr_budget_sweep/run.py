"""Run all frozen candidates, fixed checkpoint rollouts, then development selection."""
import argparse,json,subprocess,sys,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--gpu',default='1',choices=['0','1']);a=p.parse_args();src=Path(__file__).parent
plan=json.loads((a.out/'run_plan.json').read_text())
for stage in ['gate']+list(plan['arms'])+['evaluate']:
 if stage=='gate' and (a.out/'gate_acceptance.json').exists():
  assert json.loads((a.out/'gate_acceptance.json').read_text())['status']=='PASS';continue
 subprocess.run([sys.executable,str(src/'launch.py'),'--out',str(a.out),'--stage',stage,'--gpu',a.gpu],check=True)
 print('STAGE_COMPLETE',stage,flush=True)
env=os.environ.copy();cache='/data/lyc/tmp/rla_background_invariance/cache';env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',HF_HOME=cache+'/huggingface',TORCH_HOME=cache+'/torch',XDG_CACHE_HOME=cache+'/xdg',TMPDIR=cache+'/tmp',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
subprocess.run(['/data/lyc/projects/vla-adapter-ref/.venv-cpu/bin/python',str(src/'analyze.py'),'--out',str(a.out)],env=env,check=True)
print('SWEEP_COMPLETE',flush=True)
