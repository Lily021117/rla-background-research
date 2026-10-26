"""Bounded frozen stages after AE completion; stop on every failed check."""
import argparse,json,subprocess,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();here=Path(__file__).parent
while not (a.out/'ae_process.json').exists():time.sleep(10)
assert json.loads((a.out/'ae_process.json').read_text())['returncode']==0,'AE failed; no later stage authorized to bypass it'
assert json.loads((a.out/'ae_run/AE_D_inv03/acceptance.json').read_text())['successful_updates']==600
def run(cmd):
 print('START',cmd,flush=True);r=subprocess.run(cmd);assert r.returncode==0,cmd
for stage in ['representation','sidecar','rawcheck']:
 run(['python',str(here/'launch.py'),'--out',str(a.out),'--stage',stage])
env=os.environ.copy();cache=Path('/data/lyc/tmp/rla_background_invariance/cache')
env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',HF_HOME=str(cache/'huggingface'),TORCH_HOME=str(cache/'torch'),XDG_CACHE_HOME=str(cache/'xdg'),TMPDIR=str(cache/'tmp'),MPLCONFIGDIR=str(cache/'matplotlib'))
with (a.out/'target_reader.log').open('x') as f:r=subprocess.run(['/data/lyc/projects/vla-adapter-ref/.venv-cpu/bin/python',str(here/'target_ready.py'),'--out',str(a.out)],env=env,stdout=f,stderr=subprocess.STDOUT)
assert r.returncode==0,'new target audit failed'
for stage in ['policy','evaluate']:run(['python',str(here/'launch.py'),'--out',str(a.out),'--stage',stage])
with (a.out/'analysis.log').open('x') as f:r=subprocess.run(['/data/lyc/projects/vla-adapter-ref/.venv-cpu/bin/python',str(here/'analyze.py'),'--out',str(a.out)],env=env,stdout=f,stderr=subprocess.STDOUT)
assert r.returncode==0,'analysis failed; preserve outputs and inspect'
print('BOUNDED_PIPELINE_COMPLETE_NO_MORE_TRAINING',flush=True)
