"""Finish only this authorized experiment; stop on any stage failure."""
import argparse,json,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();here=Path(__file__).resolve().parent
for arm in ('old','center'):
 path=a.out/f'train-{arm}_isolated_smoke_process.json'
 while not path.exists():time.sleep(10)
 assert json.loads(path.read_text())['returncode']==0,path
for stage in ('diagnostics','evaluate'):
 subprocess.run(['/data/lyc/projects/rla-wm/.venv-sidecar/bin/python','-B',str(here/'launch.py'),'--out',str(a.out),'--stage',stage],check=True)
subprocess.run(['/data/lyc/projects/rla-wm/.venv-sidecar/bin/python','-B',str(here/'finalize.py'),'--out',str(a.out)],check=True)
print('BOUNDED_EXPERIMENT_COMPLETE',flush=True)
