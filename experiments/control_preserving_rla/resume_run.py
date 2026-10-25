"""Continue the unfinished budget only after explicit resource preflight."""
import argparse
import json
import os
from pathlib import Path
import subprocess

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();o=a.out;src=Path(__file__).parent
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',POLICY_RESUME=str(o/'E-D-keep/formal/step0300.pt'))
def run(stage,attempt):subprocess.run(['/usr/bin/python3',str(src/'launch.py'),'--out',str(o),'--stage',stage,'--attempt',attempt,'--gpu','1'],env=env,check=True)
run('resume-precheck','resume_gpu1');assert json.loads((o/'resume_precheck_acceptance.json').read_text())['optimizer_updates']==0
run('E-D-keep','resume_gpu1');assert json.loads((o/'E-D-keep/resume_gpu1/training_acceptance.json').read_text())['successful_updates']==600
registry=json.loads((o/'policy_registry.json').read_text());registry['E-D-keep']=str(o/'E-D-keep/resume_gpu1/step0600.pt')
with (o/'policy_registry_resumed.json').open('x') as f:json.dump(registry,f,indent=2)
subprocess.run(['/usr/bin/python3',str(src/'merge_recovery.py'),'--out',str(o)],env=env,check=True)
env.pop('POLICY_RESUME');env['RLA_POLICY_REGISTRY']='policy_registry_resumed.json'
run('evaluate','formal')
print('RECOVERED_EXPERIMENT_COMPLETE',flush=True)
