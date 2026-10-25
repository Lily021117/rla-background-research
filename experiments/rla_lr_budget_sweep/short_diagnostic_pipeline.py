"""Read only completed checkpoints in a separate process; launcher locks avoid repeats."""
import argparse,json,time,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();plan=json.loads((a.out/'run_plan.json').read_text());src=Path(__file__).parent
for arm in plan['arms']:
 if arm!='C_m0p1':
  while not (a.out/arm/'formal/training_acceptance.json').exists():
   f=a.out/(arm+'_formal_process.json')
   if f.exists():assert json.loads(f.read_text())['returncode']==0,'training failure; no further diagnostics'
   time.sleep(10)
 subprocess.run([sys.executable,str(src/'short_launch.py'),'--out',str(a.out),'--stage',arm,'--gpu','1','--attempt','diagnostics'],check=True)
 print('PIPELINE_DIAGNOSTIC_COMPLETE',arm,flush=True)
