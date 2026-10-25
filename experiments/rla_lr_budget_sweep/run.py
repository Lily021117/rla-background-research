"""Run all frozen candidates, fixed checkpoint rollouts, then development selection."""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--gpu',default='1',choices=['0','1']);a=p.parse_args();src=Path(__file__).parent
plan=json.loads((a.out/'run_plan.json').read_text())
for stage in ['gate']+list(plan['arms'])+['evaluate']:
 if stage=='gate' and (a.out/'gate_acceptance.json').exists():
  assert json.loads((a.out/'gate_acceptance.json').read_text())['status']=='PASS';continue
 subprocess.run([sys.executable,str(src/'launch.py'),'--out',str(a.out),'--stage',stage,'--gpu',a.gpu],check=True)
 print('STAGE_COMPLETE',stage,flush=True)
subprocess.run([sys.executable,str(src/'analyze.py'),'--out',str(a.out)],check=True)
print('SWEEP_COMPLETE',flush=True)
