"""Explicitly authorized second-seed worker; paired arms serial, no retry."""
import argparse,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seed',required=True);a=p.parse_args()
for stage in ['C','D-center']:
 r=subprocess.run(['python',str(Path(__file__).with_name('launch.py')),'--out',a.out,'--seed',a.seed,'--stage',stage,'--managed-worker'])
 if r.returncode:raise SystemExit(r.returncode)
print('AUTHORIZED_SECOND_SEED_TRAINING_COMPLETE',flush=True)
