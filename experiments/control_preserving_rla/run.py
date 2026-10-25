"""Bounded authorized experiment; no tuning, fallback, or silent retry."""
import argparse
from pathlib import Path
import subprocess

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();src=Path(__file__).parent
for stage in ('teacher','gate','E-C','E-D','E-C-keep','E-D-keep','evaluate'):
 subprocess.run(['/usr/bin/python3',str(src/'launch.py'),'--out',str(a.out),'--stage',stage],check=True)
 print('STAGE_COMPLETE',stage,flush=True)
print('GPU_EXPERIMENT_COMPLETE',flush=True)
