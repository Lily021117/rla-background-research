"""Two bounded continuations followed by exactly120 valid paired trials."""
import argparse,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();src=Path(__file__).parent
for stage in ('C_m0p1','D-center_m0p1','evaluate'):
 subprocess.run([sys.executable,str(src/'extend_launch.py'),'--out',str(a.out),'--stage',stage,'--gpu','1'],check=True)
 print('CONTINUATION_STAGE_COMPLETE',stage,flush=True)
print('CONTINUATION_TRAIN_AND_EVAL_COMPLETE',flush=True)
