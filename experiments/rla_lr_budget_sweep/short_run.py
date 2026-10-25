"""Seven bounded training continuations; eight frozen offline diagnostics; no rollout."""
import argparse,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();src=Path(__file__).parent;plan=json.loads((a.out/'run_plan.json').read_text())
for arm in plan['arms']:
 if arm=='C_m0p1':continue
 subprocess.run([sys.executable,str(src/'short_launch.py'),'--out',str(a.out),'--stage',arm,'--gpu','1'],check=True)
 print('SHORT_TRAIN_COMPLETE',arm,flush=True)
for arm in plan['arms']:
 subprocess.run([sys.executable,str(src/'short_launch.py'),'--out',str(a.out),'--stage',arm,'--gpu','1','--attempt','diagnostics'],check=True)
 print('SHORT_DIAGNOSTICS_COMPLETE',arm,flush=True)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',TMPDIR='/data/lyc/tmp/rla_background_invariance/cache/tmp',MPLCONFIGDIR='/data/lyc/tmp/rla_background_invariance/cache/matplotlib',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
subprocess.run(['/data/lyc/projects/vla-adapter-ref/.venv-cpu/bin/python',str(src/'short_analyze.py'),'--out',str(a.out)],env=env,check=True)
print('SHORT_EXPERIMENT_COMPLETE',flush=True)
