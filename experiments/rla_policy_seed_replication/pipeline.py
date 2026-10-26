"""Four authorized 600-update jobs and two 120-trial evaluations; stop on fault."""
import argparse,json,subprocess,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();here=Path(__file__).parent;protocol=json.loads((a.out/'replication_protocol.json').read_text())
def run(seed,stage):
 cmd=['python',str(here/'launch.py'),'--out',str(a.out),'--seed',str(seed),'--stage',stage];print('START',seed,stage,flush=True);r=subprocess.run(cmd);assert r.returncode==0,(seed,stage,'preserve failed attempt; no silent retry')
for seed in protocol['seeds']:
 run(seed,'gate')
 for arm in protocol['methods']:run(seed,arm)
for seed in protocol['seeds']:run(seed,'evaluate')
print('ALL_FOUR_TRAININGS_AND_240_TRIALS_COMPLETE',flush=True)
env=os.environ.copy();cache=Path('/data/lyc/tmp/rla_background_invariance/cache');env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',HF_HOME=str(cache/'huggingface'),TORCH_HOME=str(cache/'torch'),XDG_CACHE_HOME=str(cache/'xdg'),TMPDIR=str(cache/'tmp'),MPLCONFIGDIR=str(cache/'matplotlib'))
with (a.out/'analysis.log').open('x') as f:r=subprocess.run(['/data/lyc/projects/vla-adapter-ref/.venv-cpu/bin/python',str(here/'analyze.py'),'--out',str(a.out)],env=env,stdout=f,stderr=subprocess.STDOUT)
assert r.returncode==0,'analysis failed; preserve partial artifacts'
print('BOUNDED_REPLICATION_COMPLETE_NO_AUTO_EXTENSION',flush=True)
