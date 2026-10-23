"""Sequential CPU rendering of the explicitly admitted development pool."""
import argparse,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
rows=json.loads((a.out/'episode_manifest.json').read_text())
for e in rows:
    out=a.out/'episodes'/e['episode'];out.mkdir(parents=True,exist_ok=False)
    with (out/'render.log').open('x') as log:
        result=subprocess.run([sys.executable,str(Path(__file__).with_name('recover_sequence.py')),'--project',str(a.project),'--out',str(out),'--episode',e['episode'],'--split',e['split']],stdout=log,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError('Rendering failed; preserved evidence: '+str(out))
    print('PASS',e['episode'],flush=True)
