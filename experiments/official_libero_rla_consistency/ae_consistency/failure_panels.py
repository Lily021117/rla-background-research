"""Uniform first-failure video evidence; no simulator, no new policy actions."""
import argparse,json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();rows=json.loads((a.run/'per_trial_results.json').read_text());assert len(rows)==180
out=a.run/'failure_evidence';out.mkdir(exist_ok=False);selected=[]
for model in ('B20','C','D'):
 for condition in ('clean','seen_background','unseen_warm_teal'):
  failed=sorted((r for r in rows if r['model']==model and r['condition']==condition and not r['success']),key=lambda r:r['seed'])
  if not failed:continue
  r=failed[0];reader=imageio.get_reader(r['video']);frames=[frame for frame in reader];reader.close();indices=sorted(set(int(round(x)) for x in np.linspace(0,len(frames)-1,5)))
  panel=Image.new('RGB',(448*len(indices),250),'white');draw=ImageDraw.Draw(panel)
  for j,i in enumerate(indices):panel.paste(Image.fromarray(frames[i]).resize((448,224)),(j*448,26));draw.text((j*448+4,5),f'{model} {condition} seed{r["seed"]} frame{i}',fill='black')
  path=out/f'{model}_{condition}_{r["seed"]}.png';panel.save(path);selected.append(dict(model=model,condition=condition,seed=r['seed'],stage_proxy=r['stage'],max_lift_m=r['max_lift_m'],min_ee_target_distance_m=r['min_ee_target_distance_m'],steps=r['steps'],video=r['video'],panel=str(path),selection='first failed seed within every model/condition; all trials included in statistics',evidence_limit='video is visual evidence, not independent contact or grasp truth'))
(out/'manifest.json').write_text(json.dumps(selected,indent=2));print(json.dumps(selected,indent=2))
