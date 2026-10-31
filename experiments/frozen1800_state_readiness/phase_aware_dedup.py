"""Additional conservative pre-scoring dedup: ignore settling-related z/time.

This guard never replaces a state or continues on a duplicate. It stops admission.
All-object/fixture XY layouts within0.1mm are rejected even if rotation differs.
"""
import argparse
from pathlib import Path
import numpy as np
import mujoco
from .generated_states import DATA,OLD,BODIES,read,save

def main(root):
 protocol=read(root/'route_b_v3/state_protocol.json');known=[];ids=[]
 def xy(xml,s):
  m=mujoco.MjModel.from_xml_path(str(xml));d=mujoco.MjData(m);assert len(s)==1+m.nq+m.nv
  d.qpos[:]=s[1:1+m.nq];d.qvel[:]=s[1+m.nq:];mujoco.mj_forward(m,d)
  return d.xpos[[m.body(n).id for n in BODIES],:2].copy()
 ds=read(DATA/'frozen_dataset_manifest.json')
 for e in read(DATA/'dataset_selection.json')['episodes']:
  r=next(x for x in ds['episodes'] if x['episode']==e['episode']);s=np.load(Path(r['root'])/'trajectory_labels.npz')['states'][0]
  known.append(xy(Path(e['model_dir'])/'clean.xml',s));ids.append(e['source_demo'])
 for s in read(OLD/'development_protocol/evaluation_protocol.json')['new_development_states']:
  known.append(xy(Path(s['state_file']).parent/'clean.xml',np.load(s['state_file'])));ids.append('historical_reset_'+str(s['seed']))
 rows=[]
 for s in protocol['new_development_states']:
  path=Path(s['state_file']);post=np.load(path.parent/'physical_signature.npz')['position'][:,:2]
  raw=xy(path.parent/'clean.xml',np.load(path.parent/'raw_reset_state.npy'))
  distances=np.minimum(np.max(abs(np.stack(known)-post),axis=(1,2)),np.max(abs(np.stack(known)-raw),axis=(1,2)))
  i=int(np.argmin(distances));rows.append(dict(seed=s['seed'],closest_known_source=ids[i],minimum_max_coordinate_difference_m=float(distances[i]),duplicate=bool(distances[i]<=1e-4)))
  known.append(post);ids.append('new_'+str(s['seed']))
 save(root/'phase_aware_dedup.json',dict(rule='all7 named movable/fixture XY; max absolute coordinate difference<=0.0001m, both raw and settled new layouts; time,z,rotation and robot omitted conservatively',results=rows,pass_all=not any(x['duplicate'] for x in rows)))
 assert not any(x['duplicate'] for x in rows),'Duplicate layout detected: stop before scoring; no automatic substitution'
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
