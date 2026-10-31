"""Bounded source audit; reads states, never policies or scoring results."""
import argparse, hashlib, json, os
from pathlib import Path
import h5py
import numpy as np
import torch

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();o=a.root;o.mkdir(exist_ok=True)
 project=Path(os.environ['RLA_PROJECT']);libero=Path(os.environ['LIBERO_ROOT']);task='pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate'
 raw=project/f'data/libero_original_minimal/libero_spatial/{task}_demo.hdf5'
 init=libero/'libero/libero/init_files/libero_spatial'/(task+'.pruned_init')
 manifest=project/'artifacts/rla_policy_ood_and_data_readiness/held_out_manifest.json'
 selected=[r['initial_state_index'] for r in json.loads(manifest.read_text())['held_out_clean_pairs'] if r['base_task']==task]
 access=dict(purpose='source mapping and dedup only; no policy loading or scoring',sealed_contents_read=True,
  reads=[dict(path=str(init),scope='torch container deserialization; selected20 indices inspected; container may deserialize all states',sha256=sha(init)),
         dict(path=str(raw),scope='all50 episode XML/config and first up to21 physical states; no actions/images or success scores',sha256=sha(raw))])
 (o/'sealed_content_access.json').write_text(json.dumps(access,indent=2))
 states=torch.load(init,map_location='cpu',weights_only=False)
 records=[];comparisons=[]
 with h5py.File(raw,'r') as f:
  group=f['data'];attrs={k:str(v) for k,v in group.attrs.items()}
  for name in sorted(group,key=lambda n:int(n.split('_')[-1])):
   e=group[name];s=np.asarray(e['states'][:21]);xml=str(e.attrs.get('model_file',''))
   records.append(dict(demo=name,states_shape=list(e['states'].shape),read_state_rows=len(s),xml_present=bool(xml),xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),first_state_time=float(s[0,0])))
   for i in selected:
    v=np.asarray(states[i]);same_shape=s.shape[1:]==v.shape
    # Ignore time; equality can support a candidate correspondence, not certify
    # model identity. Non-match is explicitly NOT evidence of independence.
    exact=np.flatnonzero(np.all(s[:,1:]==v[None,1:],axis=1)).tolist() if same_shape else []
    if exact:comparisons.append(dict(init_index=i,demo=name,source_rows=exact,meaning='state-vector match only; XML and settling provenance still required'))
 result=dict(status='ROUTE_A_NOT_CERTIFIED_AT_LEAST20',official_init_shape=list(np.shape(states)),selected_indices=selected,
  source_attributes=attrs,episodes=records,exact_ignoring_time_matches=comparisons,
  mapping='No index-to-demo assumption. No generation/pruning map found in bounded local LIBERO scripts/utils/benchmark search; an init state array alone does not identify episode XML/fixture placement. No unmatched state is certified independent.',
  prior_metadata_audit=dict(raw_registered=15,raw_prior_use=5,raw_remaining_at_most=10,official_registered=20,official_development_overlap_indices=[13,23,35]),
  decision='Route B, separately frozen generator; no old ambiguous state admitted',policy_forward=0,rollouts=0)
 (o/'source_content_audit.json').write_text(json.dumps(result,indent=2))
 print(json.dumps({k:v for k,v in result.items() if k not in ('episodes','source_attributes')},indent=2))
if __name__=='__main__':main()
