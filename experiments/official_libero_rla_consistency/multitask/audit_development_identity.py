"""Check frozen allowed states without changing or selecting any state."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();rows=[]
for task in ('drawer','stove'):
 root=a.run/task;ae=root/('ae_run_recovery1' if task=='drawer' else 'ae_run');protocol=json.loads((ae/'development_protocol/evaluation_protocol.json').read_text());data=json.loads((root/'frozen_dataset_manifest.json').read_text());known=[]
 for e in data['episodes']:
  state=np.load(Path(e['root'])/'trajectory_labels.npz')['states'];known.extend((e['episode'],e['split'],x[1:]) for x in state)
 hashes=set();records=[]
 for r in protocol['new_development_states']:
  state=np.load(r['state_file']);digest=hashlib.sha256(state[1:].tobytes()).hexdigest();assert digest not in hashes;hashes.add(digest);matches=[dict(episode=e,split=s) for e,s,x in known if x.shape==state[1:].shape and np.array_equal(x,state[1:])]
  records.append(dict(seed=r['seed'],physical_state_sha256_excluding_clock=digest,exact_matches_allowed_source_excluding_clock=matches,palette=r['unseen_palette_id']))
 assert len(records)==20
 rows.append(dict(task=task,unique_physical_states_excluding_clock=20,states=records,sealed_contents_read=False,independence_limit='New legal reset samples are not independent final test. Absence of exact equality does not establish distributional independence or rule out all near-overlap; no protected arrays were read.'))
with (a.run/'development_identity_audit.json').open('x') as f:json.dump(rows,f,indent=2)
print(json.dumps([{k:v for k,v in r.items() if k!='states'} for r in rows],indent=2))
