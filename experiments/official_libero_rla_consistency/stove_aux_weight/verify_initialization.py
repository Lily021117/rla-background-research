import argparse,json,hashlib
from pathlib import Path
import torch,numpy as np
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
def digest(v):
 h=hashlib.sha256()
 for module in sorted(v):
  for name in sorted(v[module]):h.update((module+'.'+name).encode());h.update(v[module][name].contiguous().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
paths=[a.out/x/'isolated_smoke/step0000.pt' for x in ('C-low','D-center-low')]
cs=[torch.load(p,map_location='cpu',weights_only=False) for p in paths];initial=torch.load(json.loads((a.out/'run_plan.json').read_text())['common_step0'],map_location='cpu',weights_only=False)
hashes=[digest(c['parameters']) for c in cs+[initial]];assert len(set(hashes))==1
for c in cs:
 assert c['global_step']==0 and c['next_physical_sample']==0 and not c['optimizer']['state']
 assert torch.equal(c['torch_rng'],initial['torch_rng'])
 assert all(torch.equal(x,y) for x,y in zip(c['cuda_rng'],initial['cuda_rng']))
 assert c['python_rng']==initial['python_rng']
 assert c['numpy_rng'][0]==initial['numpy_rng'][0] and np.array_equal(c['numpy_rng'][1],initial['numpy_rng'][1]) and c['numpy_rng'][2:]==initial['numpy_rng'][2:]
with (a.out/'initialization_acceptance.json').open('x') as f:json.dump(dict(status='PASS',exact_parameters_and_RNG_equal=True,parameter_hashes=hashes,optimizer_initially_empty=True,source='accepted original common step0, not trained policy',global_steps=[0,0]),f,indent=2)
print('EXACT_COMMON_INITIALIZATION_PASS')
