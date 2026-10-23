"""CPU-only strict verification of the released LIBERO AE; no optimizer."""
import argparse, hashlib, json, sys
from pathlib import Path
import torch

p=argparse.ArgumentParser();p.add_argument('--official',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert not a.output.exists()
sys.path.insert(0,str(a.official))
from utils.misc import load_config
from src import models
cfg=load_config(str(a.run/'config.yaml'))
torch.set_num_threads(4)
rows=[]
for name in ('encoder','decoder'):
    spec=cfg['models'][name]
    m=getattr(models,spec['name'])(**spec['args']).eval().requires_grad_(False)
    path=a.run/'ckpts'/f'{name}_step0090000.pt'
    sd=torch.load(path,map_location='cpu',weights_only=True)
    if 'model_state_dict' in sd:sd=sd['model_state_dict']
    result=m.load_state_dict(sd,strict=True)
    rows.append(dict(name=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,parameters=sum(x.numel() for x in m.parameters()),strict=True,missing=list(result.missing_keys),unexpected=list(result.unexpected_keys),config=spec,finite=all(torch.isfinite(v).all().item() for v in sd.values() if torch.is_floating_point(v))))
    del m,sd
a.output.write_text(json.dumps(dict(status='PASS',released_step=90000,paper_step40000=False,models=rows,optimizer_updates=0,backward=0,GPU_used=False),indent=2))
print(a.output)
