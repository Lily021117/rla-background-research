"""Official Prismatic load, including its separate frozen vision pretraining."""
import argparse,sys,json,hashlib
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();o=a.out
sys.path.insert(0,str(o/'vla-official'));torch.set_num_threads(4)
from prismatic.models import load
root=o/'prism-qwen25-extra-dinosiglip-224px-0_5b';pt=root/'checkpoints/step-020792-epoch-01-loss=0.5268.pt'
h=hashlib.sha256(pt.read_bytes()).hexdigest();assert h=='d86525b8d21865b2e83d4a403ee9d1b641579bb41259111d7e3cc7f7ed46564a'
# Official inference loading builds the LLM from config and strictly restores the
# Prismatic checkpoint; avoids requiring FlashAttention for a CPU-only load.
m=load(str(root),hf_token=None,load_for_training=False).eval().requires_grad_(False)
assert not (o/'resolved_prismatic_initialization.pt').exists()
torch.save(m.state_dict(),o/'resolved_prismatic_initialization.pt')
with (o/'initialization_inventory.json').open('x') as f:json.dump(dict(status='PASS_OFFICIAL_LOAD',repo='Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b',revision='5cfd2cc6da00c06e0be7abf35d43ec792d8e9498',checkpoint_SHA256=h,scope='Prismatic llm/projector checkpoint plus officially specified frozen DINOv2+SigLIP pretrained vision; not Spatial-Pro',parameters=sum(p.numel() for p in m.parameters()),new_optimizer_updates=0),f,indent=2)
print('INITIALIZATION_PASS',flush=True)
