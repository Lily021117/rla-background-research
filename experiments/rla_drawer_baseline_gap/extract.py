"""Use accepted image/ray extraction; replace both AE modules strictly before inference."""
import sys,inspect,importlib.util
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_drawer_baseline_gap.common import *
sys.path.insert(0,str(HERE.parent/'rla_new_task_baseline'))
from rla_new_task_baseline import extract_frozen as original
def load(project):
 # The historical helper constructs the official extractor (including DINO).
 # Its temporary step90000 module states are strictly replaced before any forward.
 ext,init=original.load(project)
 inventory=read(ROOT/'encoder_inventory.json');assert sha(inventory['path'])==inventory['sha256']
 ck=torch.load(inventory['path'],map_location='cpu',weights_only=False)
 assert ck['global_step']==2000
 for name in ['encoder','decoder']:
  ext.models[name].load_state_dict(ck['models'][name],strict=True)
  ext.models[name].eval().requires_grad_(False)
 from rla_ae_from_scratch.run import digest
 assert digest({k:ext.models[k] for k in ['encoder','decoder']})==read(AE/'C/completion.json')['final_parameter_sha256']
 save(ROOT/'encoder_loading_audit.json',dict(strict=True,final_checkpoint=inventory,global_step=2000,temporary_extractor_constructor_load='official90000 replaced BEFORE all inference; never used for new labels',all_final_parameters_match=True,encoder_decoder_DINO_frozen=True))
 del ck
 return ext,init
def main():
 source=inspect.getsource(original.main)
 source=replace(source,"expected={'encoder':'a22efe9311f1588675f19c83f5a2af3afb1dd221e59c8170fb64ced60037086c','decoder':'916952b1da241492db095d959ce850eed2f77693b380e9f8b9756b1c7612ff1d'}","expected={'encoder':read(ROOT/'encoder_inventory.json')['sha256'],'decoder':read(ROOT/'encoder_inventory.json')['sha256']}")
 source=replace(source,"for k,h in expected.items():assert sha(init/'ckpts'/f'{k}_step0090000.pt')==h;ext.models[k].eval().requires_grad_(False)","for k in expected:ext.models[k].eval().requires_grad_(False)")
 source=replace(source,"encoder=str(init/'ckpts/encoder_step0090000.pt')","encoder=read(ROOT/'encoder_inventory.json')['path']")
 source=replace(source,"step=90000","step=2000")
 source=replace(source,"task_pretraining_exclusion='not established'","task_pretraining_exclusion='AE-C trained only cookie/drawer_bowl/stove; DINO upstream exposure unknown'")
 scope=dict(original.__dict__);scope.update(load=load,read=read,ROOT=ROOT)
 save(ROOT/'extraction_source_adaptation.json',dict(source_sha256=hashlib.sha256(inspect.getsource(original.main).encode()).hexdigest(),effective_sha256=hashlib.sha256(source.encode()).hexdigest(),horizon='current t to t+h, h1..8; all policy anchors complete; tail cache explicit masks',no_AE_update=True))
 exec(compile(source,__file__,'exec'),scope);scope['main']()
if __name__=='__main__':main()
