"""Official model bootstrap; stage entry determines training/evaluation scope.

No unrelated historical cookie batch is read. Formal initialization restores
the accepted common trainable parameters and all RNG states before training.
"""
import argparse,os,sys,random,ctypes,uuid,subprocess,time,gc,importlib.util
from types import SimpleNamespace
from pathlib import Path
import json,hashlib
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'official_libero_rla_consistency'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'official_libero_rla_consistency/stove_diagnosis'))
from common import *
import torch
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--arm',required=True);p.add_argument('--attempt',default='isolated_smoke');p.add_argument('--entry',default='rla_lr_budget_sweep.training');a=p.parse_args();plan=read(a.out/'run_plan.json');ref=Path(plan['inputs']['project'])/'experiments/official_libero_rla_consistency/run_20261023_02';a.run=a.out/a.arm;a.run.mkdir(exist_ok=True);o=a.run/a.attempt;o.mkdir(exist_ok=False)
os.environ['CENTER_RUN']=str(a.out);os.environ['CENTER_ARM']=a.arm;os.environ['POLICY_ARM']=a.arm;os.environ['POLICY_CHECKPOINTING']='1';os.environ['POLICY_MICROBATCH']='8'

GPU=os.environ['RLA_GPU_UUID'];before=subprocess.check_output(['nvidia-smi','-i',GPU,'-q','-d','ECC'],text=True);(o/'ecc_before.txt').write_text(before)
def counters(s):return [x.strip() for x in s.splitlines() if any(k in x for k in ('Correctable','Uncorrectable','SRAM SM','Threshold Exceeded'))]
assert int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))>=26624
cu=ctypes.CDLL('libcuda.so.1');cu.cuInit(0);dev=ctypes.c_int();assert cu.cuDeviceGet(ctypes.byref(dev),0)==0;buf=(ctypes.c_ubyte*16)();assert cu.cuDeviceGetUuid_v2(ctypes.byref(buf),dev)==0;assert str(uuid.UUID(bytes=bytes(buf)))==GPU.removeprefix('GPU-')
sys.path.insert(0,str(ref/'vla-official'));sys.path.insert(0,str(Path(plan['inputs']['project'])/'worktrees/rla-official-0fc6e97/rebuttal/exp2_vla_adapter/vla_adapter_overlay'));sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tensorflow as tf
tf.config.set_visible_devices([],'GPU');torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.55)
torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
random.seed(20261023);np.random.seed(20261023);torch.manual_seed(20261023);torch.cuda.manual_seed_all(20261023)
from transformers import AutoTokenizer
from peft import LoraConfig,get_peft_model
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor
from prismatic.models.projectors import ProprioProjector
from prismatic.models.backbones.llm.prompting import QwenPromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform
from prismatic.vla.constants import NUM_TOKENS
from rla.action_head import RlaL1RegressionActionHead
from rla.data import RlaCollator
from rla.sidecar import RlaStore
from coverage_gripper.dataset import DerivedEpisodeDataset
configdir=ref/'vla-official/pretrained_models/configs';model=OpenVLAForActionPrediction(OpenVLAConfig.from_pretrained(str(configdir)));source=torch.load(ref/'resolved_prismatic_initialization.pt',map_location='cpu');mapped={}
rename=[('vision_backbone.dino_featurizer','vision_backbone.featurizer'),('vision_backbone.siglip_featurizer','vision_backbone.fused_featurizer'),('llm_backbone.llm','language_model'),('projector.projector.0','projector.fc1'),('projector.projector.2','projector.fc2'),('projector.projector.4','projector.fc3'),('gamma','scale_factor')]
for k,v in source.items():
 for old,new in rename:k=k.replace(old,new)
 mapped[k]=v
state=model.state_dict();assert set(state)-set(mapped)=={'action_queries.weight'} and not set(mapped)-set(state);model.load_state_dict({k:mapped.get(k,v) for k,v in state.items()},strict=True);del state,source,mapped
model.vision_backbone.set_num_images_in_input(2);model=get_peft_model(model,LoraConfig(r=64,lora_alpha=128,lora_dropout=0.,target_modules='all-linear',init_lora_weights='gaussian'))
for n,v in model.named_parameters():
 if 'action_queries' in n:v.requires_grad_(True)
model=model.to('cuda',dtype=torch.bfloat16);prop=ProprioProjector(llm_dim=model.llm_dim,proprio_dim=8).to('cuda',dtype=torch.bfloat16);head=RlaL1RegressionActionHead(input_dim=model.llm_dim,hidden_dim=model.llm_dim,action_dim=7,use_pro_version=True).to('cuda',dtype=torch.bfloat16);mods=dict(vla=model,proprio=prop,head=head)
tokenizer=AutoTokenizer.from_pretrained(str(configdir));imageproc=PrismaticImageProcessor.from_pretrained(str(configdir));at=ActionTokenizer(tokenizer);collator=RlaCollator(2048,tokenizer.pad_token_id);transform=RLDSBatchTransform(at,tokenizer,lambda img:imageproc(img,return_tensors='pt')['pixel_values'][0],QwenPromptBuilder,use_wrist_image=True,use_proprio=True,use_minivlm=True)

collate=collator
def save(n,x):
 with (o/n).open('x') as f:json.dump(x,f,indent=2)
def ecc():return subprocess.check_output(['nvidia-smi','-i',GPU,'-q','-d','ECC'],text=True)
counts=counters
def trainable_state():return {k:{n:v.detach().cpu().clone() for n,v in m.named_parameters() if v.requires_grad} for k,m in mods.items()}
spec=importlib.util.spec_from_file_location('official_finetune_centering',ref/'vla-official/vla-scripts/finetune.py');ft=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ft;spec.loader.exec_module(ft)
params=[v for m in mods.values() for v in m.parameters() if v.requires_grad]
import importlib
importlib.import_module(a.entry).execute(globals())
(o/'ecc_after.txt').write_text(ecc());assert counts(before)==counts(ecc())
print('ENTRY_COMPLETE',a.arm,flush=True)
