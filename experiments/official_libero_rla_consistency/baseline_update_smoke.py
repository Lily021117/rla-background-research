"""Isolated official baseline update/save/reload gate; never formal initialization."""
import argparse,os,sys,json,random,ctypes,uuid,subprocess,importlib.util,hashlib,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--batch-size',type=int,default=1);p.add_argument('--mode',choices=['smoke','formal','development'],default='smoke');a=p.parse_args()
o=a.run/('development' if a.mode=='development' else 'formal_B' if a.mode=='formal' else ('isolated_smoke' if a.batch_size==1 else f'isolated_smoke_batch{a.batch_size}'));o.mkdir(exist_ok=False)
sys.path.insert(0,str(a.reference/'vla-official'))
import tensorflow as tf
tf.config.set_visible_devices([],'GPU');torch.set_num_threads(4)
from transformers import AutoTokenizer
from peft import LoraConfig,get_peft_model
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.projectors import ProprioProjector
from prismatic.models.backbones.llm.prompting import QwenPromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.datasets.rlds.traj_transforms import chunk_act_obs
from prismatic.vla.datasets.rlds.utils.data_utils import normalize_action_and_proprio
from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE,STOP_INDEX,NUM_TOKENS,IGNORE_INDEX
def save(n,x):
    with (o/n).open('x') as f:json.dump(x,f,indent=2)
def ecc():return subprocess.check_output(['nvidia-smi','-i',os.environ['RLA_GPU_UUID'],'-q','-d','ECC'],text=True)
def counts(s):return [x.strip() for x in s.splitlines() if any(k in x for k in ('Correctable','Uncorrectable','SRAM SM','Threshold Exceeded'))]
before=ecc();(o/'ecc_before.txt').write_text(before)
cu=ctypes.CDLL('libcuda.so.1');cu.cuInit(0);dev=ctypes.c_int();assert cu.cuDeviceGet(ctypes.byref(dev),0)==0
buf=(ctypes.c_ubyte*16)();assert cu.cuDeviceGetUuid_v2(ctypes.byref(buf),dev)==0
assert str(uuid.UUID(bytes=bytes(buf)))==os.environ['RLA_GPU_UUID'].removeprefix('GPU-')
assert torch.cuda.mem_get_info()[0]>=26*1024**3
torch.cuda.set_per_process_memory_fraction(.49)
torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
seed=20261023;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
configdir=a.reference/'vla-official/pretrained_models/configs'
model=OpenVLAForActionPrediction(OpenVLAConfig.from_pretrained(str(configdir)))
source=torch.load(a.reference/'resolved_prismatic_initialization.pt',map_location='cpu')
rename=[('vision_backbone.dino_featurizer','vision_backbone.featurizer'),('vision_backbone.siglip_featurizer','vision_backbone.fused_featurizer'),('llm_backbone.llm','language_model'),('projector.projector.0','projector.fc1'),('projector.projector.2','projector.fc2'),('projector.projector.4','projector.fc3'),('gamma','scale_factor')]
mapped={}
for k,x in source.items():
    for old,new in rename:k=k.replace(old,new)
    mapped[k]=x
expected=model.state_dict();missing=set(expected)-set(mapped)
assert missing=={'action_queries.weight'} and not set(mapped)-set(expected)
model.load_state_dict({k:mapped.get(k,x) for k,x in expected.items()},strict=True)
del source,mapped,expected
model.vision_backbone.set_num_images_in_input(2)
model=get_peft_model(model,LoraConfig(r=64,lora_alpha=128,lora_dropout=0.,target_modules='all-linear',init_lora_weights='gaussian'))
for n,x in model.named_parameters():
    if 'action_queries' in n:x.requires_grad_(True)
model=model.to('cuda',dtype=torch.bfloat16).train()
prop=ProprioProjector(llm_dim=model.llm_dim,proprio_dim=8).to('cuda',dtype=torch.bfloat16).train()
head=L1RegressionActionHead(input_dim=model.llm_dim,hidden_dim=model.llm_dim,action_dim=7,use_pro_version=True).to('cuda',dtype=torch.bfloat16).train()
mods={'vla':model,'proprio':prop,'head':head}
params=[x for m in mods.values() for x in m.parameters() if x.requires_grad]
save('configuration.json',dict(seed=seed,lr=2e-4,optimizer='AdamW',betas=[.9,.999],eps=1e-8,weight_decay=.01,batch=a.batch_size,backend='Math SDPA',trainable_parameters=sum(x.numel() for x in params),parameter_names={k:[n for n,x in m.named_parameters() if x.requires_grad] for k,m in mods.items()},formal_initialization=a.mode=='formal',RLA=False))
tokenizer=AutoTokenizer.from_pretrained(str(configdir));imageproc=PrismaticImageProcessor.from_pretrained(str(configdir));at=ActionTokenizer(tokenizer)
transform=RLDSBatchTransform(at,tokenizer,lambda img:imageproc(img,return_tensors='pt')['pixel_values'][0],QwenPromptBuilder,use_wrist_image=True,use_proprio=True,use_minivlm=True)
collate=PaddedCollatorForActionPrediction(2048,tokenizer.pad_token_id)
stats=json.loads((a.run/'normalization_statistics_train_only.json').read_text());stats={k:{kk:tf.convert_to_tensor(vv) for kk,vv in val.items()} for k,val in stats.items()}
lab=np.load(a.reference/'trajectory_labels.npz');N=len(lab['actions']);actions=lab['actions'].astype(np.float32).copy();actions[:,-1]=1-np.clip(actions[:,-1],0,1)
rgb=np.stack([np.load(a.reference/'frames'/f'{i:04}.npz')['clean'] for i in range(N)]).transpose(0,1,3,4,2)
lang=json.loads((a.reference/'trajectory_provenance.json').read_text())['instruction']
traj={'action':tf.constant(actions),'observation':{'proprio':tf.constant(lab['proprio']),'image_primary':tf.constant(rgb[:,0]),'image_wrist':tf.constant(rgb[:,1]),'timestep':tf.range(N)},'task':{'language_instruction':tf.constant([lang]*N)},'dataset_name':tf.constant(['single_task_dual224_fk_v3']*N),'absolute_action_mask':tf.constant(np.tile([False]*6+[True],(N,1)))}
chunk=chunk_act_obs(normalize_action_and_proprio(traj,stats,ACTION_PROPRIO_NORMALIZATION_TYPE),window_size=1,future_action_window_size=7)
raw=tf.nest.map_structure(lambda x:x[0].numpy(),chunk);item=transform(raw)
prompt_len=len(item['input_ids'])-NUM_TOKENS
item['input_ids']=torch.cat([item['input_ids'],torch.tensor([STOP_INDEX])]);item['labels']=item['input_ids'].clone();item['labels'][:prompt_len]=IGNORE_INDEX
items=[item]
for t in range(1,a.batch_size):
    raw_t=tf.nest.map_structure(lambda x:x[t].numpy(),chunk);it=transform(raw_t);pl=len(it['input_ids'])-NUM_TOKENS
    it['input_ids']=torch.cat([it['input_ids'],torch.tensor([STOP_INDEX])]);it['labels']=it['input_ids'].clone();it['labels'][:pl]=IGNORE_INDEX;items.append(it)
batch=collate(items);batch['labels']=batch['labels'].to('cuda');batch['proprio']=batch['proprio'].to('cuda')
assert int(model._process_action_masks(batch['labels']).sum())==NUM_TOKENS*a.batch_size
save('batch_audit.json',dict(episode='sealed_163_demo_12',timestep=0,actions_halfopen=[0,8],language=lang,shapes={k:list(v.shape) for k,v in batch.items() if torch.is_tensor(v)},future_input=False,stop=int(batch['input_ids'][0,-1]),train_only_normalization=True))
spec=importlib.util.spec_from_file_location('official_finetune',a.reference/'vla-official/vla-scripts/finetune.py');ft=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ft;spec.loader.exec_module(ft)
def forward():
    return ft.run_forward_pass(vla=model,action_head=SimpleNamespace(module=head),proprio_projector=prop,batch=batch,action_tokenizer=at,device_id='cuda',use_l1_regression=True,use_proprio=True,use_film=False,num_patches=model.vision_backbone.get_num_patches()*2,use_pro_version=True,cfg=SimpleNamespace(phase='Training'))
def trainable_state():return {k:{n:x.detach().cpu().clone() for n,x in m.named_parameters() if x.requires_grad} for k,m in mods.items()}
if a.mode=='development':
    from single_task_rollout import execute
    execute(globals())
    sys.exit(0)
optimizer=torch.optim.AdamW(params,lr=2e-4);scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100000],gamma=.1)
if a.mode=='formal':
    from single_task_training import execute
    execute(globals())
    sys.exit(0)
started=time.time();loss,metrics=forward();assert torch.isfinite(loss);loss.backward()
assert all(x.grad is None or torch.isfinite(x.grad).all() for x in params)
norms={k:float(torch.stack([x.grad.detach().float().norm().square() for x in m.parameters() if x.grad is not None]).sum().sqrt()) for k,m in mods.items()}
optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
snapshot=dict(parameters=trainable_state(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),global_step=1,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),scope='ISOLATED_SMOKE_NOT_FORMAL_INIT')
torch.save(snapshot,o/'isolated_update1.pt');restored=torch.load(o/'isolated_update1.pt',map_location='cpu')
for k,m in mods.items():
    names={n:x for n,x in m.named_parameters() if x.requires_grad};assert names.keys()==restored['parameters'][k].keys()
    with torch.no_grad():
        for n,x in names.items():x.copy_(restored['parameters'][k][n].to(x.device));assert torch.equal(x.cpu(),restored['parameters'][k][n])
optimizer.load_state_dict(restored['optimizer']);scheduler.load_state_dict(restored['scheduler'])
torch.set_rng_state(restored['torch_rng']);torch.cuda.set_rng_state_all(restored['cuda_rng']);random.setstate(restored['python_rng']);np.random.set_state(restored['numpy_rng'])
with torch.no_grad(): check,_=forward()
assert torch.isfinite(check)
after=ecc();(o/'ecc_after.txt').write_text(after)
def counts(s):return [x.strip() for x in s.splitlines() if any(k in x for k in ('Correctable','Uncorrectable','SRAM SM','Threshold Exceeded'))]
assert counts(before)==counts(after)
save('result.json',dict(status='PASS_ISOLATED_UPDATE_RELOAD',loss=float(loss),reloaded_loss=float(check),metrics=metrics,gradient_norms=norms,optimizer_updates=1,formal_updates=0,elapsed_s=time.time()-started,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),ECC_increment=False,backend=dict(math=torch.backends.cuda.math_sdp_enabled(),flash=torch.backends.cuda.flash_sdp_enabled(),efficient=torch.backends.cuda.mem_efficient_sdp_enabled())))
print('SMOKE PASS',flush=True)
