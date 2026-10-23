"""Official upstream batch transforms and forward, using derived real frames.
No optimizer is constructed. Sidecar variant selection is explicit and local.
"""
import argparse,os,sys,json,hashlib,importlib.util,random,subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--stop-contract',choices=['upstream','inference-aligned'],default='upstream');a=p.parse_args();o=a.out;v=o/'vla-official';w=a.project/'worktrees/rla-official-0fc6e97'
os.environ.update(RLA_STEPS='8',RLA_QUERIES='16',RLA_DIM='64',RLA_SIDECAR=str(o/'sidecar_clean'),LAMBDA_RLA='auto:0.25',LAMBDA_ACT='1')
sys.path[:0]=[str(v),str(w/'rebuttal/exp2_vla_adapter/vla_adapter_overlay')]
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
from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE,STOP_INDEX
from rla import data as rd
from rla.sidecar import RlaStore
from rla.action_head import RlaL1RegressionActionHead
from rla.loss import make_run_forward_pass,last_metrics
def js(n,x):
 if a.stop_contract!='upstream':n=n.replace('.json','_stop_aligned.json')
 if (o/n).exists():
  assert json.load(open(o/n))==x,'Refuse changed previous audit'
  return
 with (o/n).open('x') as f:json.dump(x,f,indent=2)
uuid=os.environ['RLA_GPU_UUID']
# torch2.2 has no device UUID property: CUDA driver confirms visible device UUID.
import ctypes
cu=ctypes.CDLL('libcuda.so.1');cu.cuInit(0);device=ctypes.c_int();assert cu.cuDeviceGet(ctypes.byref(device),0)==0
buf=(ctypes.c_ubyte*16)();assert cu.cuDeviceGetUuid_v2(ctypes.byref(buf),device)==0
import uuid as uuidlib
assert str(uuidlib.UUID(bytes=bytes(buf)))==uuid.removeprefix('GPU-')
assert torch.cuda.mem_get_info()[0]>=26*1024**3;torch.cuda.set_per_process_memory_fraction(.6)
ecc=subprocess.check_output(['nvidia-smi','-i',uuid,'-q','-d','ECC'],text=True);(o/'vla_ecc_before.txt').write_text(ecc)
seed=20261023;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
configdir=v/'pretrained_models/configs';cfg=OpenVLAConfig.from_pretrained(str(configdir));model=OpenVLAForActionPrediction(cfg)
source=torch.load(o/'resolved_prismatic_initialization.pt',map_location='cpu');rename=[('vision_backbone.dino_featurizer','vision_backbone.featurizer'),('vision_backbone.siglip_featurizer','vision_backbone.fused_featurizer'),('llm_backbone.llm','language_model'),('projector.projector.0','projector.fc1'),('projector.projector.2','projector.fc2'),('projector.projector.4','projector.fc3'),('gamma','scale_factor')]
mapped={}
for k,x in source.items():
 for old,new in rename:k=k.replace(old,new)
 mapped[k]=x
expected=model.state_dict();missing=set(expected)-set(mapped);extra=set(mapped)-set(expected)
print('MAPPING_MISSING',sorted(missing),'EXTRA',sorted(extra),flush=True)
assert not extra
# Official finetune intentionally adds policy-only action query parameters.
assert all('action_queries' in k for k in missing),missing
merged={k:mapped.get(k,x) for k,x in expected.items()};model.load_state_dict(merged,strict=True)
del source,mapped,merged,expected
model.vision_backbone.set_num_images_in_input(2)
model=get_peft_model(model,LoraConfig(r=64,lora_alpha=128,lora_dropout=0.,target_modules='all-linear',init_lora_weights='gaussian')).to('cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
prop=ProprioProjector(llm_dim=model.llm_dim,proprio_dim=8).to('cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
tokenizer=AutoTokenizer.from_pretrained(str(configdir));imageproc=PrismaticImageProcessor.from_pretrained(str(configdir));at=ActionTokenizer(tokenizer)
def transform(img):return imageproc(img,return_tensors='pt')['pixel_values'][0]
base_transform=RLDSBatchTransform(at,tokenizer,transform,QwenPromptBuilder,use_wrist_image=True,use_proprio=True,use_minivlm=True)
rla_transform=rd.RlaBatchTransform(at,tokenizer,transform,QwenPromptBuilder,use_wrist_image=True,use_proprio=True,use_minivlm=True)
base_collate=PaddedCollatorForActionPrediction(2048,tokenizer.pad_token_id);rla_collate=rd.RlaCollator(2048,tokenizer.pad_token_id)
spec=importlib.util.spec_from_file_location('official_finetune',v/'vla-scripts/finetune.py');ft=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ft;spec.loader.exec_module(ft)
normpath=a.project/'artifacts/rla_policy_closed_loop_pilot/runtime_checkpoint/dataset_statistics.json';stats=json.load(open(normpath))['libero_spatial_no_noops'];stats={k:{kk:tf.convert_to_tensor(vv) for kk,vv in vals.items()} for k,vals in stats.items() if k in ['action','proprio']}
lab=np.load(o/'trajectory_labels.npz');N=len(lab['proprio']);instruction=json.load(open(o/'trajectory_provenance.json'))['instruction'];batches={};joins=[]
for appearance in ['clean','background']:
 store=RlaStore(str(o/('sidecar_'+appearance)),chunk=8,num_tokens=16,token_dim=64,norm='center');rd.get_store=lambda:store
 rgb=np.stack([np.load(o/'frames'/f'{t:04}.npz')[appearance] for t in range(N)]).transpose(0,1,3,4,2)
 actions=lab['actions'].astype(np.float32).copy();actions[:,-1]=1-np.clip(actions[:,-1],0,1)
 traj={'action':tf.constant(actions),'observation':{'proprio':tf.constant(lab['proprio']),'image_primary':tf.constant(rgb[:,0]),'image_wrist':tf.constant(rgb[:,1]),'timestep':tf.range(N)},'task':{'language_instruction':tf.constant([instruction]*N)},'dataset_name':tf.constant(['derived_libero_spatial_v2']*N),'absolute_action_mask':tf.constant(np.tile([False]*6+[True],(N,1)))}
 traj=rd._normalize_with_rla(traj,stats,ACTION_PROPRIO_NORMALIZATION_TYPE);chunk=chunk_act_obs(traj,window_size=1,future_action_window_size=7)
 # Last action-complete window lacks its final saved future observation: do not accept it.
 t=0;assert lab['valid_horizon_mask'][t].all();raw=tf.nest.map_structure(lambda x:x[t].numpy(),chunk)
 rng=random.getstate();base=base_collate([base_transform(raw)]);after_rng=random.getstate()
 random.setstate(rng);rb=rla_collate([rla_transform(raw)]);random.setstate(after_rng)
 if a.stop_contract=='inference-aligned':
  # Explicit dataset-boundary fix, not an upstream model/head change. The
  # deployment helper appends64 placeholders and STOP; miniVLM training omits STOP.
  from prismatic.vla.constants import NUM_TOKENS,IGNORE_INDEX
  for item in [base,rb]:
   assert item['input_ids'].shape[0]==1
   prompt_len=item['input_ids'].shape[1]-NUM_TOKENS
   item['input_ids']=torch.cat([item['input_ids'],torch.full((1,1),STOP_INDEX,dtype=item['input_ids'].dtype)],1)
   item['labels']=item['input_ids'].clone();item['labels'][:,:prompt_len]=IGNORE_INDEX
   item['attention_mask']=torch.cat([item['attention_mask'],torch.ones((1,1),dtype=torch.bool)],1)
   assert int(model._process_action_masks(item['labels']).sum())==NUM_TOKENS
 for k in base:
  if torch.is_tensor(base[k]):assert torch.equal(base[k],rb[k]),k
 assert rb['rla'].shape==(1,8,1024) and np.array_equal(rb['rla'][0].numpy(),store.frame(store.slot(lab['proprio']),t))
 batches[appearance]=(base,rb);joins.append(dict(appearance=appearance,slot=store.slot(lab['proprio']),source_t=0,target_frames=list(range(1,9)),input_shape=list(base['pixel_values'].shape),proprio_shape=list(base['proprio'].shape),action_shape=list(base['actions'].shape),rla_shape=list(rb['rla'].shape),last_token=int(base['input_ids'][0,-1]),STOP_INDEX=int(STOP_INDEX),language=instruction,normalized_proprio=base['proprio'].tolist()))
js('real_batch_audit.json',dict(joins=joins,stats_SHA256=hashlib.sha256(normpath.read_bytes()).hexdigest(),normalization='official BOUNDS_Q99 with published Spatial stats; same stats both variants',tail_official_action_windows=N-7,strict_complete_future_windows=N-8,STOP=a.stop_contract,updates=0))
results=[];isolation={}
for name,cls in [('baseline',L1RegressionActionHead),('RLA',RlaL1RegressionActionHead)]:
 torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 head=cls(input_dim=model.llm_dim,hidden_dim=model.llm_dim,action_dim=7,use_pro_version=True).to('cuda',dtype=torch.bfloat16).requires_grad_(False)
 # Official overlay only emits auxiliary output in train mode; no autograd/update.
 head.train();captured={};original=head.predict_action
 def capture(*args,**kwargs):
  x=original(*args,**kwargs);captured['action']=x.detach().clone();return x
 head.predict_action=capture
 fn=ft.run_forward_pass if name=='baseline' else make_run_forward_pass(ft.run_forward_pass)
 for appearance,(base,rb) in batches.items():
  batch=base if name=='baseline' else rb
  # Single-process equivalent of DDP's device transfer for the proprio module.
  batch=dict(batch);batch['proprio']=batch['proprio'].to('cuda');batch['labels']=batch['labels'].to('cuda')
  with torch.no_grad():
   torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
   loss,metrics=fn(vla=model,action_head=SimpleNamespace(module=head),proprio_projector=prop,batch=batch,action_tokenizer=at,device_id='cuda',use_l1_regression=True,use_proprio=True,use_film=False,num_patches=model.vision_backbone.get_num_patches()*2,use_pro_version=True,cfg=SimpleNamespace(phase='Training'))
  assert torch.isfinite(loss) and torch.isfinite(captured['action']).all()
  results.append(dict(path=name,appearance=appearance,action_shape=list(captured['action'].shape),loss=float(loss),metrics=metrics,auxiliary=last_metrics() if name=='RLA' else None))
  print('FORWARD',name,appearance,float(loss),flush=True)
  if name=='RLA' and appearance=='clean':
   prior=captured['action'].clone();changed=dict(batch);changed['actions']=torch.zeros_like(batch['actions']);changed['rla']=torch.zeros_like(batch['rla'])
   with torch.no_grad():
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    fn(vla=model,action_head=SimpleNamespace(module=head),proprio_projector=prop,batch=changed,action_tokenizer=at,device_id='cuda',use_l1_regression=True,use_proprio=True,use_film=False,num_patches=model.vision_backbone.get_num_patches()*2,use_pro_version=True,cfg=SimpleNamespace(phase='Training'))
   assert torch.equal(prior,captured['action'])
   isolation=dict(numeric_action_labels_and_z_changed=True,action_output_max_difference=0.,future_RGB_in_batch=False,scope='supervised target isolation, not a closed-loop evaluation')
 del head
after=subprocess.check_output(['nvidia-smi','-i',uuid,'-q','-d','ECC'],text=True);(o/'vla_ecc_after.txt').write_text(after)
def counts(x):return [l.strip() for l in x.splitlines() if ':' in l and any(k in l for k in ['Correctable','Uncorrectable','SRAM SM','Threshold Exceeded'])]
assert counts(ecc)==counts(after)
js('official_forward_results.json',dict(status='PASS_FORWARD_NOT_POLICY_QUALITY',results=results,isolation=isolation,initialization='official Prismatic multimodal checkpoint plus pretrained vision; fresh action/proprio/LoRA',missing_source_policy_queries=sorted(missing),strict_assembled_load=True,peak_GPU_allocated=torch.cuda.max_memory_allocated(),peak_GPU_reserved=torch.cuda.max_memory_reserved(),ECC_increment=False,optimizer_updates=0,backward=0,rollouts=0))
