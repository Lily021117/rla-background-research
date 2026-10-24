"""Frozen policy checkpoints: same-sample forward and separate loss gradients.

Uses accepted official model/loader/forward, without the old cookie bootstrap
batch. No optimizer is instantiated, no weight is updated or saved.
"""
import argparse,os,sys,random,ctypes,uuid,subprocess,time,gc,importlib.util
from types import SimpleNamespace
from common import *
import torch
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--result-name',default='policy');a=p.parse_args();cfg=read(a.out/'configuration.json');root=Path(cfg['source']);plan=read(root/'policy_run/run_plan.json');ref=Path(plan['inputs']['project'])/'experiments/official_libero_rla_consistency/run_20261023_02';fixed=read(a.out/'fixed_sample_manifest.json');o=a.out/a.result_name;o.mkdir(exist_ok=False)
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
ds=DerivedEpisodeDataset(plan['inputs']['data_manifest'],plan['inputs']['action_proprio_statistics'],transform,collator);stores={}
for arm in ('C','D'):
 stores[arm]={c:RlaStore(str(root/'ae_run/sidecars'/arm/c),chunk=8,num_tokens=16,token_dim=64,norm='center') for c in ('clean','background')}
slots={e:{c:stores['C'][c].slot(np.load(r/'trajectory_labels.npz')['proprio']) for c in ('clean','background')} for e,(r,_,_) in ds.episodes.items()}
spec=importlib.util.spec_from_file_location('official_finetune_diagnostic',ref/'vla-official/vla-scripts/finetune.py');ft=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ft;spec.loader.exec_module(ft)
capture={};predict=head.predict_action
def pred(*args,**kw):
 v=predict(*args,**kw);capture['action']=v;return v
head.predict_action=pred
def batch(xs,arm):
 items=[]
 for x in xs:
  e,t,c=x['episode'],x['t'],x['appearance'];it=ds.sample(e,t,c);it['rla']=stores[arm][c].frame(slots[e][c],t);items.append(it)
 b=collator(items);b['labels']=b['labels'].cuda();b['proprio']=b['proprio'].cuda();assert int(model._process_action_masks(b['labels']).sum())==len(xs)*NUM_TOKENS
 for i,x in enumerate(xs):assert ds.episodes[x['episode']][2]['task'].lower() in tokenizer.decode(b['input_ids'][i].tolist()).lower()
 return b
def forward(b):
 capture.clear();act,_=ft.run_forward_pass(vla=model,action_head=SimpleNamespace(module=head),proprio_projector=prop,batch=b,action_tokenizer=at,device_id='cuda',use_l1_regression=True,use_proprio=True,use_film=False,num_patches=model.vision_backbone.get_num_patches()*2,use_pro_version=True,cfg=SimpleNamespace(phase='Inference'));z=head.pop_z_hat();assert z is not None
 aux=torch.nn.functional.l1_loss(z,b['rla'].cuda().to(z.dtype));return act,aux
def reset_rng():
 torch.manual_seed(20261023);torch.cuda.manual_seed_all(20261023);np.random.seed(20261023);random.seed(20261023)
def zero():
 for m in mods.values():m.zero_grad(set_to_none=True)
def parameter_digest():return digest_state(dict(parameters={k:{n:v.detach().cpu() for n,v in m.named_parameters() if v.requires_grad} for k,m in mods.items()}))
rows=[];grads=[];checks=[];start=time.time();lam=plan['shared_protocol']['lambda_aux'];shared=[(n,v) for n,v in model.named_parameters() if v.requires_grad]
stats=read(plan['inputs']['action_proprio_statistics'])['action'];scale=(np.array(stats['q99'])-np.array(stats['q01']))/2;mask=np.array(stats['mask'],bool);scale=np.where(mask,scale,1.)
for arm in ('C','D'):
 for step in cfg['policy_steps']:
  path=root/'policy_run'/arm/'isolated_smoke'/f'step{step:04}.pt';ck=torch.load(path,map_location='cpu');assert ck['global_step']==step and ck['configuration']==plan
  for k,m in mods.items():
   assert set(ck['parameters'][k])=={n for n,v in m.named_parameters() if v.requires_grad};full=m.state_dict();full.update(ck['parameters'][k]);m.load_state_dict(full,strict=True);m.eval()
  beforehash=digest_state(ck);del ck;gc.collect();zero();reset_rng()
  with torch.inference_mode():
   for starti in range(0,len(fixed['policy']),4):
    xs=fixed['policy'][starti:starti+4];b=batch(xs,arm);act,aux=forward(b);predicted=capture['action'].float();target=b['actions'].cuda().float();err=abs(predicted-target);assert torch.isfinite(err).all() and torch.isfinite(aux)
    # Auxiliary per-batch loss has official dtype/reduction; per-sample predictions saved locally only.
    for i,x in enumerate(xs):
     for h in range(8):
      row=dict(model=arm,step=step,**x,chunk_position=h,action_L1=float(err[i,h].mean()),auxiliary_L1_batch=float(aux),weighted_auxiliary_batch=lam*float(aux))
      for d in range(7):row[f'dim{d}_L1']=float(err[i,h,d]);row[f'dim{d}_decoded_command_L1']=float(err[i,h,d])*float(scale[d]);row[f'dim{d}_prediction']=float(predicted[i,h,d]);row[f'dim{d}_target']=float(target[i,h,d])
      rows.append(row)
    capture.clear();del b,act,aux,predicted,target,err
  # Same effective batch8 for each split. Independent forwards, identical RNG;
  # accumulate in FP32 to avoid perturbing the diagnostic with BF16 accumulation.
  for split in ('train','validation'):
   xs=[dict(episode=x['episode'],t=x['t'],appearance=c) for x in fixed['gradient'] if x['split']==split for c in ('clean','background')];assert len(xs)==8;vectors=[];losses=[]
   for objective in ('action','weighted_auxiliary'):
    accum={};lossvalues=[];rawauxvalues=[];zero();reset_rng()
    for offset in range(0,len(xs),2):
     b=batch(xs[offset:offset+2],arm);act,aux=forward(b);loss=act if objective=='action' else lam*aux;assert torch.isfinite(loss);lossvalues.append(float(loss.detach()));rawauxvalues.append(float(aux.detach()));(loss/4).backward()
     for n,v in shared:
      if v.grad is not None:
       g=v.grad.detach().float().cpu();assert torch.isfinite(g).all()
       if n in accum:accum[n].add_(g)
       else:accum[n]=g.clone()
     zero();capture.clear();del loss,act,aux,b
    vectors.append(accum);losses.append(float(np.mean(lossvalues)))
   aa,bb=vectors;keys=set(aa)|set(bb);na=sum(float(x.double().square().sum()) for x in aa.values())**.5;nb=sum(float(x.double().square().sum()) for x in bb.values())**.5;dot=sum(float((aa[n].double()*bb[n].double()).sum()) for n in set(aa)&set(bb));cos=dot/max(na*nb,1e-30)
   grads.append(dict(model=arm,step=step,split=split,samples=8,mode='eval_with_grad',action_loss=losses[0],auxiliary_raw_loss=float(np.mean(rawauxvalues)),weighted_auxiliary_loss=losses[1],lambda_aux=lam,shared_action_gradient_norm=na,shared_weighted_aux_gradient_norm=nb,shared_raw_aux_gradient_norm_approx=nb/lam,cosine=cos,weighted_aux_action_norm_ratio=nb/max(na,1e-30),dot_product=dot));del vectors,aa,bb,accum;gc.collect()
  afterhash=parameter_digest();assert beforehash==afterhash;checks.append(dict(arm=arm,step=step,before=beforehash,after=afterhash,strict_load=True,updates=0));assert counters(before)==counters(subprocess.check_output(['nvidia-smi','-i',GPU,'-q','-d','ECC'],text=True));print('DIAG',arm,step,'seconds',time.time()-start,flush=True)
  torch.cuda.empty_cache()
csvsave(o/'action_by_dimension_horizon.csv',rows);csvsave(o/'gradient_conflict.csv',grads);save(o/'verification.json',dict(status='PASS_DIAGNOSTIC',parameter_checks=checks,optimizer_created=False,optimizer_updates=0,backward_calls=160,new_rollouts=0,mode='eval: deterministic frozen-model derivative, not stochastic training-step replay',peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),seconds=time.time()-start,shared_gradient_scope='all trainable VLM/backbone parameters; action and z branch both feed these; head excluded',normalization='official per-task training Q99; decoded_command_L1 before gripper threshold; translation/rotation command units, not meters/radians'))
(o/'ecc_after.txt').write_text(subprocess.check_output(['nvidia-smi','-i',GPU,'-q','-d','ECC'],text=True));print('POLICY DIAGNOSTIC COMPLETE',flush=True)
