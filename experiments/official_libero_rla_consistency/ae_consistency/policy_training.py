"""Official RLA head/loss and accepted derived loader, bounded paired training."""
import os,json,csv,time,random,hashlib,resource,dataclasses
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F

def initialize(g,kind='rla'):
 a=g['a'];root=a.run.parent;plan=json.loads((root/'run_plan.json').read_text());prev=Path(plan['previous']);ck=torch.load(plan['common_step0'],map_location='cpu')
 if kind=='rla':
  import sys
  sys.path.insert(0,str(Path(plan['inputs']['project'])/'worktrees/rla-official-0fc6e97/rebuttal/exp2_vla_adapter/vla_adapter_overlay'))
  from rla.action_head import RlaL1RegressionActionHead
  with torch.random.fork_rng(devices=[0]):
   torch.manual_seed(20261024);g['head']=RlaL1RegressionActionHead(input_dim=g['model'].llm_dim,hidden_dim=g['model'].llm_dim,action_dim=7,use_pro_version=True).to('cuda',dtype=torch.bfloat16)
  g['mods']['head']=g['head']
 for k,m in g['mods'].items():
  assert {n for n,p in m.named_parameters() if p.requires_grad}==set(ck['parameters'][k]);full=m.state_dict();full.update(ck['parameters'][k]);m.load_state_dict(full,strict=True)
 restore(ck);g['params']=[p for m in g['mods'].values() for p in m.parameters() if p.requires_grad];g.pop('optimizer',None)
 return plan,prev,ck

def restore(ck):
 torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng'])

def execute(g):
 a,o=g['a'],g['o'];arm=os.environ['POLICY_ARM'];plan,prev,initial=initialize(g);mods=g['mods'];model,head,prop=mods['vla'],mods['head'],mods['proprio']
 from coverage_gripper.dataset import DerivedEpisodeDataset
 from rla.data import RlaCollator
 from rla.sidecar import RlaStore
 import rla.loss as official_loss
 lam=plan['shared_protocol']['lambda_aux'];official_loss.CFG=dataclasses.replace(official_loss.CFG,lambda_auto=False,lambda_rla=lam)
 collator=RlaCollator(2048,g['tokenizer'].pad_token_id);ds=DerivedEpisodeDataset(plan['inputs']['data_manifest'],plan['inputs']['action_proprio_statistics'],g['transform'],collator)
 stores={c:RlaStore(str(prev/'sidecars'/('C' if arm.startswith('preflight') else arm)/c),chunk=8,num_tokens=16,token_dim=64,norm='center') for c in ('clean','background')};slots={}
 for e,(root,ch,meta) in ds.episodes.items():
  raw=np.load(root/'trajectory_labels.npz')['proprio'];slots[e]={c:stores[c].slot(raw) for c in stores};assert min(slots[e].values())>=0
 order=json.loads(Path(plan['inputs']['B20_sampling_order']).read_text());assert len(order)==4800
 def batch(examples):
  items=[]
  for e,t,c in examples:
   item=ds.sample(e,t,c);item['rla']=stores[c].frame(slots[e][c],t);assert item['rla'].shape==(8,1024);items.append(item)
  b=collator(items);b['labels']=b['labels'].cuda();b['proprio']=b['proprio'].cuda();assert int(model._process_action_masks(b['labels']).sum())==len(examples)*g['NUM_TOKENS'];return b
 capture={};predict=head.predict_action;pop=head.pop_z_hat
 def pred(*args,**kw):
  v=predict(*args,**kw);capture['action']=v.detach();return v
 def popz():
  z=pop();capture['z']=z;return z
 head.predict_action=pred;head.pop_z_hat=popz;wrapped=official_loss.make_run_forward_pass(g['ft'].run_forward_pass)
 def forward(b,phase='Training'):
  capture.clear();return wrapped(vla=model,action_head=SimpleNamespace(module=head),proprio_projector=prop,batch=b,action_tokenizer=g['at'],device_id='cuda',use_l1_regression=True,use_proprio=True,use_film=False,num_patches=model.vision_backbone.get_num_patches()*2,use_pro_version=True,cfg=SimpleNamespace(phase=phase))[0]
 # Explicit optional activation checkpointing preserves the microbatch and RNG.
 checkpointing=os.environ.get('POLICY_CHECKPOINTING','0')=='1'
 if checkpointing:
  from torch.utils.checkpoint import checkpoint
  model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
  for block in head.model.mlp_resnet_blocks:
   original=block.forward
   def checkpointed(*args,_original=original,**kwargs):return checkpoint(_original,*args,use_reentrant=False,preserve_rng_state=True,**kwargs)
   block.forward=checkpointed
  for visual in (model.vision_backbone.featurizer,model.vision_backbone.fused_featurizer):
   if hasattr(visual,'set_grad_checkpointing'):visual.set_grad_checkpointing(True)
 opt=torch.optim.AdamW(g['params'],lr=2e-4,weight_decay=.01);scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[600],gamma=.1)
 def snapshot(step):
  path=o/f'step{step:04}.pt';assert not path.exists();torch.save(dict(parameters=g['trainable_state'](),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),global_step=step,next_physical_sample=step*8,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),configuration=plan,arm=arm,scope='OFFICIAL_PAIRED_RLA_POLICY',AMP_scaler=None,dtype='bfloat16',activation_checkpointing=checkpointing),path)
 micro=int(os.environ.get('POLICY_MICROBATCH','8'));assert 16%micro==0
 g['save']('execution_configuration.json',dict(arm=arm,microbatch=micro,accumulation=16//micro,effective_appearance_batch=16,checkpointing=checkpointing,backend=dict(math=torch.backends.cuda.math_sdp_enabled(),flash=torch.backends.cuda.flash_sdp_enabled(),efficient=torch.backends.cuda.mem_efficient_sdp_enabled()),lambda_aux=lam,policy_updates=0 if arm.startswith('preflight') else 600))
 if arm.startswith('preflight'):
  moments=[torch.zeros_like(p) for p in g['params'] for _ in range(2)] # persistent AdamW moment-sized storage, present during forward
  examples=[(e,t,c) for e,t in order[:8] for c in ('clean','background')]
  for repeat in range(2):
   opt.zero_grad(set_to_none=True)
   for start in range(0,16,micro):
    b=batch(examples[start:start+micro]);loss=forward(b);assert torch.isfinite(loss);(loss/(16//micro)).backward();capture.clear();del loss,b
   workspace=[torch.zeros_like(p) for p in g['params']];del workspace;torch.cuda.empty_cache()
  assert all(p.grad is None or torch.isfinite(p.grad).all() for p in g['params'])
  g['save']('preflight_acceptance.json',dict(status='PASS_PERSISTENT_OPTIMIZER_MEMORY_TWO_FULL_EFFECTIVE_BATCHES',microbatch=micro,effective_appearance_batch=16,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),optimizer_updates=0,initialization_unchanged=True));print('FULL BATCH PREFLIGHT PASS',flush=True);return
 dev=[(e,t,c) for e,(_,_,meta) in ds.episodes.items() if meta['split']=='validation' for t in range(meta['valid_chunks']) for c in ('clean','background')]
 vf=(o/'validation_metrics.csv').open('x');vw=csv.DictWriter(vf,fieldnames=['step','episode','timestep','appearance','action_L1','translation_L1','rotation_L1','gripper_L1','auxiliary_L1']);vw.writeheader()
 def validate(step):
  states=dict(torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),numpy_rng=np.random.get_state())
  # The actual accepted overlay emits z in eval too; do not add training noise to validation.
  for m in mods.values():m.eval()
  with torch.inference_mode():
   for start in range(0,len(dev),micro):
    ex=dev[start:start+micro];b=batch(ex);loss=forward(b,'Inference');err=(capture['action'].float()-b['actions'].cuda().float()).abs();zerr=(capture['z'].float()-b['rla'].cuda().float()).abs();assert torch.isfinite(err).all() and torch.isfinite(zerr).all()
    for i,(e,t,c) in enumerate(ex):vw.writerow(dict(step=step,episode=e,timestep=t,appearance=c,action_L1=float(err[i].mean()),translation_L1=float(err[i,:,:3].mean()),rotation_L1=float(err[i,:,3:6].mean()),gripper_L1=float(err[i,:,6].mean()),auxiliary_L1=float(zerr[i].mean())))
  vf.flush();os.fsync(vf.fileno());capture.clear()
  for m in mods.values():m.train()
  restore(states)
 accepted=json.loads((a.run.parent/'preflight6/isolated_smoke/preflight_acceptance.json').read_text());assert accepted['microbatch']==micro==8 and accepted['optimizer_updates']==0 and checkpointing
 snapshot(0);validate(0);del initial
 fields=['step','action_loss','auxiliary_loss','weighted_auxiliary','total_loss','lr','gradient_vla','gradient_head','gradient_proprio','gradient_aux_head','allocated','reserved','cpu_maxrss_kib','seconds','successful_updates']
 f=(o/'training_metrics.csv').open('x');w=csv.DictWriter(f,fieldnames=fields);w.writeheader();started=time.time()
 for step in range(1,601):
  opt.zero_grad(set_to_none=True);examples=[(e,t,c) for e,t in order[(step-1)*8:step*8] for c in ('clean','background')];acts=[];auxs=[];totals=[]
  for start in range(0,16,micro):
   b=batch(examples[start:start+micro]);total=forward(b);assert torch.isfinite(total);m=official_loss.last_metrics();acts.append(m['action_l1']);auxs.append(m['rla_loss']);totals.append(float(total.detach()));(total/(16//micro)).backward();capture.clear();del total,b
  norms={}
  for k,m in mods.items():
   gs=[p.grad.detach().float() for p in m.parameters() if p.grad is not None];assert all(torch.isfinite(x).all() for x in gs);norms[k]=float(torch.stack([x.norm().square() for x in gs]).sum().sqrt())
  auxnorm=sum(float(p.grad.detach().float().square().sum()) for n,p in head.named_parameters() if p.grad is not None and ('z_' in n or 'rla' in n))**.5
  lr=opt.param_groups[0]['lr'];opt.step();scheduler.step();assert all(torch.isfinite(p).all() for p in g['params'])
  torch.cuda.empty_cache()
  w.writerow(dict(step=step,action_loss=np.mean(acts),auxiliary_loss=np.mean(auxs),weighted_auxiliary=lam*np.mean(auxs),total_loss=np.mean(totals),lr=lr,gradient_vla=norms['vla'],gradient_head=norms['head'],gradient_proprio=norms['proprio'],gradient_aux_head=auxnorm,allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved(),cpu_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,seconds=time.time()-started,successful_updates=step));f.flush()
  if step%25==0:print('POLICY',arm,step,'action',np.mean(acts),'aux',np.mean(auxs),'seconds',time.time()-started,flush=True);assert g['counts'](g['before'])==g['counts'](g['ecc']())
  if step==1:snapshot(step)
  if step in (100,300,600):snapshot(step);validate(step)
 f.close();vf.close();ck=torch.load(o/'step0600.pt',map_location='cpu');assert ck['global_step']==600 and ck['next_physical_sample']==4800
 for k,m in mods.items():
  full=m.state_dict();full.update(ck['parameters'][k]);m.load_state_dict(full,strict=True)
  for n,p in m.named_parameters():
   if p.requires_grad:assert torch.equal(p.detach().cpu(),ck['parameters'][k][n])
 g['save']('training_acceptance.json',dict(status='PASS_TRAINING_NOT_METHOD',successful_updates=600,strict_reload=True,seconds=time.time()-started,peak_allocated=torch.cuda.max_memory_allocated(),physical_exposures=4800,appearance_exposures=9600,ECC_increment=False))
