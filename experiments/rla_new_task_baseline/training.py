"""Three frozen new-task baselines, using official forward/loss, no AE updates."""
import os,json,csv,time,random,dataclasses,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from stove_centering.training import initialize,restore

def execute(g):
 arm=os.environ['POLICY_ARM'];standard=arm in ('BC-standard','preflight-standard')
 if standard:
  from prismatic.models.action_heads import L1RegressionActionHead
  with torch.random.fork_rng(devices=[0]):g['head']=L1RegressionActionHead(input_dim=g['model'].llm_dim,hidden_dim=g['model'].llm_dim,action_dim=7,use_pro_version=True).to('cuda',dtype=torch.bfloat16)
  g['mods']['head']=g['head']
 plan,prev,initial=initialize(g,'baseline' if standard else 'rla');mods=g['mods'];model,head,prop=mods['vla'],mods['head'],mods['proprio'];out=g['o']
 # Accepted RLA construction artifact predates the global_step field; its
 # exact file hash is verified by prepare_policy before this entry is used.
 assert initial.get('global_step',0)==0
 from coverage_gripper.dataset import DerivedEpisodeDataset
 from rla.data import RlaCollator
 from rla.sidecar import RlaStore
 import rla.loss as official_loss
 lam=0. if arm in ('BC-standard','BC-matched','preflight-standard') else .22408536585365854
 official_loss.CFG=dataclasses.replace(official_loss.CFG,lambda_auto=False,lambda_rla=lam)
 collator=RlaCollator(2048,g['tokenizer'].pad_token_id);ds=DerivedEpisodeDataset(plan['inputs']['data_manifest'],plan['inputs']['action_proprio_statistics'],g['transform'],collator)
 stores={} if standard else {c:RlaStore(str(prev/c),chunk=8,num_tokens=16,token_dim=64,norm='center') for c in ('clean','background')}
 slots={e:{c:store.slot(np.load(root/'trajectory_labels.npz')['proprio']) for c,store in stores.items()} for e,(root,_,_) in ds.episodes.items()}
 assert all(v>=0 for x in slots.values() for v in x.values())
 order=json.loads(Path(plan['inputs']['B20_sampling_order']).read_text());assert len(order)==4800
 capture={};predict=head.predict_action
 def pred(*args,**kw):
  v=predict(*args,**kw);capture['action']=v;return v
 head.predict_action=pred
 if not standard:
  pop=head.pop_z_hat
  def popz():
   z=pop();capture['z']=z;return z
  head.pop_z_hat=popz
 def batch(ex):
  items=[]
  for e,t,c in ex:
   item=ds.sample(e,t,c)
   if not standard:item['rla']=stores[c].frame(slots[e][c],t);assert item['rla'].shape==(8,1024)
   items.append(item)
  b=collator(items);b['labels']=b['labels'].cuda();b['proprio']=b['proprio'].cuda()
  assert int(model._process_action_masks(b['labels']).sum())==len(ex)*g['NUM_TOKENS']
  for i,(e,t,c) in enumerate(ex):assert ds.episodes[e][2]['task'].lower() in g['tokenizer'].decode(b['input_ids'][i].tolist()).lower()
  return b
 wrapped=g['ft'].run_forward_pass if standard else official_loss.make_run_forward_pass(g['ft'].run_forward_pass)
 def forward(b,phase='Training'):
  capture.clear();return wrapped(vla=model,action_head=SimpleNamespace(module=head),proprio_projector=prop,batch=b,action_tokenizer=g['at'],device_id='cuda',use_l1_regression=True,use_proprio=True,use_film=False,num_patches=model.vision_backbone.get_num_patches()*2,use_pro_version=True,cfg=SimpleNamespace(phase=phase))[0]
 from torch.utils.checkpoint import checkpoint
 model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
 for block in head.model.mlp_resnet_blocks:
  original=block.forward
  def checkpointed(*args,_original=original,**kwargs):return checkpoint(_original,*args,use_reentrant=False,preserve_rng_state=True,**kwargs)
  block.forward=checkpointed
 for visual in (model.vision_backbone.featurizer,model.vision_backbone.fused_featurizer):
  if hasattr(visual,'set_grad_checkpointing'):visual.set_grad_checkpointing(True)
 opt=torch.optim.AdamW(g['params'],lr=2e-4,weight_decay=.01);sched=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[600],gamma=.1)
 def rng():return dict(torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),numpy_rng=np.random.get_state())
 restore(initial)
 assert all(torch.equal(p.detach().cpu(),initial['parameters'][k][n]) for k,m in mods.items() for n,p in m.named_parameters() if p.requires_grad)
 g['save']('initialization_audit.json',dict(status='PASS',source=plan['inputs']['B20_step0'] if standard else plan['common_step0'],global_step=0,trainable_parameters_strictly_restored=True,model_RNG_restored=True,head='official standard' if standard else 'official RLA',auxiliary_weight=lam,task='open the middle drawer of the cabinet',backend='Math SDPA',BF16=True,activation_checkpointing=True))
 if arm.startswith('preflight'):
  examples=[(e,t,c) for e,t in order[:8] for c in ('clean','background')];checks=[]
  moments=[torch.zeros_like(p) for p in g['params'] for _ in range(2)]
  for mode in ['zero_original','zero_changed_target']:
   restore(initial);opt.zero_grad(set_to_none=True);official_loss.CFG=dataclasses.replace(official_loss.CFG,lambda_rla=0.);ah=hashlib.sha256();losses=[]
   for start in [0,8]:
    b=batch(examples[start:start+8])
    if not standard:
     assert not b['rla'].requires_grad
     if mode=='zero_changed_target':b['rla']=b['rla']+7
    loss=forward(b);assert torch.isfinite(loss);ah.update(capture['action'].detach().cpu().view(torch.uint8).numpy().tobytes());losses.append(float(loss.detach()));(loss/2).backward();capture.clear();del b,loss
   gh=hashlib.sha256()
   for k,m in mods.items():
    for n,p in m.named_parameters():
     if p.grad is not None:assert torch.isfinite(p.grad).all();gh.update((k+n).encode());gh.update(p.grad.detach().cpu().view(torch.uint8).numpy().tobytes())
   checks.append(dict(mode=mode,action_hash=ah.hexdigest(),gradient_hash=gh.hexdigest(),losses=losses))
  assert checks[0]['action_hash']==checks[1]['action_hash'] and checks[0]['gradient_hash']==checks[1]['gradient_hash'] and checks[0]['losses']==checks[1]['losses']
  shared=None
  if not standard:
   restore(initial);opt.zero_grad(set_to_none=True);official_loss.CFG=dataclasses.replace(official_loss.CFG,lambda_rla=lam);b=batch(examples[:8]);loss=forward(b);aux=(capture['z']-b['rla'].to('cuda',dtype=capture['z'].dtype)).abs().mean();(lam*aux/2).backward();shared=sum(float(p.grad.float().square().sum()) for p in model.parameters() if p.grad is not None)**.5;assert shared>0 and np.isfinite(shared)
  assert all(torch.equal(p.detach().cpu(),initial['parameters'][k][n]) for k,m in mods.items() for n,p in m.named_parameters() if p.requires_grad)
  restore(initial);g['save']('preflight_acceptance.json',dict(status='PASS',checks=checks,shared_auxiliary_gradient_norm=shared,optimizer_updates=0,effective_batch=16,microbatch=8,matching_RLA_parameters_and_rng=True,supervision_only=True,peak_allocated=torch.cuda.max_memory_allocated()));return
 gate=Path(plan['standard_gate'] if standard else plan['rla_gate']);assert json.loads(gate.read_text())['status']=='PASS'
 def snapshot(step):
  path=out/f'step{step:04}.pt';assert not path.exists();torch.save(dict(parameters=g['trainable_state'](),optimizer=opt.state_dict(),scheduler=sched.state_dict(),global_step=step,next_physical_sample=step*8,**rng(),configuration=plan,arm=arm,AMP_scaler=None,dtype='bfloat16',activation_checkpointing=True),path)
 vf=(out/'validation_metrics.csv').open('x');vw=csv.DictWriter(vf,fieldnames=['step','episode','timestep','appearance','action_L1','translation_L1','rotation_L1','gripper_L1','auxiliary_L1']);vw.writeheader()
 dev=[(e,t,c) for e,(_,_,meta) in ds.episodes.items() if meta['split']=='validation' for t in range(meta['valid_chunks']) for c in ('clean','background')]
 def validate(step):
  saved=rng()
  for m in mods.values():m.eval()
  with torch.inference_mode():
   for start in range(0,len(dev),8):
    ex=dev[start:start+8];b=batch(ex);loss=forward(b,'Inference');err=(capture['action'].float()-b['actions'].cuda().float()).abs();ae=torch.zeros(len(ex),device='cuda') if standard else (capture['z'].float()-b['rla'].cuda().float()).abs().mean((1,2));assert torch.isfinite(err).all() and torch.isfinite(loss)
    for i,(e,t,c) in enumerate(ex):vw.writerow(dict(step=step,episode=e,timestep=t,appearance=c,action_L1=float(err[i].mean()),translation_L1=float(err[i,:,:3].mean()),rotation_L1=float(err[i,:,3:6].mean()),gripper_L1=float(err[i,:,6].mean()),auxiliary_L1=float(ae[i])))
  vf.flush();os.fsync(vf.fileno());capture.clear()
  for m in mods.values():m.train()
  restore(saved)
 start_step=0
 if os.environ.get('POLICY_RESUME'):
  ck=torch.load(os.environ['POLICY_RESUME'],map_location='cpu');assert ck['configuration']==plan and ck['arm']==arm;start_step=ck['global_step'];assert ck['next_physical_sample']==start_step*8
  for k,m in mods.items():full=m.state_dict();full.update(ck['parameters'][k]);m.load_state_dict(full,strict=True)
  opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler']);restore(ck);g['save']('resume_provenance.json',dict(source=os.environ['POLICY_RESUME'],step=start_step,full_state=True));del ck
 snapshot(start_step);validate(start_step);del initial
 f=(out/'training_metrics.csv').open('x');w=csv.DictWriter(f,fieldnames=['step','action_loss','auxiliary_loss','weighted_auxiliary','total_loss','lr','gradient_vla','gradient_head','gradient_proprio','seconds','successful_updates']);w.writeheader();trace=(out/'actual_sample_ids.jsonl').open('x');started=time.time()
 for step in range(start_step+1,601):
  examples=[(e,t,c) for e,t in order[(step-1)*8:step*8] for c in ('clean','background')];opt.zero_grad(set_to_none=True);acts=[];auxs=[];tot=[]
  for start in [0,8]:
   b=batch(examples[start:start+8]);loss=forward(b);assert torch.isfinite(loss)
   metrics={'action_l1':float(loss.detach())} if standard else official_loss.last_metrics();acts.append(metrics['action_l1']);auxs.append(0. if standard else float((capture['z']-b['rla'].to('cuda',dtype=capture['z'].dtype)).abs().mean().detach()));tot.append(float(loss.detach()));(loss/2).backward();capture.clear();del loss,b
  norms={}
  for k,m in mods.items():
   gs=[p.grad.detach().float() for p in m.parameters() if p.grad is not None];assert all(torch.isfinite(x).all() for x in gs);norms[k]=float(torch.stack([x.norm().square() for x in gs]).sum().sqrt())
  lr=opt.param_groups[0]['lr'];opt.step();sched.step();assert all(torch.isfinite(p).all() for p in g['params'])
  w.writerow(dict(step=step,action_loss=np.mean(acts),auxiliary_loss=np.mean(auxs),weighted_auxiliary=lam*np.mean(auxs),total_loss=np.mean(tot),lr=lr,gradient_vla=norms['vla'],gradient_head=norms['head'],gradient_proprio=norms['proprio'],seconds=time.time()-started,successful_updates=step));f.flush();trace.write(json.dumps(dict(step=step,successful_update=True,actual_sample_ids=examples))+'\n');trace.flush();torch.cuda.empty_cache()
  if step%25==0:print('POLICY',arm,step,flush=True);assert g['counts'](g['before'])==g['counts'](g['ecc']())
  if step in [100,300,600]:snapshot(step);validate(step)
 f.close();vf.close();trace.close();ck=torch.load(out/'step0600.pt',map_location='cpu');assert ck['global_step']==600
 for k,m in mods.items():
  full=m.state_dict();full.update(ck['parameters'][k]);m.load_state_dict(full,strict=True)
  assert all(torch.equal(p.detach().cpu(),ck['parameters'][k][n]) for n,p in m.named_parameters() if p.requires_grad)
 actual=[json.loads(x)['actual_sample_ids'] for x in (out/'actual_sample_ids.jsonl').read_text().splitlines()];assert actual==[[[e,t,c] for e,t in order[j:j+8] for c in ('clean','background')] for j in range(start_step*8,4800,8)]
 g['save']('training_acceptance.json',dict(status='PASS_PROCESS_NOT_METHOD',successful_updates=600,strict_reload=True,actual_sample_IDs_verified=True,peak_allocated=torch.cuda.max_memory_allocated(),AE_updates=0))
