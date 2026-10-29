"""Historical official policy graph and losses; only target and lambda schedule change."""
import os,time,random,json,csv,hashlib,dataclasses,pickle
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from stove_centering.training import initialize,restore
from rla_scratch_policy_aux.common import read,sha,weight,LAMBDA,ARMS

def execute(g):
 plan,_,initial=initialize(g);mods=g['mods'];model,head,prop=mods['vla'],mods['head'],mods['proprio'];out=g['o'];arm=os.environ['POLICY_ARM'];preflight=arm=='preflight'
 assert initial['global_step']==0 and sha(plan['common_step0'])==plan['common_step0_sha256']
 assert read(Path(plan['targets']).parent/'target_acceptance.json')['status']=='PASS'
 from coverage_gripper.dataset import DerivedEpisodeDataset
 from rla.data import RlaCollator
 import rla.loss as official
 official.CFG=dataclasses.replace(official.CFG,lambda_auto=False,lambda_act=1.,lambda_rla=LAMBDA)
 collator=RlaCollator(2048,g['tokenizer'].pad_token_id);ds=DerivedEpisodeDataset(plan['inputs']['data_manifest'],plan['inputs']['action_proprio_statistics'],g['transform'],collator)
 stats={a:read(Path(plan['targets'])/a/'statistics.json') for a in ['C','D']}
 arrays={(a,e,c):np.load(Path(plan['targets'])/a/c/(e+'.npy'),mmap_mode='r') for a in ['C','D'] for e in ds.episodes for c in ['clean','background']}
 means={a:np.array(s['mean'],np.float32) for a,s in stats.items()};scales={a:np.float32(s['global_scale']) for a,s in stats.items()}
 order=read(plan['inputs']['B20_sampling_order']);assert len(order)==4800 and sha(plan['inputs']['B20_sampling_order'])==plan['inputs']['B20_sampling_sha256']
 expected={r['step']:r['actual_sample_ids'] for r in map(json.loads,Path(plan['paired_C_sample_trace']).read_text().splitlines())}
 target_arm='C' if preflight or arm=='C-fixed' else 'D';micro=8;accum=2;capture={}
 original_predict=head.predict_action
 def predict(*a,**kw):
  value=original_predict(*a,**kw);capture['action']=value;return value
 head.predict_action=predict
 original_pop=head.pop_z_hat
 def pop():
  value=original_pop();capture['z']=value;return value
 head.pop_z_hat=pop
 def orig(**kw):
  value,metrics=g['ft'].run_forward_pass(**kw);capture['act_loss']=value;return value,metrics
 wrapped=official.make_run_forward_pass(orig)
 def hook(module,args,result):
  if isinstance(module,torch.nn.L1Loss) and args[0].shape[-2:]==(8,1024):capture['aux_loss']=result
 loss_hook=torch.nn.modules.module.register_module_forward_hook(hook)
 def batch(examples,source=target_arm):
  items=[]
  for e,t,c in examples:
   data=ds.sample(e,t,c);assert 0<=t<ds.episodes[e][2]['valid_chunks']
   data['rla']=((arrays[source,e,c][t].astype(np.float32)-means[source])/scales[source]).reshape(8,1024);assert np.isfinite(data['rla']).all();items.append(data)
  b=collator(items);b['labels']=b['labels'].cuda();b['proprio']=b['proprio'].cuda()
  assert not b['rla'].requires_grad and int(model._process_action_masks(b['labels']).sum())==len(examples)*g['NUM_TOKENS']
  for i,(e,t,c) in enumerate(examples):assert ds.episodes[e][2]['task'].lower() in g['tokenizer'].decode(b['input_ids'][i].tolist()).lower()
  return b
 def forward(b,lam,phase='Training'):
  capture.clear();official.CFG=dataclasses.replace(official.CFG,lambda_rla=lam)
  value=wrapped(vla=model,action_head=SimpleNamespace(module=head),proprio_projector=prop,batch=b,action_tokenizer=g['at'],device_id='cuda',use_l1_regression=True,use_proprio=True,use_film=False,num_patches=model.vision_backbone.get_num_patches()*2,use_pro_version=True,cfg=SimpleNamespace(phase=phase))[0]
  assert capture['z'].shape==(len(b['actions']),8,1024)
  assert torch.isfinite(value) and torch.isfinite(capture['act_loss']) and torch.isfinite(capture['aux_loss'])
  return value
 from torch.utils.checkpoint import checkpoint
 model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
 for block in head.model.mlp_resnet_blocks:
  original=block.forward
  def checkpointed(*a,_original=original,**kw):return checkpoint(_original,*a,use_reentrant=False,preserve_rng_state=True,**kw)
  block.forward=checkpointed
 for visual in (model.vision_backbone.featurizer,model.vision_backbone.fused_featurizer):
  if hasattr(visual,'set_grad_checkpointing'):visual.set_grad_checkpointing(True)
 opt=torch.optim.AdamW(g['params'],lr=2e-4,weight_decay=.01,betas=(.9,.999),eps=1e-8)
 scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[600],gamma=.1)
 def rng():return dict(torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),numpy_rng=np.random.get_state())
 def rng_hash():
  s=rng();h=hashlib.sha256(pickle.dumps((s['python_rng'],s['numpy_rng'])))
  for x in [s['torch_rng']]+s['cuda_rng']:h.update(x.cpu().numpy().tobytes())
  return h.hexdigest()
 def digest():
  h=hashlib.sha256()
  for k,m in mods.items():
   for n,p in m.named_parameters():
    if p.requires_grad:h.update((k+'.'+n).encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
  return h.hexdigest()
 restore(initial)
 assert all(torch.equal(p.detach().cpu(),initial['parameters'][k][n]) for k,m in mods.items() for n,p in m.named_parameters() if p.requires_grad)
 init_hash=digest();initial_rng_hash=rng_hash()
 g['save']('initialization_audit.json',dict(status='PASS',common_step0_sha256=plan['common_step0_sha256'],parameter_sha256=init_hash,RNG_hash=initial_rng_hash,all_trainable_parameters_exact=True,no_auxiliary_reinitialization=True,no_trained_policy_loaded=True,sampling_hash=sha(plan['inputs']['B20_sampling_order'])))
 g['save']('execution_configuration.json',dict(arm=arm,backend={'math':torch.backends.cuda.math_sdp_enabled(),'flash':torch.backends.cuda.flash_sdp_enabled(),'efficient':torch.backends.cuda.mem_efficient_sdp_enabled()},dtype='bfloat16',microbatch=8,accumulation=2,effective_appearance_batch=16,activation_checkpointing=True,lambda0=LAMBDA,target=target_arm,lambda_schedule='successful update k0..599; restore global_step',optimizer='AdamW',lr=2e-4,betas=[.9,.999],eps=1e-8,weight_decay=.01,scheduler='MultiStepLR600 gamma0.1; optimizer then scheduler; all600 actual update LR2e-4',gradient_clipping=None,AE_updates=0))
 fixed=[(e,t,c) for e,t in order[:8] for c in ['clean','background']]
 g['save']('fixed_diagnostic_batch.json',fixed)
 scope=[]
 for k,m in mods.items():
  for n,p in m.named_parameters():
   if p.requires_grad and (k!='head' or (n.startswith('model.mlp_resnet_blocks.') and 'gating_factor' not in n)):scope.append((k+'.'+n,p))
 g['save']('gradient_scope.json',dict(parameters=[dict(name=n,numel=p.numel()) for n,p in scope],rule='VLA/proprio/shared head trunk only, further intersect actual per-objective autograd support; exclude final readouts, action-only fc1/layernorm1 and head position-specific gates/one-way bridge gates; zero gradients reported rather than assumed active',units='unscaled raw gradients, not AdamW update ratio',near_zero=1e-12))
 diag_rows=[]
 def objective(source,kind,lam,saved):
  restore(saved);vectors=[None]*len(scope);outputs=[];losses=[];inputs=hashlib.sha256()
  for start in [0,8]:
   b=batch(fixed[start:start+8],source)
   for key in ['pixel_values','input_ids','labels','proprio','actions']:inputs.update(b[key].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
   total=forward(b,lam);obj=capture['act_loss'] if kind=='action' else capture['aux_loss']
   vs=torch.autograd.grad(obj/accum,[p for n,p in scope],allow_unused=True)
   for i,v in enumerate(vs):
    if v is not None:
     assert torch.isfinite(v).all();v=v.detach().cpu().float();vectors[i]=v if vectors[i] is None else vectors[i]+v
   outputs.append(capture['action'].detach().cpu());losses.append([float(capture['act_loss'].detach()),float(capture['aux_loss'].detach())]);capture.clear();del total,obj,vs,b
  after=rng_hash();restore(saved)
  return vectors,np.mean(losses,axis=0).tolist(),torch.cat(outputs),inputs.hexdigest(),after
 def diagnostic(step,source,lam):
  saved=rng();before=digest();buffers={(k,n):b.detach().cpu().clone() for k,m in mods.items() for n,b in m.named_buffers()};states={k:m.training for k,m in mods.items()}
  for m in mods.values():m.train()
  va,la,oa,ia,ra=objective(source,'action',lam,saved);vz,lz,oz,iz,rz=objective(source,'auxiliary',lam,saved)
  assert torch.equal(oa,oz) and la==lz and ia==iz and ra==rz
  for module in ['ALL_SHARED','vla','proprio','head']:
   aa=bb=ab=0.;num=0;unused=[]
   for (name,p),a,z in zip(scope,va,vz):
    if module!='ALL_SHARED' and not name.startswith(module+'.'):continue
    if a is None or z is None:unused.append(name);continue
    a=a.double();z=z.double();aa+=float(a.square().sum());bb+=float(z.square().sum());ab+=float((a*z).sum());num+=a.numel()
   an=aa**.5;zn=bb**.5
   diag_rows.append(dict(step=step,target=source,module=module,action_L1=la[0],auxiliary_L1=la[1],lambda_RLA=lam,weighted_auxiliary_L1=lam*la[1],action_gradient_norm=an,raw_auxiliary_gradient_norm=zn,weighted_auxiliary_gradient_norm=lam*zn,weighted_aux_to_action_ratio=lam*zn/an if an>1e-12 else None,raw_cosine=ab/(an*zn) if min(an,zn)>1e-12 else None,weighted_cosine=ab/(an*zn) if min(an,zn)>1e-12 and lam>0 else None,joint_parameter_elements=num,unused_parameters=unused))
  del va,vz,oa,oz
  assert before==digest()
  assert all(torch.equal(b.detach().cpu(),buffers[k,n]) for k,m in mods.items() for n,b in m.named_buffers())
  for k,m in mods.items():m.train(states[k])
  restore(saved);g['save'](f'gradient_step{step:04}_{source}.json',dict(statistics=[r for r in diag_rows if r['step']==step and r['target']==source],parameter_buffer_RNG_unchanged=True,forward_microbatches=4,objective_gradient_extractions=4,full_effective_objectives=2,optimizer_updates=0,matched_random_conditions=True))
  opt.zero_grad(set_to_none=True);capture.clear();torch.cuda.empty_cache()
 def snapshot(step):
  p=out/f'step{step:04}.pt';assert not p.exists()
  torch.save(dict(parameters=g['trainable_state'](),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),global_step=step,next_physical_sample=step*8,**rng(),configuration=plan,arm=arm,lambda_next=None if preflight else weight(arm,min(step,599)),AMP_scaler=None,dtype='bfloat16',activation_checkpointing=True,auxiliary_schedule_counter=step),p)
  return p
 def reload(path):
  before=digest();saved=rng();ck=torch.load(path,map_location='cpu');step=ck['global_step'];assert ck['auxiliary_schedule_counter']==step and ck['next_physical_sample']==step*8
  for k,m in mods.items():full=m.state_dict();full.update(ck['parameters'][k]);m.load_state_dict(full,strict=True)
  opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);restore(ck)
  assert before==digest() and torch.equal(saved['torch_rng'],torch.get_rng_state()) and all(torch.equal(x,y) for x,y in zip(saved['cuda_rng'],torch.cuda.get_rng_state_all()))
  g['save'](f'reload_step{step:04}.json',dict(strict=True,global_step=step,parameters_hash=before,optimizer_steps=sorted(set(int(v['step']) for v in opt.state.values())),scheduler_last_epoch=scheduler.last_epoch,next_sample=step*8,AMP_scaler=None,bfloat16=True))
 if preflight:
  checks=[];moments=[torch.zeros_like(p) for p in g['params'] for _ in range(2)]
  for source in ['C','D']:
   restore(initial);opt.zero_grad(set_to_none=True);hh=hashlib.sha256();ih=hashlib.sha256();losses=[]
   for start in [0,8]:
    b=batch(fixed[start:start+8],source);value=forward(b,0.)
    for key in ['pixel_values','input_ids','labels','proprio','actions']:ih.update(b[key].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    hh.update(capture['action'].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes());losses.append(float(value.detach()));(value/accum).backward();capture.clear();del b,value
   gh=hashlib.sha256()
   for k,m in mods.items():
    for n,p in m.named_parameters():
     if p.grad is not None:assert torch.isfinite(p.grad).all();gh.update((k+n).encode());gh.update(p.grad.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
   checks.append(dict(source=source,action_hash=hh.hexdigest(),input_hash=ih.hexdigest(),gradient_hash=gh.hexdigest(),losses=losses))
  assert checks[0]['action_hash']==checks[1]['action_hash'] and checks[0]['gradient_hash']==checks[1]['gradient_hash'] and checks[0]['input_hash']==checks[1]['input_hash'] and checks[0]['losses']==checks[1]['losses']
  opt.zero_grad(set_to_none=True);del moments
  for source in ['C','D']:restore(initial);diagnostic(0,source,LAMBDA)
  assert all(r['raw_auxiliary_gradient_norm']>0 for r in diag_rows if r['module']=='ALL_SHARED')
  restore(initial);assert digest()==init_hash and rng_hash()==initial_rng_hash
  reload(snapshot(0));g['save']('preflight_acceptance.json',dict(status='PASS',checks=checks,optimizer_updates=0,AE_updates=0,full_effective_batch=16,microbatch=8,accumulation=2,peak_allocated=torch.cuda.max_memory_allocated(),no_target_in_action_path=True,mean_loss_over_all_valid_8x1024=True,initialization_restored=True,strict_zero_checkpoint_reload=True));loss_hook.remove();return
 assert read(out.parent.parent/'preflight/formal/preflight_acceptance.json')['status']=='PASS'
 start_step=0
 if os.environ.get('POLICY_RESUME'):
  ck=torch.load(os.environ['POLICY_RESUME'],map_location='cpu');assert ck['configuration']==plan and ck['arm']==arm;start_step=ck['global_step'];assert ck['auxiliary_schedule_counter']==start_step and ck['next_physical_sample']==start_step*8
  for k,m in mods.items():full=m.state_dict();full.update(ck['parameters'][k]);m.load_state_dict(full,strict=True)
  opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);restore(ck);del ck
  g['save']('resume_provenance.json',dict(source=os.environ['POLICY_RESUME'],step=start_step,all_model_optimizer_scheduler_RNG_sampling_restored=True,schedule_not_reset=True))
 vf=(out/'validation_metrics.csv').open('x');vw=csv.DictWriter(vf,fieldnames=['step','episode','timestep','appearance','action_L1','translation_L1','rotation_L1','gripper_L1','auxiliary_L1','per_dimension_L1','per_chunk_position_L1']);vw.writeheader()
 dev=[(e,t,c) for e,(_,_,m) in ds.episodes.items() if m['split']=='validation' for t in range(m['valid_chunks']) for c in ['clean','background']]
 def validate(step):
  saved=rng();lam=weight(arm,min(step,599))
  for m in mods.values():m.eval()
  with torch.inference_mode():
   for start in range(0,len(dev),micro):
    ex=dev[start:start+micro];b=batch(ex);loss=forward(b,lam,'Inference');err=(capture['action'].float()-b['actions'].cuda().float()).abs();zerr=(capture['z'].float()-b['rla'].cuda().float()).abs()
    assert torch.isfinite(err).all() and torch.isfinite(zerr).all()
    for i,(e,t,c) in enumerate(ex):vw.writerow(dict(step=step,episode=e,timestep=t,appearance=c,action_L1=float(err[i].mean()),translation_L1=float(err[i,:,:3].mean()),rotation_L1=float(err[i,:,3:6].mean()),gripper_L1=float(err[i,:,6].mean()),auxiliary_L1=float(zerr[i].mean()),per_dimension_L1=json.dumps(err[i].mean(0).cpu().tolist()),per_chunk_position_L1=json.dumps(err[i].mean(1).cpu().tolist())))
    capture.clear();del b,loss,err,zerr
  vf.flush();os.fsync(vf.fileno())
  for m in mods.values():m.train()
  restore(saved)
 reload(snapshot(start_step));validate(start_step);diagnostic(start_step,target_arm,weight(arm,min(start_step,599)));del initial
 fields=['step','action_loss','auxiliary_loss','lambda_RLA','weighted_auxiliary','total_loss','lr','gradient_vla','gradient_head','gradient_proprio','seconds','successful_updates','peak_allocated']
 f=(out/'training_metrics.csv').open('x');w=csv.DictWriter(f,fieldnames=fields);w.writeheader();trace=(out/'actual_sample_ids.jsonl').open('x');started=time.time()
 for step in range(start_step+1,601):
  lam=weight(arm,step-1);examples=[(e,t,c) for e,t in order[(step-1)*8:step*8] for c in ['clean','background']];assert [list(x) for x in examples]==expected[step]
  opt.zero_grad(set_to_none=True);acts=[];aux=[];totals=[]
  for start in [0,8]:
   b=batch(examples[start:start+8]);loss=forward(b,lam);acts.append(float(capture['act_loss'].detach()));aux.append(float(capture['aux_loss'].detach()));totals.append(float(loss.detach()));(loss/accum).backward();capture.clear();del loss,b
  norms={}
  for k,m in mods.items():
   grads=[p.grad.detach().float() for p in m.parameters() if p.grad is not None];assert all(torch.isfinite(x).all() for x in grads);norms[k]=float(torch.stack([x.norm().square() for x in grads]).sum().sqrt())
  lr=opt.param_groups[0]['lr'];assert lr==2e-4
  opt.step();scheduler.step();assert all(torch.isfinite(p).all() for p in g['params'])
  w.writerow(dict(step=step,action_loss=np.mean(acts),auxiliary_loss=np.mean(aux),lambda_RLA=lam,weighted_auxiliary=lam*np.mean(aux),total_loss=np.mean(totals),lr=lr,gradient_vla=norms['vla'],gradient_head=norms['head'],gradient_proprio=norms['proprio'],seconds=time.time()-started,successful_updates=step,peak_allocated=torch.cuda.max_memory_allocated()));f.flush();trace.write(json.dumps(dict(step=step,successful_update=True,actual_sample_ids=examples))+'\n');trace.flush();torch.cuda.empty_cache()
  if step%25==0:print('POLICY',arm,step,'action',np.mean(acts),'aux',np.mean(aux),'lambda',lam,'seconds',time.time()-started,flush=True);assert g['counts'](g['before'])==g['counts'](g['ecc']())
  if step in [1,100,300,600]:p=snapshot(step)
  if step in [100,300,600]:validate(step)
  if step in [300,600]:opt.zero_grad(set_to_none=True);diagnostic(step,target_arm,weight(arm,min(step,599)))
 f.close();vf.close();trace.close();reload(out/'step0600.pt');loss_hook.remove()
 g['save']('training_acceptance.json',dict(status='PASS_TRAINING_NOT_METHOD',successful_updates=600,new_successful_updates=600-start_step,actual_update_computations=600-start_step,replayed_updates=0,strict_reload=True,seconds=time.time()-started,peak_allocated=torch.cuda.max_memory_allocated(),physical_exposures=4800,appearance_exposures=9600,AE_updates=0))
