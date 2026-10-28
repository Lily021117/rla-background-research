"""Architecture-matched BC versus 12-coordinate geometry supervision.

Only the final independent auxiliary output is sliced. No target enters forward.
"""
import os,json,csv,random,time,hashlib,dataclasses
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from stove_centering.training import initialize,restore

def execute(g):
 plan,prev,initial=initialize(g);mods=g['mods'];model,head,prop=mods['vla'],mods['head'],mods['proprio'];out=g['o'];arm=os.environ['POLICY_ARM'];micro=8;accum=2
 from coverage_gripper.dataset import DerivedEpisodeDataset
 from rla.data import RlaCollator
 collator=RlaCollator(2048,g['tokenizer'].pad_token_id);ds=DerivedEpisodeDataset(plan['inputs']['data_manifest'],plan['inputs']['action_proprio_statistics'],g['transform'],collator)
 geomroot=Path(plan['geometry_labels']);stats=json.loads((geomroot/'statistics.json').read_text());mu=np.array(stats['mean']);scale=np.array(stats['denominator']);labels={e:dict(np.load(geomroot/(e+'.npz'))) for e in ds.episodes}
 order=json.loads(Path(plan['inputs']['B20_sampling_order']).read_text());assert len(order)==4800
 expected={r['step']:r['actual_sample_ids'] for r in map(json.loads,Path(plan['paired_C_sample_trace']).read_text().splitlines())}
 lam=0. if arm=='Action-only-match' else plan['shared_protocol']['lambda_aux']
 capture={};original_predict=head.predict_action
 def predict(*args,**kw):
  action=original_predict(*args,**kw);capture['action']=action;return action
 head.predict_action=predict
 def batch(examples):
  items=[];masks=[]
  for e,t,c in examples:
   item=ds.sample(e,t,c);item['rla']=((labels[e]['raw'][t]-mu)/scale).astype(np.float32);assert item['rla'].shape==(8,12);items.append(item);masks.append(labels[e]['mask'][t])
  b=collator(items);b['labels']=b['labels'].cuda();b['proprio']=b['proprio'].cuda();b['geometry_mask']=torch.from_numpy(np.stack(masks)).cuda()
  assert int(model._process_action_masks(b['labels']).sum())==len(examples)*g['NUM_TOKENS'];assert b['geometry_mask'].all()
  for i,(e,t,c) in enumerate(examples):assert ds.episodes[e][2]['task'].lower() in g['tokenizer'].decode(b['input_ids'][i].tolist()).lower()
  return b
 def action_forward(b):
  return g['ft'].run_forward_pass(vla=model,action_head=SimpleNamespace(module=head),proprio_projector=prop,batch=b,action_tokenizer=g['at'],device_id='cuda',use_l1_regression=True,use_proprio=True,use_film=False,num_patches=model.vision_backbone.get_num_patches()*2,use_pro_version=True,cfg=SimpleNamespace(phase='Training' if model.training else 'Inference'))[0]
 def forward(b,weight=lam):
  capture.clear();act=action_forward(b);z=head.pop_z_hat();assert z.shape==(len(b['actions']),8,1024)
  target=b['rla'].cuda();assert not target.requires_grad;pred=z[:,:,:12].float();mask=b['geometry_mask'];error=(pred-target.float()).abs();aux=error[mask].mean();capture.update(action_loss=act,aux=aux,z=z,geometry_error=error)
  return act+weight*aux
 from torch.utils.checkpoint import checkpoint
 model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
 for block in head.model.mlp_resnet_blocks:
  original=block.forward
  def checkpointed(*args,_original=original,**kwargs):return checkpoint(_original,*args,use_reentrant=False,preserve_rng_state=True,**kwargs)
  block.forward=checkpointed
 for visual in (model.vision_backbone.featurizer,model.vision_backbone.fused_featurizer):
  if hasattr(visual,'set_grad_checkpointing'):visual.set_grad_checkpointing(True)
 opt=torch.optim.AdamW(g['params'],lr=2e-4,weight_decay=.01);scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[600],gamma=.1)
 def digest_grad():
  h=hashlib.sha256()
  for k,m in mods.items():
   for n,p in m.named_parameters():
    if p.grad is not None:h.update((k+n).encode());h.update(p.grad.detach().cpu().view(torch.uint8).numpy().tobytes())
  return h.hexdigest()
 def params_equal():return all(torch.equal(p.detach().cpu(),initial['parameters'][k][n]) for k,m in mods.items() for n,p in m.named_parameters() if p.requires_grad)
 def rng():return dict(torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),numpy_rng=np.random.get_state())
 restore(initial);assert params_equal()
 g['save']('initialization_audit.json',dict(status='PASS',all_trainable_parameters_equal_common_step0=True,common_step0_sha256=plan['common_step0_sha256'],RNG_restored=True,auxiliary_head_reinitialized=False,final_auxiliary_readout='head.model.fc2_rla(layer_norm2_rla(x_rla)); reshape8x1024 then slice[:12]; not consumed by action',lambda_aux=lam,geometry_FP32_mean=True))
 if arm=='preflight':
  import rla.loss as official_loss
  from stove_centering.store import CenterStore
  official_loss.CFG=dataclasses.replace(official_loss.CFG,lambda_auto=False,lambda_rla=0.)
  stores={c:CenterStore(str(prev/'sidecars/C'/c),chunk=8,num_tokens=16,token_dim=64,norm='center') for c in ('clean','background')}
  slots={e:{c:stores[c].slot(np.load(root/'trajectory_labels.npz')['proprio']) for c in stores} for e,(root,_,_) in ds.episodes.items()}
  examples=[(e,t,c) for e,t in order[:8] for c in ('clean','background')];checks=[];moments=[torch.zeros_like(p) for p in g['params'] for _ in range(2)]
  for mode in ('official_zero','geometry_zero','changed_target_zero'):
   restore(initial);opt.zero_grad(set_to_none=True);h=hashlib.sha256();losses=[]
   for start in (0,8):
    ex=examples[start:start+8];b=batch(ex)
    for i in range(0,8,2):assert torch.equal(b['rla'][i],b['rla'][i+1]);assert ex[i][:2]==ex[i+1][:2]
    if mode=='official_zero':
     b['rla']=torch.from_numpy(np.stack([stores[c].frame(slots[e][c],t) for e,t,c in ex]));loss=official_loss.make_run_forward_pass(g['ft'].run_forward_pass)(vla=model,action_head=SimpleNamespace(module=head),proprio_projector=prop,batch=b,action_tokenizer=g['at'],device_id='cuda',use_l1_regression=True,use_proprio=True,use_film=False,num_patches=model.vision_backbone.get_num_patches()*2,use_pro_version=True,cfg=SimpleNamespace(phase='Training'))[0]
    else:
     if mode=='changed_target_zero':b['rla']=b['rla']+7
     loss=forward(b,0.)
    h.update(capture['action'].detach().cpu().view(torch.uint8).numpy().tobytes());losses.append(float(loss.detach()));(loss/2).backward();capture.clear();del loss,b
   checks.append(dict(mode=mode,action_hash=h.hexdigest(),gradient_hash=digest_grad(),losses=losses))
  assert len({r['action_hash'] for r in checks})==1 and len({r['gradient_hash'] for r in checks})==1 and all(r['losses']==checks[0]['losses'] for r in checks)
  restore(initial);opt.zero_grad(set_to_none=True);b=batch(examples[:8]);total=forward(b);err=capture['geometry_error'];assert torch.equal(capture['aux'],err.sum()/err.numel());(capture['aux']*lam/2).backward();sharednorm=sum(float(p.grad.float().square().sum()) for p in model.parameters() if p.grad is not None)**.5;assert sharednorm>0
  assert all(p.grad is None or torch.isfinite(p.grad).all() for p in g['params']);assert params_equal();restore(initial)
  g['save']('preflight_acceptance.json',dict(status='PASS',microbatch=8,effective_appearance_batch=16,optimizer_updates=0,checks=checks,exact_action_and_gradient_equivalence=True,geometry_shared_backbone_gradient_norm=sharednorm,mask_mean_verified=True,target_no_gradient=True,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),target_input_path='supervision only',no_zero_padding=True));return
 assert json.loads(Path(plan['geometry_gate']).read_text())['status']=='PASS'
 def snapshot(step):
  path=out/f'step{step:04}.pt';assert not path.exists();torch.save(dict(parameters=g['trainable_state'](),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),global_step=step,next_physical_sample=step*8,**rng(),configuration=plan,arm=arm,AMP_scaler=None,dtype='bfloat16',activation_checkpointing=True),path)
 vf=(out/'validation_metrics.csv').open('x');vw=csv.DictWriter(vf,fieldnames=['step','episode','timestep','appearance','action_L1','translation_L1','rotation_L1','gripper_L1','geometry_L1','geometry_MAE_m']);vw.writeheader()
 dev=[(e,t,c) for e,(_,_,m) in ds.episodes.items() if m['split']=='validation' for t in range(m['valid_chunks']) for c in ('clean','background')]
 def validate(step):
  saved=rng()
  for m in mods.values():m.eval()
  with torch.inference_mode():
   for start in range(0,len(dev),8):
    ex=dev[start:start+8];b=batch(ex);loss=forward(b);error=(capture['action'].float()-b['actions'].cuda().float()).abs();ge=capture['geometry_error'];assert torch.isfinite(loss) and torch.isfinite(error).all()
    for i,(e,t,c) in enumerate(ex):vw.writerow(dict(step=step,episode=e,timestep=t,appearance=c,action_L1=float(error[i].mean()),translation_L1=float(error[i,:,:3].mean()),rotation_L1=float(error[i,:,3:6].mean()),gripper_L1=float(error[i,:,6].mean()),geometry_L1=float(ge[i].mean()),geometry_MAE_m=float((ge[i]*torch.as_tensor(scale,device='cuda')).mean())))
  vf.flush();os.fsync(vf.fileno());capture.clear()
  for m in mods.values():m.train()
  restore(saved)
 start_step=0
 if os.environ.get('POLICY_RESUME'):
  ck=torch.load(os.environ['POLICY_RESUME'],map_location='cpu');assert ck['configuration']==plan and ck['arm']==arm;start_step=ck['global_step'];assert ck['next_physical_sample']==start_step*8
  for k,m in mods.items():full=m.state_dict();full.update(ck['parameters'][k]);m.load_state_dict(full,strict=True)
  opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);restore(ck);g['save']('resume_provenance.json',dict(source=os.environ['POLICY_RESUME'],step=start_step,full_state=True));del ck
 snapshot(start_step);validate(start_step);del initial
 f=(out/'training_metrics.csv').open('x');fields=['step','action_loss','auxiliary_loss','weighted_auxiliary','total_loss','lr','gradient_vla','gradient_head','gradient_proprio','seconds','successful_updates'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();trace=(out/'actual_sample_ids.jsonl').open('x');started=time.time()
 for step in range(start_step+1,601):
  examples=[(e,t,c) for e,t in order[(step-1)*8:step*8] for c in ('clean','background')];assert [list(x) for x in examples]==expected[step]
  opt.zero_grad(set_to_none=True);act=[];aux=[];tot=[]
  for i in (0,8):
   b=batch(examples[i:i+8]);loss=forward(b);assert torch.isfinite(loss);act.append(float(capture['action_loss'].detach()));aux.append(float(capture['aux'].detach()));tot.append(float(loss.detach()));(loss/2).backward();capture.clear();del loss,b
  norms={}
  for k,m in mods.items():
   grads=[p.grad.detach().float() for p in m.parameters() if p.grad is not None];assert all(torch.isfinite(x).all() for x in grads);norms[k]=float(torch.stack([x.norm().square() for x in grads]).sum().sqrt())
  lr=opt.param_groups[0]['lr'];opt.step();scheduler.step();assert all(torch.isfinite(p).all() for p in g['params'])
  w.writerow(dict(step=step,action_loss=np.mean(act),auxiliary_loss=np.mean(aux),weighted_auxiliary=lam*np.mean(aux),total_loss=np.mean(tot),lr=lr,gradient_vla=norms['vla'],gradient_head=norms['head'],gradient_proprio=norms['proprio'],seconds=time.time()-started,successful_updates=step));f.flush();trace.write(json.dumps(dict(step=step,successful_update=True,actual_sample_ids=examples))+'\n');trace.flush();torch.cuda.empty_cache()
  if step%25==0:print('POLICY',arm,step,'action',np.mean(act),'geometry',np.mean(aux),flush=True);assert g['counts'](g['before'])==g['counts'](g['ecc']())
  if step in (1,100,300,600):snapshot(step)
  if step in (100,300,600):validate(step)
 f.close();vf.close();trace.close();ck=torch.load(out/'step0600.pt',map_location='cpu');assert ck['global_step']==600
 for k,m in mods.items():
  full=m.state_dict();full.update(ck['parameters'][k]);m.load_state_dict(full,strict=True)
  for n,p in m.named_parameters():
   if p.requires_grad:assert torch.equal(p.detach().cpu(),ck['parameters'][k][n])
 g['save']('training_acceptance.json',dict(status='PASS_TRAINING_NOT_METHOD',successful_updates=600,strict_reload=True,seconds=time.time()-started,peak_allocated=torch.cuda.max_memory_allocated(),physical_exposures=4800,appearance_exposures=9600,AE_updates=0))
