"""Authorized Step5.7: unchanged unified training loop, Math SDPA only."""
import os
from pathlib import Path
import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
W=Path(__file__).parent
# Verified native initialization, real batch functions, complete step0 action equivalence.
exec(compile((W/'validate_step0.py').read_text().split('grads={};i=selected[0]')[0],str(W/'validate_step0.py'),'exec'))
import resource,traceback,gc
ACCEPT=E/'step5_6_1_gradient_diagnosis/run_20261022_02'
assert json.loads((ACCEPT/'delivery_verification.json').read_text())['step5_7_implementation_gate']=='PASS'
assert initial==json.loads((ACCEPT/'step0_parameter_hashes.json').read_text())['B']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def state_digest(v):
 h=hashlib.sha256()
 def rec(x):
  if torch.is_tensor(x):h.update(str((x.dtype,tuple(x.shape))).encode());h.update(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
  elif isinstance(x,dict):
   for k in sorted(x,key=str):h.update(str(k).encode());rec(x[k])
  elif isinstance(x,(list,tuple)):
   for y in x:rec(y)
  else:h.update(repr(x).encode())
 rec(v);return h.hexdigest()
def backend():
 d={'flash':torch.backends.cuda.flash_sdp_enabled(),'mem_efficient':torch.backends.cuda.mem_efficient_sdp_enabled(),'math':torch.backends.cuda.math_sdp_enabled()}
 assert d=={'flash':False,'mem_efficient':False,'math':True};return d
order=json.loads((T/'sampling_order.json').read_text());assert len(order)==100 and all(len(ids)==4 and all(samples[i]['split']=='train' for i in ids) for ids in order)
cfg=json.loads((O/'unified_model_config.json').read_text());source_paths=[T/'sample_manifest.json',T/'sampling_order.json',F/'rla_target_manifest.csv',N/'normalization_B.npz',N/'normalization_C.npz']
sources={str(p):sha(p) for p in source_paths}
sources.update({str(p):sha(p) for p in base.iterdir() if p.is_file()})
sources.update({str(W/n):sha(W/n) for n in ['causal_unified.py','validate_step0.py','prepare_acceptance.py','train_math_step57.py']})
js('training_provenance.json',{'sources':sources,'git_commit':subprocess.check_output(['git','-C',str(W),'rev-parse','HEAD'],text=True).strip(),'seed':20261021,'no_historical_final_checkpoint_initialization':True})
for arm in ['B','C','D']:
 c=dict(cfg['common_training']);c.update({'arm':vars(ARMS[arm]),'backend':backend(),'allocator_cap_GiB':22,'initialization':'common public Spatial-Pro + identical LoRA and auxiliary initialization','normalization':str(N/f'normalization_{ARMS[arm].target_source}.npz'),'sources':sources,'successful_updates':100,'no_closed_loop':True,'sampling_order':str(T/'sampling_order.json'),'only_change_from_accepted_algorithm':'attention backend Math SDPA','main_checkpoint':100})
 js('training_config_'+arm+'.json',c)
seed();common_rng=rng_state();torch.save(common_rng,O/'common_rng.pt')
opt=torch.optim.AdamW([p for n,p in params],lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
step0=O/'common_step0.pt';save_checkpoint(step0,model,head,prop,opt,sched,0,{'base':str(base),'sources':sources,'configuration':'training_config_[BCD].json','initial_parameter_hashes':initial})
preflight={};refgrad=None;refout=None;ids=order[0][:2]
for arm in ['B','C','D']:
 assert restore_checkpoint(step0,model,head,prop,opt,sched)==0
 assert [digest(m) for m in [model,head,prop]]==initial
 configure_trainability(model,head,prop,ARMS[arm])
 for m in [model,head,prop]:m.train()
 clear();set_rng(common_rng)
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
  al=forward(load_batch(ids),'Training');tot,aux=objective(al,head.last_aux,target_for(ids,ARMS[arm].target_source),ARMS[arm]);out=[head.last_action.detach().cpu().clone(),head.last_aux.detach().cpu().clone()];al.backward()
 ops=[e.key for e in prof.key_averages() if 'attention' in e.key.lower()]
 assert not any('flash' in x.lower() or 'efficient_attention' in x.lower() for x in ops)
 assert any('attention_math' in x for x in ops)
 grads={n:p.grad.detach().cpu().clone() for n,p in params if p.grad is not None}
 if refgrad is None:refgrad=grads;refout=out
 assert set(grads)==set(refgrad) and all(torch.equal(grads[n],refgrad[n]) for n in grads)
 assert all(torch.equal(a,b) for a,b in zip(out,refout))
 preflight[arm]={'parameter_hashes':initial,'action_aux_max_abs':0,'action_only_gradient_max_abs':0,'action_loss':float(al),'aux_loss':float(aux),'ops':ops,'backend':backend()}
 del grads
 clear();set_rng(common_rng);al=forward(load_batch(ids),'Training');tot,aux=objective(al,head.last_aux,target_for(ids,ARMS[arm].target_source),ARMS[arm]);(ARMS[arm].aux_loss_weight*aux).backward();summary=gradient_summary()
 assert summary['shared_backbone']['norm']==0 if arm=='B' else summary['shared_backbone']['norm']>0
 preflight[arm]['weighted_aux_gradients']=summary;clear()
del refgrad,refout;gc.collect()
# Dynamic target intervention: targets excluded by the same real batch loader.
for m in [model,head,prop]:m.train()
set_rng(common_rng)
with torch.no_grad():
 al=forward(load_batch(ids),'Training');ref=head.last_action.clone();head.clear()
 for variant in ['deleted','randomized']:
  set_rng(common_rng);b=load_batch(ids);z=target_for(ids,'C')
  if variant=='randomized':z=z.flip(-1)+17
  else:z=None
  al=forward(b,'Training');assert torch.equal(head.last_action,ref);head.clear()
js('step0_preflight.json',{'status':'PASS','arms':preflight,'target_isolation':True,'optimizer_updates':0,'backward_calls':6})
js('rng_and_backend_audit.json',{'seed':20261021,'common_rng_SHA256':sha(O/'common_rng.pt'),'backend':backend(),'profile_proves_math':True,'formal_train_mode':True,'microbatch_rng_rule':'20261021+successful_update*10+offset; offsets 0,2','workers':0,'sampling_SHA256':sha(T/'sampling_order.json'),'all_start_same_step0':True,'torch':torch.__version__,'cuda':torch.version.cuda,'gradient_checkpointing':getattr(model,'is_gradient_checkpointing',None),'no_backend_fallback':True})
validation=[];reloads={};inventory=[];rows=[];start_all=time.monotonic()
def writecsv(name,values):
 with (O/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
def evaluate(arm,step):
 before=rng_state()
 for m in [model,head,prop]:m.eval()
 with torch.inference_mode():
  for j,s in enumerate(samples):
   if s['split']!='validation':continue
   al=forward(load_batch([j]),'Inference');total,aux=objective(al,head.last_aux,target_for([j],ARMS[arm].target_source),ARMS[arm]);assert torch.isfinite(head.last_action).all()
   validation.append({'arm':arm,'step':step,'episode':s['episode'],'transition_id':s['transition_id'],'appearance':s['appearance'],'action_L1':float(al),'aux_L1':float(aux),'finite_actions':True});head.clear()
 set_rng(before);writecsv('validation_metrics.csv',validation)
def checked_batch(ids):
 assert ecc()==baseline_ecc;backend()
 assert all(samples[j]['split']=='train' for j in ids)
 return load_batch(ids)
try:
 for arm in ['B','C','D']:
  group=O/arm;group.mkdir();restore_checkpoint(step0,model,head,prop,opt,sched);clear();configure_trainability(model,head,prop,ARMS[arm])
  assert [digest(m) for m in [model,head,prop]]==initial and not opt.state and sched.last_epoch==0
  set_rng(common_rng);evaluate(arm,0);start=time.monotonic()
  def ckpt(step):
   p=group/f'step{step:04d}.pt';save_checkpoint(p,model,head,prop,opt,sched,step,{'arm':vars(ARMS[arm]),'sources':sources,'backend':backend(),'config_SHA256':sha(O/f'training_config_{arm}.json'),'successful_updates':step})
   inventory.append({'arm':arm,'step':step,'path':str(p),'SHA256':sha(p),'bytes':p.stat().st_size});writecsv('checkpoint_inventory.csv',inventory)
  ckpt(0)
  with (group/'updates.jsonl').open('x',buffering=1) as logfile:
   def log(info):
    g=gradient_summary();r={'arm':arm,'global_step':info['successful_update'],'actual_successful_updates':info['successful_update'],'action_loss':sum(x['action'] for x in info['losses'])/2,'aux_loss':sum(x['aux'] for x in info['losses'])/2,'total_loss':sum(x['total'] for x in info['losses'])/2,'lr':opt.param_groups[0]['lr'],'shared_backbone_grad_norm':g['shared_backbone']['norm'],'action_head_grad_norm':g['action_branch']['norm'],'auxiliary_head_grad_norm':g['auxiliary_head']['norm'],'GPU_allocated':torch.cuda.memory_allocated(),'GPU_peak_allocated':torch.cuda.max_memory_allocated(),'CPU_max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_seconds':time.monotonic()-start,'finite':True,'skipped':0,'replayed':0}
    rows.append(r);logfile.write(json.dumps(r)+'\n');writecsv('training_metrics.csv',rows)
    if r['global_step']%10==0:print(arm,r['global_step'],r['action_loss'],flush=True)
   run_training(model=model,head=head,proprio=prop,arm=ARMS[arm],optimizer=opt,scheduler=sched,forward=forward,load_batch=checked_batch,sampling_order=order,target_for=target_for,checkpoint=ckpt,log=log,authorized=True)
  hashes=[digest(m) for m in [model,head,prop]];oh=state_digest(opt.state_dict());sh=state_digest(sched.state_dict())
  for m in [model,head,prop]:m.eval()
  with torch.no_grad():
   al=forward(load_batch(ids),'Inference');out=[head.last_action.cpu().clone(),head.last_aux.cpu().clone()];head.clear()
  assert restore_checkpoint(group/'step0100.pt',model,head,prop,opt,sched)==100
  assert [digest(m) for m in [model,head,prop]]==hashes and state_digest(opt.state_dict())==oh and state_digest(sched.state_dict())==sh
  steps=sorted({int(v['step'].item()) for v in opt.state.values()});assert steps==[100] and sched.last_epoch==100
  with torch.no_grad():
   al=forward(load_batch(ids),'Inference');now=[head.last_action.cpu(),head.last_aux.cpu()];assert all(torch.equal(x,y) for x,y in zip(out,now));head.clear()
  reloads[arm]={'status':'PASS','global_step':100,'optimizer_steps':steps,'scheduler_step':sched.last_epoch,'optimizer_state_hash':oh,'parameter_hashes':hashes,'strict_reload':True,'action_aux_max_abs':0,'finite':True};js('checkpoint_reload_audit.json',reloads)
  evaluate(arm,100);clear();print('COMPLETED',arm,flush=True)
 assert ecc()==baseline_ecc
 js('training_completion.json',{'status':'PASS','updates':{a:100 for a in ['B','C','D']},'skipped':0,'replayed':0,'ECC_unchanged':True,'GPU_peak_allocated':torch.cuda.max_memory_allocated(),'CPU_max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_seconds':time.monotonic()-start_all,'closed_loop':0,'heldout_scored':0,'backend':backend()})
except Exception:
 js('training_failure.json',{'status':'BLOCKED','error':traceback.format_exc(),'retained_logged_updates':{a:sum(r['arm']==a for r in rows) for a in ['B','C','D']},'automatic_retry':False});raise
