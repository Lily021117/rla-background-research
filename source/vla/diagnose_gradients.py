"""Step5.6.1 diagnostic only: no optimizer.step and no historical writes."""
from pathlib import Path
import os
W=Path(__file__).parent
# Reuse verified initialization, 440-audit inputs and native/wrapper equivalence tests.
exec(compile((W/'validate_step0.py').read_text().split('grads={};i=selected[0]')[0],str(W/'validate_step0.py'),'exec'))
import contextlib,traceback,pickle,gc,inspect
def writecsv(name,rows):
 with (O/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
inventory=[]
for prefix,m in [('vlm',model),('head',head),('proprio',prop)]:
 for n,mod in m.named_modules():
  attrs={k:str(getattr(mod,k)) for k in ['p','drop_prob','attention_dropout','dropout','gradient_checkpointing','use_reentrant','preserve_rng_state','fused_attn'] if hasattr(mod,k) and not isinstance(getattr(mod,k),torch.nn.Module)}
  if attrs or any(s in type(mod).__name__.lower() for s in ['dropout','droppath','attention']):inventory.append({'module':prefix+'.'+n,'class':type(mod).__name__,'attributes':json.dumps(attrs),'training':mod.training,'scope':'shared action/aux' if prefix=='vlm' or n.startswith('native') else prefix})
inventory.append({'module':'head.native.predict_action','class':'learnable_random_perturbations','attributes':'normal_(mean=0,std=0.02), phase==Training; fresh ephemeral nn.Parameter, not in optimizer','training':'phase controlled, not .training','scope':'shared action/aux'})
writecsv('stochastic_module_inventory.csv',inventory)
kernel={'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,'dtype':'bf16 model; official autocast','compile':False,'checkpointing_initial':getattr(model,'is_gradient_checkpointing',None),'attention_config':str(model.config._attn_implementation),'flash_sdpa':torch.backends.cuda.flash_sdp_enabled(),'mem_efficient_sdpa':torch.backends.cuda.mem_efficient_sdp_enabled(),'math_sdpa':torch.backends.cuda.math_sdp_enabled(),'tests':{}}
js('kernel_determinism_audit.json',kernel)
fixed_batch=load_batch([selected[0]]);i=selected[0];rng_records=[];diffrows=[];outcomes={};calls=0
def rng_record(label):
 s=rng_state();p=O/(label+'_rng.pt');torch.save(s,p)
 hashes={k:hashlib.sha256(pickle.dumps(v)).hexdigest() for k,v in s.items()}
 # Tensor byte hashes avoid pickle storage identity effects.
 hashes['torch']=hashlib.sha256(s['torch'].numpy().tobytes()).hexdigest();hashes['cuda']=[hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for v in s['cuda']]
 hashes['numpy']=hashlib.sha256(s['numpy'][1].tobytes()+repr((s['numpy'][0],*s['numpy'][2:])).encode()).hexdigest()
 rng_records.append({'label':label,'path':str(p),'hashes':hashes,'dataloader_workers':0});js('rng_state_audit.json',rng_records)
 return s
def one(label,mode='eval',phase='Inference',arm='B',component='action',restore=None,profile=False):
 global calls
 clear();configure_trainability(model,head,prop,ARMS[arm])
 for m in [model,head,prop]:m.train(mode=='train')
 if restore is not None:set_rng(restore)
 rng_record(label+'_before')
 ctx=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) if profile else contextlib.nullcontext()
 with ctx as prof:
  loss=forward(fixed_batch,phase);total,aux=objective(loss,head.last_aux,target_for([i],ARMS[arm].target_source),ARMS[arm]);obj={'action':loss,'auxiliary':ARMS[arm].aux_loss_weight*aux,'total':total}[component]
  outputs={'action':head.last_action.detach().cpu().clone(),'aux':head.last_aux.detach().cpu().clone()}
  vals={'action':float(loss),'aux':float(aux),'total':float(total),'component':component};obj.backward();calls+=1
 if profile:js(label+'_operator_profile.json',[{'key':e.key,'count':e.count} for e in prof.key_averages() if any(x in e.key.lower() for x in ['attention','flash','sdpa','dropout','checkpoint'])])
 stats=gradient_summary();result={n:(p.grad.detach().cpu().clone() if p.grad is not None else None) for n,p in params};rng_record(label+'_after');clear()
 js(label+'_loss_gradnorm.json',{'loss':vals,'modules':stats})
 return result,outputs,vals
def compare(label,a,b):
 ds=aa=bb=dot=0.;mx=0.;different=0
 for n in a:
  x=a[n];y=b[n]
  if x is None and y is None:continue
  if x is None:x=torch.zeros_like(y)
  if y is None:y=torch.zeros_like(x)
  x=x.double();y=y.double();d=x-y;d2=float(d.square().sum());x2=float(x.square().sum());y2=float(y.square().sum());xy=float((x*y).sum());m=float(d.abs().max());same=torch.equal(x,y)
  diffrows.append({'test':label,'parameter':n,'max_abs':m,'mean_abs':float(d.abs().mean()),'relative_L2':(d2/max(x2,1e-30))**.5,'cosine':xy/(x2*y2)**.5 if x2*y2 else (1. if not d2 else 0.),'exact':same})
  ds+=d2;aa+=x2;bb+=y2;dot+=xy;mx=max(mx,m);different+=not same
 r={'max_abs':mx,'relative_L2':(ds/max(aa,1e-30))**.5,'cosine':dot/(aa*bb)**.5 if aa*bb else 1.,'different_tensors':different}
 writecsv('per_parameter_gradient_diff.csv',diffrows);print(label,r,flush=True);return r
seed();common_rng=rng_state()
def pair(label,mode='eval',phase='Inference',reset=True,profile=False):
 a,ao,av=one(label+'_1',mode,phase,restore=common_rng,profile=profile)
 b,bo,bv=one(label+'_2',mode,phase,restore=common_rng if reset else None)
 r=compare(label,a,b);r['action_max_abs']=float((ao['action']-bo['action']).abs().max());r['losses']=[av,bv];outcomes[label]=r;kernel['tests']=outcomes;js('kernel_determinism_audit.json',kernel);del a,b;gc.collect();return r
pair('eval_no_rng_reset',reset=False,profile=True)
pair('eval_full_rng_reset')
pair('train_no_rng_reset',mode='train',phase='Training',reset=False)
pair('train_full_rng_reset',mode='train',phase='Training')
try:
 torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
 pair('deterministic_flag_only')
except Exception:
 kernel['deterministic_error']=traceback.format_exc();js('kernel_determinism_audit.json',kernel);clear()
finally:torch.use_deterministic_algorithms(False)
# Force only math SDPA as a temporary single-factor diagnostic.
def math_sdpa():
 return torch.backends.cuda.sdp_kernel(enable_flash=False,enable_math=True,enable_mem_efficient=False)
with math_sdpa():
 pair('math_sdpa_eval',profile=True)
 pair('math_sdpa_train',mode='train',phase='Training')
 # Explicit RNG-preserving non-reentrant recomputation; never change permanent trainer.
 try:
  model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False,'preserve_rng_state':True})
  pair('math_sdpa_checkpoint_on',mode='train',phase='Training')
 except Exception:kernel['checkpoint_diagnostic_error']=traceback.format_exc();clear()
 finally:model.gradient_checkpointing_disable()
js('kernel_determinism_audit.json',kernel)
js('diagnosis_stage_complete.json',{'backward_calls':calls,'optimizer_updates':0,'parameter_hashes_unchanged':[digest(m) for m in [model,head,prop]]==initial,'ECC_unchanged':ecc()==baseline_ecc,'GPU_peak_allocated':torch.cuda.max_memory_allocated()})
# Remaining acceptance is run only if math backend demonstrates exact reproducibility.
assert outcomes['math_sdpa_train']['different_tensors']==0,'Still unresolved; stop before checkpoint acceptance'
with math_sdpa():
 # Reload explicit step0 tensors before every arm; no optimizer step.
 opt=torch.optim.AdamW([p for n,p in params],lr=1e-4,weight_decay=0);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
 set_rng(common_rng);path=O/'shared_step0.pt';save_checkpoint(path,model,head,prop,opt,sched,0,{'config':str(O/'unified_model_config.json'),'base':str(base),'base_hashes':initial,'diagnostic_backend':'math SDPA','not_formal_training':True})
 fairness={};ref=None
 for arm in ['B','C','D']:
  assert restore_checkpoint(path,model,head,prop,opt,sched)==0
  a,_,_=one('fair_'+arm,'train','Training',arm,'action',common_rng)
  if ref is None:ref=a
  fairness[arm]=compare('action_only_B_vs_'+arm,ref,a)
  assert fairness[arm]['different_tensors']==0
  del a
 del ref
 for arm in ['B','C','D']:
  restore_checkpoint(path,model,head,prop,opt,sched)
  a,_,_=one('contribution_'+arm,'train','Training',arm,'auxiliary',common_rng)
  norm=sum(float(v.double().square().sum()) for v in a.values() if v is not None)**.5
  fairness[arm]['weighted_aux_norm']=norm
  assert (norm==0) if arm=='B' else (norm>0)
  del a
 js('bcd_gradient_fairness.json',{'status':'PASS','diagnostic_backend':'math SDPA','arms':fairness,'train_mode':True,'same_rng_and_step0_reloaded':True,'optimizer_updates':0})
 def stochastic_output():
  for m in [model,head,prop]:m.train()
  with torch.no_grad():
   l=forward(fixed_batch,'Training');total,aux=objective(l,head.last_aux,target_for([i],'C'),ARMS['D']);out=[head.last_action.cpu().clone(),head.last_aux.cpu().clone(),total.cpu().clone()];head.clear();return out
 restore_checkpoint(path,model,head,prop,opt,sched);expected=stochastic_output();seed(18)
 restore_checkpoint(path,model,head,prop,opt,sched);actual=stochastic_output()
 assert all(torch.equal(x,y) for x,y in zip(expected,actual)) and not opt.state and sched.last_epoch==0
 assert [digest(m) for m in [model,head,prop]]==initial
 js('checkpoint_reload_audit.json',{'status':'PASS','action_aux_loss_max_abs':[float((x-y).abs().max()) for x,y in zip(expected,actual)],'optimizer_empty_state_equal':True,'scheduler_equal':True,'RNG_stochastic_next_forward_equal':True,'parameter_hashes_equal':True,'step':0,'base_plus_adapter_checkpoint':True,'nonempty_optimizer_moments_not_tested':True})
 # Delete/randomize training targets while keeping native input and RNG fixed.
 for variant in ['deleted','randomized']:
  set_rng(common_rng);out=stochastic_output()
  assert torch.equal(out[0],expected[0])
  if variant=='randomized':unused=torch.randn((1,1,2048),device='cuda');del unused
 js('target_isolation.json',{'status':'PASS','action_difference':0,'target_not_in_forward_signature':True,'variants':['deleted from native batch','random target not passed into action path']})
assert ecc()==baseline_ecc
js('runtime_acceptance.json',{'status':'PASS','backward_calls':calls,'optimizer_updates':0,'closed_loop':0,'parameter_hashes_unchanged':True,'GPU_peak_allocated':torch.cuda.max_memory_allocated(),'ECC_unchanged':True})
print('DIAGNOSTICS_COMPLETE',flush=True)
