"""Preflight is a hard gate. Its successful retained updates count toward 200."""
import gc,traceback
import torch
from .common import *
from .trainer import make,batch,update,save,evaluate,digest_optimizer

def init_output(t,index):
 with torch.no_grad():
  x=batch([index])['features'];out=[]
  for v in range(2):
   y=t.inference_batch(t.models,dict(features=x[:,v].cuda()))
   out.append(torch.cat([y['enc_tokens'].flatten(),y['x1_pred'].flatten()]).cpu().numpy())
 return np.stack(out)

def release(t):
 t.writer.close();t.last_output=None
 # optimizer.step has a closure; break it before dropping the trainer.
 t.optimizer.step=None

def main():
 assert read(OUT/'feature_cache_acceptance.json')['status']=='PASS'
 assert (OUT/'cache_exit_code.txt').read_text().strip()=='0'
 write(OUT/'training_gpu_gate.json',gpu_gate())
 seq=read(OUT/'sampling_sequence.json')['steps'];seq[0]=seq[0][:1]
 write(OUT/'actual_sampling_sequence.json',dict(steps=seq,reason='step1 batch1 preflight retained; steps2..200 batch4 using original schedule; identical B/C'))
 initial={};gates={};lam=None
 for group in ('B','C'):
  torch.cuda.reset_peak_memory_stats()
  t=make(group,0 if group=='B' else lam)
  initial[group]={k:model_hash(m) for k,m in t.models.items()}
  if group=='C':assert initial['C']==initial['B'],'Initial parameters differ'
  baseline=init_output(t,seq[0][0])
  if group=='B':
   np.save(OUT/'initial_output_B.npy',baseline)
   terms,_=t.training_losses(**batch(seq[0]))
   probe=t.models['encoder'].input_layer.weight
   block=next(t.models['encoder'].blocks[0].parameters())
   r,=torch.autograd.grad(terms['reconstruction'],probe,retain_graph=True)
   a,b=torch.autograd.grad(terms['consistency'],[probe,block],retain_graph=False)
   nr=float(r.float().norm());nc=float(a.float().norm());nb=float(b.float().norm())
   assert all(np.isfinite([nr,nc,nb])) and min(nr,nc,nb)>0,'Consistency gradient disconnected/invalid'
   lam=min(.1,.1*nr/(nc+1e-12))
   write(OUT/'gradient_scale_and_fixed_lambda.json',dict(lambda_cons=lam,raw_reconstruction=float(terms['reconstruction'].detach()),
    raw_consistency=float(terms['consistency'].detach()),reconstruction_encoder_input_gradient_norm=nr,
    consistency_encoder_input_gradient_norm=nc,consistency_first_block_gradient_norm=nb,
    weighted_probe_gradient_ratio=lam*nc/nr,selection='one train batch only; analytical conservative cap, no validation/test search',
    probe=seq[0],DINO='frozen upstream cache, not present in optimizer',head_added=False))
   del terms,r,a,b
  else:
   reference=np.load(OUT/'initial_output_B.npy',mmap_mode='r')
   delta=float(np.abs(reference-baseline).max());assert delta<=1e-5,delta
   write(OUT/'initialization_pairing.json',dict(initial_hashes=initial,initial_outputs_max_abs=delta,outputs_match=True,
    initial_optimizer='fresh empty states for both, identical AdamW hyperparameters; hash of empty state shared',source_step=100000,new_step=0))
  cfg=read(OUT/'protocol_before_execution.json').copy();cfg.update(group=group,lambda_cons=0 if group=='B' else lam,
      actual_sampling_sha256=sha(OUT/'actual_sampling_sequence.json'),formal_success_budget=200,
      initial_hashes=initial[group],num_workers=0)
  write(OUT/group/'config.json',cfg)
  save(t);evaluate(t,0)
  first=update(t,seq[0]);save(t)
  for k,m in t.models.items():assert model_hash(m)!=initial[group][k],('No parameter update',group,k)
  opt_before=digest_optimizer(t.optimizer);params_before={k:model_hash(m) for k,m in t.models.items()}
  pred_before=init_output(t,seq[0][0]);scale_before=t.log_scale
  release(t);del t;gc.collect();torch.cuda.empty_cache()
  t=make(group,0 if group=='B' else lam,resume_step=1)
  assert t.step==1 and t.log_scale==scale_before
  assert digest_optimizer(t.optimizer)==opt_before,'Optimizer state not exactly restored'
  assert {k:model_hash(m) for k,m in t.models.items()}==params_before,'Parameters not restored'
  pred_after=init_output(t,seq[0][0]);delta=float(np.abs(pred_before-pred_after).max());assert delta<=1e-5
  second=update(t,seq[1]);save(t)
  health()
  gates[group]=dict(status='PASS',batch1_successful_update_retained=1,resumed_successful_update_retained=2,
      model_strict_load=True,optimizer_exact_state_restored=True,global_step_restored=1,fp16_log_scale_restored=True,
      inference_reload_max_abs=delta,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
      attempts=2,successful_updates=2,discarded_updates=0,replays=0,resources=resources())
  write(OUT/group/'preflight.json',gates[group])
  release(t);del t;gc.collect();torch.cuda.empty_cache()
 write(OUT/'all_preflight_passed.json',dict(status='PASS',groups=gates,formal_training_allowed=True,
    retained_successful_updates_each=2,train_sample_test_and_resume_updates_counted=True,lambda_cons=lam,
    no_test_scoring=True,**resources()))
 # Only now may formal training continue; no new initialization or discarded smoke.
 for group in ('B','C'):
  t=make(group,0 if group=='B' else lam,resume_step=2)
  assert t.step==2
  for index in range(2,200):
   update(t,seq[index])
   if t.step in (50,100,200):save(t);evaluate(t,t.step)
  assert t.step==200
  final_hash={k:model_hash(m) for k,m in t.models.items()};opt=digest_optimizer(t.optimizer)
  reference=init_output(t,seq[0][0]);scale=t.log_scale
  release(t);del t;gc.collect();torch.cuda.empty_cache()
  t=make(group,0 if group=='B' else lam,resume_step=200)
  assert t.step==200 and t.log_scale==scale and digest_optimizer(t.optimizer)==opt
  assert {k:model_hash(m) for k,m in t.models.items()}==final_hash
  delta=float(np.abs(reference-init_output(t,seq[0][0])).max());assert delta<=1e-5
  write(OUT/group/'complete.json',dict(status='PASS',successful_updates=200,attempts=200,preflight_retained=2,
    extra_updates=0,discarded_updates=0,replays=0,overflow_skips=0,final_parameter_hashes=final_hash,
    final_optimizer_digest=opt,final_reload_max_abs=delta,fp16_scale_restored=True,optimizer_restored=True,
    checkpoints={s:{n:dict(path=str(OUT/group/'ckpts'/f'{n}_step{s:07d}.pt'),sha256=sha(OUT/group/'ckpts'/f'{n}_step{s:07d}.pt')) for n in ('encoder','decoder','misc')} for s in (0,1,2,50,100,200)},
    peak_gpu_bytes=torch.cuda.max_memory_allocated()))
  release(t);del t;gc.collect();torch.cuda.empty_cache();health()
 write(OUT/'training_complete.json',dict(status='PASS',groups=['B','C'],successful_updates=400,extra_updates=0,
       test_scoring=False,gpu_health=health(),feature_cache_sha256=sha(OUT/'feature_cache_acceptance.json')))

if __name__=='__main__':
 try:main()
 except BaseException as e:
  p=OUT/f'failure_{int(time.time())}.json'
  write(p,dict(error=repr(e),traceback=traceback.format_exc(),formal_preflight_gate_passed=(OUT/'all_preflight_passed.json').exists(),
              policy='STOP; no device switch, automatic budget extension, or destructive cleanup',resources=resources()))
  raise
