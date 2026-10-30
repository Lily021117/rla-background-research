"""Exact accepted objective and batching; explicit bounded continuation edits."""
import inspect,hashlib
from rla_scratch_policy_aux import training as original
from rla_low_lr_continue.common import weight,LAMBDA,save
RESUME_CHECK='''
 def check_resume_forward():
  saved=rng();before=digest();buffers={(k,n):b.detach().cpu().clone() for k,m in mods.items() for n,b in m.named_buffers()};results=[]
  examples=[(e,t,c) for e,t in order[4800:4808] for c in ['clean','background']]
  for repeat in range(2):
   restore(saved);outputs=[]
   with torch.no_grad():
    for start in [0,8]:
     b=batch(examples[start:start+8]);total=forward(b,weight(arm,600));outputs.append((capture['action'].detach().cpu().clone(),capture['z'].detach().cpu().clone(),float(capture['act_loss'].detach()),float(total.detach())));capture.clear();del b,total
   results.append(outputs)
   if repeat==0:
    source=torch.load(plan['resume_checkpoint'],map_location='cpu')
    for k,m in mods.items():full=m.state_dict();full.update(source['parameters'][k]);m.load_state_dict(full,strict=True)
    opt.load_state_dict(source['optimizer']);scheduler.load_state_dict(source['scheduler']);restore(source);del source
  assert all(torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]) and a[2:]==b[2:] for a,b in zip(*results))
  assert digest()==before and all(torch.equal(b.detach().cpu(),buffers[k,n]) for k,m in mods.items() for n,b in m.named_buffers())
  restore(saved);assert all(pg['lr']==2e-5 for pg in opt.param_groups) and scheduler.last_epoch==600
  g['save']('resume_forward_equivalence.json',dict(status='PASS',global_step=600,strict_model_optimizer_scheduler_reload=True,actions_auxiliary_outputs_losses_exact=True,full_effective_batch=16,examples=examples,RNG_and_buffers_unchanged=True,optimizer_updates=0,backward=0,LR=2e-5))
 check_resume_forward();reload(snapshot(start_step));del initial
'''
def adapted():
 source=inspect.getsource(original.execute);patched=source;changes=[]
 def replace(a,b):
  nonlocal patched
  n=patched.count(a);assert n>0,('missing source fragment',a);patched=patched.replace(a,b);changes.append(dict(old=a,new=b,count=n))
 replace("initial['global_step']==0","initial['global_step']==600")
 replace('len(order)==4800','len(order)==14400')
 replace("target_arm='C' if preflight or arm=='C-fixed' else 'D'","target_arm='C' if arm=='C-low' else 'D'")
 replace('no_trained_policy_loaded=True','continuation_from_trained_step600=True')
 replace("common_step0_sha256=plan['common_step0_sha256']","resume_checkpoint_sha256=plan['common_step0_sha256']")
 replace("assert read(out.parent.parent/'preflight/formal/preflight_acceptance.json')['status']=='PASS'","assert read(plan['source_acceptance'])['successful_updates']==600")
 replace("assert ck['configuration']==plan and ck['arm']==arm","assert ck['configuration']==read(plan['source_run_plan']) and ck['arm']==arm")
 replace('min(step,599)','min(step,1799)')
 replace(" reload(snapshot(start_step));validate(start_step);diagnostic(start_step,target_arm,weight(arm,min(start_step,599)));del initial",RESUME_CHECK.strip('\n'))
 replace('range(start_step+1,601)','range(start_step+1,1801)')
 replace('assert lr==2e-4','assert lr==2e-5')
 replace('if step in [1,100,300,600]:','if step in [900,1200,1500,1800]:')
 replace('if step in [100,300,600]:','if step in [900,1200,1500,1800]:')
 replace('if step in [300,600]:','if False:')
 replace("reload(out/'step0600.pt')","reload(out/'step1800.pt')")
 replace('successful_updates=600,new_successful_updates=600-start_step,actual_update_computations=600-start_step','successful_updates=1800,new_successful_updates=1800-start_step,actual_update_computations=1800-start_step')
 replace('physical_exposures=4800,appearance_exposures=9600','physical_exposures=9600,appearance_exposures=19200')
 replace("optimizer='AdamW',lr=2e-4","optimizer='AdamW',lr=2e-5")
 replace("scheduler='MultiStepLR600 gamma0.1; optimizer then scheduler; all600 actual update LR2e-4'","scheduler='restore MultiStepLR last_epoch600; all updates601..1800 LR2e-5, no additional decay'")
 replace("lambda_schedule='successful update k0..599; restore global_step'","lambda_schedule='constant half weight; global updates601..1800'")
 return source,patched,changes
def execute(g):
 source,patched,changes=adapted();save(g['o']/'source_adaptation.json',dict(original_sha256=hashlib.sha256(source.encode()).hexdigest(),effective_sha256=hashlib.sha256(patched.encode()).hexdigest(),changes=changes))
 scope=dict(original.__dict__);scope.update(weight=weight,LAMBDA=LAMBDA)
 exec(compile(patched,__file__,'exec'),scope);scope['execute'](g)
