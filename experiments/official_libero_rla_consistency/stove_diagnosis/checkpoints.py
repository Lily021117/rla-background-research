"""CPU full-state checkpoint identities and replay audit; no training."""
import argparse,gc
from common import *
import torch
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=Path(read(a.out/'configuration.json')['source']);plan=read(root/'policy_run/run_plan.json');rows=[];policy=[]
for arm in ('B','C','D'):
 ck=torch.load(root/'policy_run'/arm/'isolated_smoke/step0000.pt',map_location='cpu',weights_only=False);ref=torch.load(plan['inputs']['B20_step0'] if arm=='B' else plan['common_step0'],map_location='cpu',weights_only=False)
 assert ck['global_step']==0 and not ck['optimizer']['state'];assert digest_state(ck)==digest_state(ref)
 for k in ('torch_rng','python_rng','numpy_rng','cuda_rng'):
  def equal(x,y):
   if torch.is_tensor(x):return torch.equal(x,y)
   if isinstance(x,np.ndarray):return np.array_equal(x,y)
   if isinstance(x,(list,tuple)):return len(x)==len(y) and all(equal(xx,yy) for xx,yy in zip(x,y))
   return x==y
  assert equal(ck[k],ref[k])
 policy.append(dict(arm=arm,initial_digest=digest_state(ck),same_as_agreed_untrained_initial=True,initial_RNG_matches_reference=True,initial_optimizer_empty=True,initial_global_step=0))
 del ck,ref;gc.collect();ck=torch.load(root/'policy_run'/arm/'isolated_smoke/step0600.pt',map_location='cpu',weights_only=False);assert ck['configuration']==plan and ck['global_step']==600 and ck['next_physical_sample']==4800 and ck['scheduler']['last_epoch']==600;assert all(int(v['step'])==600 for v in ck['optimizer']['state'].values());assert ck['activation_checkpointing']
 policy[-1].update(final_step=600,sampler_position=4800,optimizer_steps_all600=True,scheduler_last_epoch=600,configuration_exact=True,AMP_scaler=ck['AMP_scaler'],resume_files=[str(p) for p in (root/'policy_run'/arm).rglob('*resume*.json')]);del ck;gc.collect()
init=read(root/'ae_run/official_initialization.json');registry=read(sorted((root/'ae_run/model_registry_versions').glob('*.json'))[-1]);ae=[]
for arm in ('C','D'):
 for module in ('encoder','decoder'):
  initial=torch.load(init[module],map_location='cpu',weights_only=True);original=torch.load(root/'ae_run'/('AE_'+arm)/(module+'_step0000.pt'),map_location='cpu',weights_only=True);assert initial.keys()==original.keys() and all(torch.equal(initial[k].to(original[k].dtype),original[k]) for k in initial);ae.append(dict(arm=arm,module=module,step0_exact_official90000_after_configured_dtype_load=True,dtype_evidence='official SimpleTokenTransformer use_fp16 torso conversion before strict load',cast_parameters=sum(initial[k].dtype!=original[k].dtype for k in initial),max_cast_rounding=max(float((initial[k].float()-original[k].float()).abs().max()) for k in initial),initial_SHA256=sha(init[module]),initial_hash_matches_frozen=sha(init[module])==init[module+'_sha256']));assert ae[-1]['initial_hash_matches_frozen'];del initial,original
  final=Path(registry[arm])/(module+'_step0600.pt');rows.append(dict(stage='AE',arm=arm,module=module,path=str(final),sha256=sha(final)))
 ck=torch.load(Path(registry[arm])/'resume_step0600.pt',map_location='cpu',weights_only=False);assert ck['global_step']==600 and ck['next_sample']==4800 and ck['optimizer']['state'];assert all(int(x['step'])==600 for x in ck['optimizer']['state'].values());ae.append(dict(arm=arm,module='resume',global_step=600,next_sample=4800,keys=sorted(ck),scheduler='none in frozen AE recipe',AMP_scaler='none: official fixed256 FP32-master scaling',lambda_cons=ck['lambda_cons'],plan_matches_frozen=ck['plan']==read(root/'ae_run/training_plan.json')));del ck;gc.collect()
for arm in ('B','C','D'):rows.append(dict(stage='policy',arm=arm,module='trainables+optimizer+scheduler+RNG',path=str(root/'policy_run'/arm/'isolated_smoke/step0600.pt'),sha256=sha(root/'policy_run'/arm/'isolated_smoke/step0600.pt')))
save(a.out/'checkpoint_state_audit.json',dict(policy=policy,AE=ae,replay=read(root/'recovery_prefix_audit.json'),policy_training_resumed=False,common_C_D_initial_exact=policy[1]['initial_digest']==policy[2]['initial_digest'],historical_bootstrap_names_not_training_identity=True));csvsave(a.out/'checkpoint_inventory.csv',rows);print('CHECKPOINT STATES VERIFIED',flush=True)
