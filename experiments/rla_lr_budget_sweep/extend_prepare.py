"""Read-only recovery validation and pre-result freezing of the600-step continuation."""
import argparse,json,csv,hashlib,subprocess,copy
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();o=a.out;o.mkdir(parents=True,exist_ok=False)
def read(p):return json.loads(Path(p).read_text())
def save(n,x):
 with (o/n).open('x') as f:json.dump(x,f,indent=2)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
base=o.parent;short=base/'short200';long=base/'results';plan=read(short/'run_plan.json');arms=['C_m0p1','D-center_m0p1'];resume={};audits=[]
for arm,source,step in [(arms[0],long,400),(arms[1],short,200)]:
 folder=source/arm/'formal';files=sorted(folder.glob('step*.pt'));available=[int(x.stem[4:]) for x in files];assert max(available)==step,(arm,available)
 path=folder/f'step{step:04}.pt';ck=torch.load(path,map_location='cpu');sourceplan=read(source/'run_plan.json')
 required=['parameters','optimizer','scheduler','global_step','next_physical_sample','configuration','arm','AMP_scaler','dtype','activation_checkpointing','torch_rng','cuda_rng','python_rng','numpy_rng'];assert all(k in ck for k in required)
 assert ck['global_step']==step and ck['next_physical_sample']==step*8 and ck['arm']==arm and ck['configuration']==sourceplan
 assert ck['AMP_scaler'] is None and ck['dtype']=='bfloat16' and ck['activation_checkpointing']
 assert ck['scheduler']['last_epoch']==step and ck['scheduler']['milestones']=={1200:1}
 assert len(ck['optimizer']['param_groups'])==1 and ck['optimizer']['param_groups'][0]['lr']==2e-5
 assert all(int(s['step'])==step for s in ck['optimizer']['state'].values());assert ck['optimizer']['state']
 for key in ('common_step0','common_step0_sha256','inputs','seed','previous','target_center_definition'):assert sourceplan[key]==plan[key],key
 for key in ('lr','optimizer','weight_decay','lambda_aux','microbatch_appearance','accumulation','dtype','backend','beta','teacher_enabled'):assert sourceplan['shared_protocol'][key]==plan['shared_protocol'][key],key
 logs=list(csv.DictReader((folder/'training_metrics.csv').open()));logged=int(logs[-1]['step']);assert logged==(416 if arm==arms[0] else 200)
 resume[arm]=dict(checkpoint=str(path),sha256=sha(path),source_plan=str(source/'run_plan.json'),source_plan_sha256=sha(source/'run_plan.json'),step=step,logged_successful_updates=logged,known_replay=logged-step,expected_computed_updates=600-step,expected_new_distinct_update_indices=600-logged)
 audits.append(dict(arm=arm,available_checkpoints=available,restored_keys=required,optimizer_states=len(ck['optimizer']['state']),AMP='BF16 no scaler, explicitly None',lr=ck['optimizer']['param_groups'][0]['lr'],sampling_position=ck['next_physical_sample'],**resume[arm]));del ck
plan.update(experiment='low_lr_continuation600_and_paired_development',arms={k:plan['arms'][k] for k in arms},updates=600,closed_loop_trials=120,checkpoints=[200,300,400,500,600],validation=[200,300,400,500,600],resume=resume,short_source=str(short),long_run_source=str(long))
plan['shared_protocol'].update(successful_updates=600,physical_window_exposures=4800,appearance_exposures=9600,policy_updates_this_run=600,selection='fixed final600 only; no outcome-based checkpoint selection')
save('run_plan.json',plan);save('normalization.json',read(short/'normalization.json'))
ep=read(long/'evaluation_protocol.json');ep.update(models=[x+'_s600' for x in arms],trials=120,primary_comparison='D-center x0.1 minus C x0.1 at fixed600',role='previously tuned reused development; not independent test')
save('evaluation_protocol.json',ep);save('policy_registry.json',{x+'_s600':str(o/x/'formal/step0600.pt') for x in arms})
save('continuation_protocol.json',dict(frozen=True,terminal_step=600,scheduler_milestone=1200,valid_rollouts=120,actual_expected_updates=600,expected_new_distinct_updates=584,known_C_replay=16,no_other_LR=True,no_AE_updates=True,no_teacher=True,no_additional_diagnostic_backward=True,sealed_reads=0,previous_C_inflight417_unconfirmed=True,checkpoint_selection='final600',invalid_rule=ep['invalid_rule'],retries='only documented infrastructure exact-triplet retries; task failure never retried'))
save('recovery_precheck.json',dict(status='PASS',arms=audits,missing_C416_full_state=True,weights_only_restart=False))
for arm in arms:save('training_config_'+arm+'.json',dict(arm=arm,target=plan['arms'][arm]['target'],resume=resume[arm],terminal_step=600,lr=2e-5,lambda_aux=plan['shared_protocol']['lambda_aux'],unchanged_shared_protocol=plan['shared_protocol']))
with (o/'lr_curves.csv').open('x') as f:
 w=csv.DictWriter(f,fieldnames=['arm','update','lr_used','lr_after_scheduler']);w.writeheader()
 for arm in arms:
  for step in range(1,1201):w.writerow(dict(arm=arm,update=step,lr_used=2e-5,lr_after_scheduler=2e-6 if step==1200 else 2e-5))
paths=[short/'report.md',short/'run_plan.json',short/'normalization.json',Path(plan['fixed_eval_manifest']),Path(plan['inputs']['data_manifest']),Path(plan['inputs']['action_proprio_statistics']),Path(plan['inputs']['B20_sampling_order']),long/'evaluation_protocol.json']+[Path(v['checkpoint']) for v in resume.values()]
save('protected_inputs.json',[dict(path=str(p),sha256=sha(p)) for p in paths]);save('source_version.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip()))
print('RECOVERY_PASS C400 D200;600 COMPUTED INCLUDING16 KNOWN REPLAY;584 NEW DISTINCT',flush=True)
