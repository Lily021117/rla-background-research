"""Freeze all eight candidates, sampling and per-update LR before model results."""
import argparse,csv,json,hashlib,subprocess,copy
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
r=Path('/data/lyc/projects/rla-wm/experiments/official_libero_rla_consistency');old=r/'stove_centering/run_20261024_01';plan=read(old/'run_plan.json');historical=Path(plan['historical_source'])/'policy_run/C/isolated_smoke/step0001.pt'
ck=torch.load(historical,map_location='cpu');assert ck['global_step']==1
groups=[{k:v for k,v in g.items() if k!='params'}|{'parameter_tensor_count':len(g['params'])} for g in ck['optimizer']['param_groups']]
assert len(groups)==1 and groups[0]['lr']==groups[0]['initial_lr']==0.0002 and groups[0]['weight_decay']==.01
assert ck['scheduler']['milestones']=={600:1};del ck
save('historical_optimizer_audit.json',dict(source=str(historical),sha256=sha(historical),groups=groups,scheduler='MultiStepLR milestone600 gamma0.1; no warmup'))
arms={f'{method}_m{str(m).replace(".","p")}':dict(target='C' if method=='C' else 'D',method=method,multiplier=m,beta=0.) for method in ('C','D-center') for m in (.1,.3,1,3)}
order=read(plan['inputs']['B20_sampling_order']);assert len(order)==4800
save('sampling_order.json',order+order)
plan['inputs']['B20_sampling_order']=str(o/'sampling_order.json');plan['inputs']['B20_sampling_sha256']=sha(o/'sampling_order.json')
plan.update(experiment='stove_lr_budget_sweep',arms=arms,updates=1200,seed=20261023,closed_loop_trials=960,checkpoints=list(range(100,1201,100)),validation=[0,100,300,600,900,1200],sealed_read=False)
plan['shared_protocol'].update(successful_updates=1200,lr=groups[0]['lr'],scheduler='MultiStepLR milestones=[1200], gamma=0.1 after update, no warmup',physical_window_exposures=9600,appearance_exposures=19200,policy_updates_this_run=9600,center_only_difference=False,C_D_only_target_source_differs=True,sampler='accepted4800 train physical-window order repeated twice; same sequence in all8 candidates',beta=0.,teacher_enabled=False)
plan['target_center_definition']={'C':'(z_C-mu_C)/s_C','D-center':'(z_D-mu_D)/s_C','scale':'unchanged frozen s_C'}
plan['selection']={'score':'three appearances equal-weight mean success rate','tie_break':['higher Clean success rate','fewer updates','smaller LR multiplier'],'candidates_steps':[600,1200],'separate_selection_per_method':True,'independent_validation_claim':False}
plan['recovery']='full model/optimizer/scheduler/AMP(None BF16)/all RNG/nextsample/globalstep; stop and preserve CUDA/ECC/nonfinite failure; no numerical hyperparameter fallback'
save('normalization.json',read(old/'normalization.json'));save('run_plan.json',plan)
ep=read(old/'evaluation_protocol.json');models=[f'{arm}_s{step}' for arm in arms for step in (600,1200)];ep.update(models=models,trials=960,primary_comparison='D-center minus C at each matched multiplier and step; separately selected method optima are tuning estimates',role='reused development states used for hyperparameter selection; not independent validation',automatic_infrastructure_retries=0)
save('evaluation_protocol.json',ep);save('policy_registry.json',{f'{arm}_s{step}':str(o/arm/f'formal/step{step:04}.pt') for arm in arms for step in (600,1200)})
manifest=read(plan['inputs']['data_manifest']);assert sum(e['split']=='train' for e in manifest['episodes'])==20 and sum(e['split']=='validation' for e in manifest['episodes'])==5
for arm,cfg in arms.items():save('training_config_'+arm+'.json',dict(**cfg,common_step0=plan['common_step0'],common_step0_sha256=plan['common_step0_sha256'],base_lr=groups[0]['lr'],lr=groups[0]['lr']*cfg['multiplier'],updates=1200,seed=plan['seed'],teacher=False,keep=False,lambda_aux=plan['shared_protocol']['lambda_aux'],protocol=plan['shared_protocol']))
with (o/'lr_curves.csv').open('x') as f:
 w=csv.DictWriter(f,fieldnames=['configuration','group','update','lr_used','lr_after_scheduler']);w.writeheader()
 for arm,cfg in arms.items():
  for step in range(1,1201):w.writerow(dict(configuration=arm,group=0,update=step,lr_used=groups[0]['lr']*cfg['multiplier'],lr_after_scheduler=groups[0]['lr']*cfg['multiplier']*(.1 if step==1200 else 1)))
paths=[old/'run_plan.json',old/'normalization.json',old/'evaluation_protocol.json',Path(plan['common_step0']),Path(plan['inputs']['data_manifest']),Path(plan['inputs']['action_proprio_statistics']),historical]
save('protected_inputs.json',[dict(path=str(x),sha256=sha(x)) for x in paths]);save('source_version.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),official_commit='0fc6e97212442ebccffa06f1d42f5481e5f22950'))
save('search_protocol.json',dict(frozen=True,methods=['C','D-center'],multipliers=[.1,.3,1,3],steps=[600,1200],updates_per_configuration=1200,total_updates=9600,total_rollouts=960,selection=plan['selection'],scheduler=plan['shared_protocol']['scheduler'],historical_first600_lr_equal_for_m1=True,training_windows=len(order)//1,window_epochs=9600/2555,next_unselected_matching_seeds=[20261026,20261027],no_automatic_extra_seed=True))
print('FROZEN8_CONFIGURATIONS',flush=True)
