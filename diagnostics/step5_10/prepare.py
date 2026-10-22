"""Freeze inputs, verify checkpoints and inspect existing Adam states on CPU."""
from common import *
import torch,collections,math
torch.set_num_threads(4)
assert not torch.cuda.is_available(),'preparation must hide CUDA'
protected=read(P/'protection_before.json')
for p in P.rglob('*'):
 if p.is_file() and not p.is_symlink() and '.git' not in p.parts:protected[str(p)]=sha(p)
for p in W.glob('*.py'):protected[str(p)]=sha(p)
js('protection_before.json',protected);js('git_before.json',{str(p):git(p) for p in [R,W,P/'share_repository_final']})
print('PROTECTED',len(protected),flush=True)
inventory=csvread(T/'checkpoint_inventory.csv');cp0=T/'B/step0000.pt';cp1=T/'B/step0001.pt';common=T/'common_step0.pt'
for step,p in [(0,cp0),(1,cp1)]:assert sha(p)==next(r['SHA256'] for r in inventory if r['arm']=='B' and int(r['step'])==step)
a=torch.load(str(cp0),map_location='cpu',weights_only=False,mmap=True);b=torch.load(str(cp1),map_location='cpu',weights_only=False,mmap=True);z=torch.load(str(common),map_location='cpu',weights_only=False,mmap=True)
assert a['step']==0 and b['step']==1 and z['step']==0
for k in ['vlm','head','proprio']:
 assert set(a[k])==set(b[k])==set(z[k]);assert all(torch.equal(a[k][n],z[k][n]) for n in a[k])
 assert all(torch.isfinite(x).all() for x in b[k].values())
assert a['provenance']['arm']['name']==b['provenance']['arm']['name']=='B'
assert a['provenance']['sources']==b['provenance']['sources']==z['provenance']['sources']
for p,h in a['provenance']['sources'].items():assert sha(p)==h,p
cfg=read(T/'training_config_B.json');assert a['provenance']['config_SHA256']==b['provenance']['config_SHA256']==sha(T/'training_config_B.json')
assert b['provenance']['successful_updates']==1 and b['provenance']['backend']==dict(flash=False,mem_efficient=False,math=True)
updates=[json.loads(l) for l in (T/'B/updates.jsonl').read_text().splitlines()];assert updates[0]['global_step']==1 and updates[0]['actual_successful_updates']==1 and updates[0]['skipped']==updates[0]['replayed']==0
assert not a['optimizer']['state'] and not z['optimizer']['state'];assert a['scheduler']['last_epoch']==0 and b['scheduler']['last_epoch']==1
optsteps={int(v['step'].item()) for v in b['optimizer']['state'].values()};assert optsteps=={1}
samples=read(S/'sample_manifest.json');ids=[int(r['sample_id']) for r in csvread(P/'fixed_input_action_probe.csv') if r['arm']=='A' and int(r['step'])==0]
assert len(ids)==len(set(ids))==14
selected=[samples[i] for i in ids]
for s in selected:assert sha(s['path'])==s['SHA256'] and s['split'] in ['train','validation']
stats=read(BASE/'dataset_statistics.json')['libero_spatial_no_noops'];oldstats=read(E/'step4_1_adapter_readiness/run_20261021_03/runtime_metadata/dataset_statistics.json')['libero_spatial_no_noops'];assert stats==oldstats
js('fixed_sample_manifest.json',{'frozen_before_scoring':True,'selection':'exact Step5.9 A14 IDs in recorded order; no reselection','samples':selected,'batch_size':1,'cameras':['agentview','robot0_eye_in_hand'],'pixels':'same accepted cached policy tensor 1x12x224x224','seed':20261021,'spaces':'normalized action and separately decoded command groups; no pooled physical-unit MAE','STOP':2,'terminal_STOP_added_in_memory_only':True,'Math_SDPA':True,'group_order':['000','100','010','001','110','101','011','111'],'meaning':'bits VLM, action head, proprio; 1=existing B step1,0=B step0','step0_reference':'frozen A, verified native and unified on same fixed inputs','no_new_heldout_scores':True})
js('checkpoint_provenance.json',{'status':'PASS_CPU_PROVENANCE','checkpoints':{'B_step0':{'path':str(cp0),'SHA256':sha(cp0),'global_step':0},'B_step1':{'path':str(cp1),'SHA256':sha(cp1),'global_step':1},'common_step0':{'path':str(common),'SHA256':sha(common),'global_step':0}},'same_training_run':str(T),'common_step0_parameter_exact_equal':True,'same_config_SHA256':sha(T/'training_config_B.json'),'sources_verified':a['provenance']['sources'],'published_action_proprio_statistics_SHA256':sha(BASE/'dataset_statistics.json'),'action_proprio_statistics':stats,'Math_SDPA':True,'first_update_log':updates[0],'step0_A_inference_equivalence':'pending separate GPU acceptance','strict_key_coverage_saved_groups':True,'no_new_checkpoint':True})
names=[r['name'] for r in csvread(T/'parameter_structure_comparison.csv') if r['arm']=='B' and r['trainable']=='True']
assert len(names)==sum(len(g['params']) for g in a['optimizer']['param_groups'])
grps=[]
for i,(x,y) in enumerate(zip(a['optimizer']['param_groups'],b['optimizer']['param_groups'])):
 assert x['params']==y['params'];assert x['lr']==y['lr']==1e-4
 grps.append({'index':i,'step0':{k:v for k,v in x.items() if k!='params'},'step1':{k:v for k,v in y.items() if k!='params'},'parameter_names':[names[j] for j in x['params']]})
rows=[];moments=collections.defaultdict(collections.Counter)
for storage,prefix in [('vlm','vlm'),('head','head'),('proprio','proprio')]:
 for n,x in a[storage].items():
  full=prefix+'.'+n;y=b[storage][n];delta=y.float()-x.float();refnorm=float(x.double().norm());upnorm=float(delta.double().norm());zero=refnorm==0
  component='VLM' if storage=='vlm' else 'proprio' if storage=='proprio' else 'auxiliary_fixed' if n.startswith('auxiliary.') else 'action_head'
  if component=='auxiliary_fixed':assert torch.equal(x,y)
  rows.append({'component':component,'parameter':full,'numel':x.numel(),'historically_trainable':full in names,'dtype':str(x.dtype),'parameter_norm_step0':refnorm,'parameter_norm_step1':float(y.double().norm()),'update_norm':upnorm,'relative_update_norm':upnorm/refnorm if not zero else 'UNDEFINED_ZERO_INITIALIZATION','zero_initialization':zero,'near_zero_RMS_lt_1e-12':refnorm/math.sqrt(x.numel())<1e-12,'max_absolute_update':float(delta.abs().max()),'changed_coordinates':int(delta.count_nonzero()),'component_scope':'saved changing subset only for VLM; frozen base covered in GPU inventory'})
for j,state in b['optimizer']['state'].items():
 name=names[j];component='VLM' if name.startswith('vlm.') else 'proprio' if name.startswith('proprio.') else 'auxiliary_fixed' if name.startswith('head.auxiliary.') else 'action_head'
 for key in ['exp_avg','exp_avg_sq']:moments[component][key+'_'+str(state[key].dtype)]+=state[key].numel()
cs('parameter_update_statistics.csv',rows)
js('optimizer_first_step_audit.json',{'status':'PASS','optimizer':'AdamW','parameter_groups':grps,'step0_state_entries':len(a['optimizer']['state']),'step1_state_entries':len(b['optimizer']['state']),'step1_all_Adam_steps':[1],'scheduler_step0':a['scheduler'],'scheduler_step1':b['scheduler'],'scheduler_behavior':'constant multiplier1; lr before/after step1 is1e-4','gradient_accumulation':2,'microbatch_size':2,'effective_batch':4,'gradient_clipping':False,'raw_first_step_gradients_available':False,'gradient_evidence':'existing aggregate component grad norms only; no new backward or reconstruction of quantized Adam gradients','first_step_record':updates[0],'fresh_initialization_not_resume':True,'optimizer_moment_dtypes_by_component':{k:dict(v) for k,v in moments.items()},'proprio_gradient_not_separately_logged':True,'no_optimizer_created_by_diagnostic':True})
js('preparation_acceptance.json',{'status':'PASS','fixed_samples':14,'training_samples':sum(s['split']=='train' for s in selected),'validation_samples':sum(s['split']=='validation' for s in selected),'parameter_rows':len(rows),'optimizer_updates':0,'backward_calls':0,'new_rollouts':0})
print('CPU_PREPARATION_PASS',flush=True)
