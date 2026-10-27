"""Complete balanced C/D-center pairs, reuse C results, no new sampler seeds."""
import argparse,collections,csv,fcntl,hashlib,json,os,pickle,re,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_sampling_stability.experiment import read,save,sha,table,R,HERE,REPO,SEEDS

C_RUN=R/'experiments/rla_sampling_stability/run_20261027_01'
GPU='GPU-37d3f7d4-309e-2bb8-2d51-292dbbcce419'
def same_parameters(a,b):
 assert a.keys()==b.keys()
 for k in a:
  assert a[k].keys()==b[k].keys()
  for n in a[k]:assert torch.equal(a[k][n],b[k][n]),(k,n)
def same_rng(a,b):
 assert torch.equal(a['torch_rng'],b['torch_rng'])
 assert all(torch.equal(x,y) for x,y in zip(a['cuda_rng'],b['cuda_rng']))
 for k in ['python_rng','numpy_rng']:assert pickle.dumps(a[k])==pickle.dumps(b[k])

def prepare(out):
 assert read(C_RUN/'delivery_verification.json')['valid_rollouts']==120
 protocol=dict(experiment='balanced sampler D-center paired completion',seeds=SEEDS,methods=['D-center'],C_reference=str(C_RUN),
  new_updates=1200,updates_per_group=600,new_rollouts=120,AE_updates=0,C_updates=0,C_new_rollouts=0,
  initial_source='each matching C actual saved step0000, including exact model and RNG; fresh empty optimizer',
  sampler='read existing C frozen index file directly; do not regenerate',target='historical original AE-D; (z_D-mu_D)/s_C',
  lambda_aux=.22408536585365854,LR=.0002,scheduler='original600 MultiStepLR, post-update, all600 update LR2e-4',
  primary='D-center minus C within each order, then two-order equal mean',
  uncertainty='physical states clustered across both orders and all appearances; conditional on two orders, not population seed uncertainty',
  checkpoint_selection='final600 only',GPU='same authorized GPU1 as C; prior ECC baseline retained, stop on increments',
  keep=False,teacher=False,sealed_reads=0,automatic_extra_runs=False)
 save(out/'replication_protocol.json',protocol)
 protected={};audit=[]
 for seed in SEEDS:
  source=C_RUN/f'seed_{seed}';root=out/f'seed_{seed}';root.mkdir()
  original=read(source/'run_plan.json');ckpath=source/'C/formal/step0000.pt'
  ck=torch.load(ckpath,map_location='cpu');old=torch.load(original['common_step0'],map_location='cpu')
  assert ck['global_step']==0 and ck['next_physical_sample']==0 and not ck['optimizer']['state']
  same_parameters(ck['parameters'],old['parameters']);same_rng(ck,old);del ck,old
  order=read(original['inputs']['B20_sampling_order']);assert len(order)==4800
  trace=source/'C/formal/actual_sample_ids.jsonl';records=[json.loads(x) for x in trace.read_text().splitlines()];assert len(records)==600
  for s,row in enumerate(records,1):assert row['step']==s and row['actual_sample_ids']==[[e,t,c] for e,t in order[(s-1)*8:s*8] for c in ['clean','background']]
  plan=json.loads(json.dumps(original));plan.update(common_step0=str(ckpath),common_step0_sha256=sha(ckpath),arms=['D-center'],
   paired_C_sample_trace=str(trace),paired_C_result_source=str(source),causal_design='balanced sampler D-center completion; only AE target and own center differ',target_source_by_arm={'C':'C','D-center':'D'})
  save(root/'run_plan.json',plan)
  save(root/'normalization.json',read(source/'normalization.json'))
  save(root/'initialization_gate.json',dict(status='PASS',actual_matching_C_step0=str(ckpath),sha256=sha(ckpath),all_parameters_and_RNG_equal=True,auxiliary_not_reinitialized=True,optimizer_fresh=True))
  save(root/'preflight6/isolated_smoke/preflight_acceptance.json',read(source/'preflight6/isolated_smoke/preflight_acceptance.json'))
  ep=read(source/'evaluation_protocol.json');ep.update(models=['D-center_s600'],role='paired D-center completion, same reused development states as matching C')
  save(root/'evaluation_protocol.json',ep);save(root/'policy_registry.json',{'D-center_s600':str(root/'D-center/formal/step0600.pt')})
  save(root/'resource_assignment.json',dict(GPU_UUID=GPU,authorization='user previously explicitly authorized GPU1; reuse same device as C',ECC_volatile_baseline=1,aggregate_parity_baseline=2))
  accepted=read(source/'C/formal/execution_configuration.json');assert accepted['lambda_aux']==protocol['lambda_aux'] and accepted['effective_appearance_batch']==16 and accepted['backend']=={'math':True,'flash':False,'efficient':False}
  audit.append(dict(order_seed=seed,initialization_actual_C_step0_verified=True,all_C_actual_sample_ID_rows_match=True,index_source=original['inputs']['B20_sampling_order'],index_sha256=sha(original['inputs']['B20_sampling_order']),C_execution_configuration=accepted,
   config_difference_keys=[k for k in plan if plan[k]!=original.get(k)],training_shared_protocol_exact=plan['shared_protocol']==original['shared_protocol']))
  for p in [ckpath,Path(original['inputs']['B20_sampling_order']),trace,source/'normalization.json',source/'per_trial_results.json',source/'run_plan.json',source/'evaluation_protocol.json']:
   protected[str(p)]=sha(p)
 # Small actual target-reader check on fixed training windows, not a full re-audit.
 root=out/f'seed_{SEEDS[0]}';plan=read(root/'run_plan.json');ae=Path(plan['previous']);normal=read(root/'normalization.json')
 sys.path[:0]=[str(HERE.parent/'official_libero_rla_consistency'),str(R/'experiments/official_libero_rla_consistency/run_20261023_02/vla-official'),str(R/'worktrees/rla-official-0fc6e97/rebuttal/exp2_vla_adapter/vla_adapter_overlay')]
 os.environ['CENTER_RUN']=str(root);os.environ['CENTER_ARM']='D-center'
 from stove_centering.store import CenterStore
 manifest=read(plan['inputs']['data_manifest']);es={e['episode']:e for e in manifest['episodes']};provenance=read(ae/'sidecar_provenance.json');checks=[]
 selected=read(plan['inputs']['B20_sampling_order'])[:3]
 for appearance in ['clean','background']:
  store=CenterStore(str(ae/'sidecars/D'/appearance),chunk=8,num_tokens=16,token_dim=64,norm='center')
  np.testing.assert_array_equal(store.offset,np.array(normal['mu_D'],np.float32));assert store.scale==np.float32(normal['s_C'])
  for episode,t in selected:
   e=es[episode];assert e['split']=='train'
   with np.load(Path(e['root'])/'trajectory_labels.npz') as lab:slot=store.slot(lab['proprio']);valid=lab['valid_horizon_mask'][t].copy()
   rec=next(x for x in provenance if x['model']=='D' and x['episode']==episode and x['appearance']==appearance)
   assert sha(rec['file'])==rec['sha256'];z=np.load(rec['file'],mmap_mode='r');mask=np.load(rec['mask_file'],mmap_mode='r');assert valid.all();np.testing.assert_array_equal(mask[t],valid)
   expected=((z[t].astype(np.float32)-np.array(normal['mu_D'],np.float32))/np.float32(normal['s_C'])).reshape(8,1024)
   actual=store.frame(slot,t);np.testing.assert_array_equal(actual,expected);assert np.isfinite(actual).all()
   checks.append(dict(episode=episode,timestep=t,appearance=appearance,encoder_sha256=rec['encoder_sha256'],sidecar_sha256=rec['sha256'],shape=list(actual.shape),exact=True))
 save(out/'frozen_target_audit.json',dict(status='PASS',checks=checks,source='original historical AE-D, NOT inv03',own_D_mean=True,common_C_scale=True,new_targets=0,AE_updates=0))
 save(out/'configuration_matching_audit.json',audit);save(out/'protected_inputs.json',protected)
 print('D_PAIRED_PREPARATION_PASS_ZERO_UPDATES',flush=True)

def run(out):
 lock=(out/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 save(out/'execution_started.json',dict(pid=os.getpid(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),GPU_UUID=GPU))
 for stage in ['D-center','evaluate']:
  for seed in SEEDS:
   while int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))<26624:
    print('WAIT_GPU1_CAPACITY',stage,seed,flush=True);time.sleep(30)
   assert int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True))==1
   print('START',stage,seed,flush=True)
   subprocess.run(['python',str(HERE.parent/'rla_policy_seed_replication/launch.py'),'--out',str(out),'--seed',str(seed),'--stage',stage],cwd=REPO,check=True)
 analyze(out);publish(out)

def analyze(out):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 rates=[];flips=[];phases=[];inventory=[];metrics=[];validation=[];arrays={};trial_rows=[];checks=[]
 conditions=['clean','seen_background','unseen_background'];state_ids=None
 for seed in SEEDS:
  groups={}
  for method,root,arm in [('C',C_RUN/f'seed_{seed}','C'),('D-center',out/f'seed_{seed}','D-center')]:
   formal=root/arm/'formal';rr=read(root/'per_trial_results.json');assert len(rr)==60 and all(r['status']=='COMPLETE' for r in rr)
   groups[method]={(r['seed'],r['condition']):r for r in rr};states=sorted({r['seed'] for r in rr})
   if state_ids is None:state_ids=states
   assert state_ids==states and len(states)==20
   arrays[(seed,method)]=np.array([[int(groups[method][s,c]['success']) for c in conditions] for s in states])
   for r in rr:trial_rows.append(dict(order_seed=seed,method=method,**r))
   for c in conditions:
    q=[r for r in rr if r['condition']==c];rates.append(dict(order_seed=seed,method=method,appearance=c,successes=sum(r['success'] for r in q),trials=20,SR=sum(r['success'] for r in q)/20))
    for phase,n in collections.Counter(r['stage'] for r in q).items():phases.append(dict(order_seed=seed,method=method,appearance=c,stage=phase,count=n))
   ckpath=formal/'step0600.pt';ck=torch.load(ckpath,map_location='cpu');assert ck['global_step']==600 and ck['next_physical_sample']==4800 and ck['scheduler']['last_epoch']==600
   assert all(int(x['step'])==600 for x in ck['optimizer']['state'].values());del ck
   inventory.append(dict(order_seed=seed,method=method,path=str(ckpath),sha256=sha(ckpath),step=600,reused=method=='C'))
   assert read(formal/'training_acceptance.json')['strict_reload']
   ts=list(csv.DictReader((formal/'training_metrics.csv').open()));assert len(ts)==600 and all(float(r['lr'])==.0002 for r in ts)
   metrics.extend(dict(order_seed=seed,method=method,**r) for r in ts)
   vs=list(csv.DictReader((formal/'validation_metrics.csv').open()));validation.extend(dict(order_seed=seed,method=method,**r) for r in vs)
  croot=C_RUN/f'seed_{seed}'/'C/formal';droot=out/f'seed_{seed}'/'D-center/formal'
  c0=torch.load(croot/'step0000.pt',map_location='cpu');d0=torch.load(droot/'step0000.pt',map_location='cpu');same_parameters(c0['parameters'],d0['parameters']);same_rng(c0,d0);assert not d0['optimizer']['state'];del c0,d0
  ct=[json.loads(x) for x in (croot/'actual_sample_ids.jsonl').read_text().splitlines()];dt=[json.loads(x) for x in (droot/'actual_sample_ids.jsonl').read_text().splitlines()];assert ct==dt and len(dt)==600
  cc=read(croot/'execution_configuration.json');dc=read(droot/'execution_configuration.json');assert {k:v for k,v in cc.items() if k!='arm'}=={k:v for k,v in dc.items() if k!='arm'}
  checks.append(dict(order_seed=seed,step0_parameters_RNG_exact=True,actual600_sample_ID_sequences_exact=True,execution_config_exact_except_arm=True))
  assert groups['C'].keys()==groups['D-center'].keys()
  for key,c in groups['C'].items():
   d=groups['D-center'][key]
   for k in ['state_sha256','xml_sha256','language']:assert c[k]==d[k]
   flips.append(dict(order_seed=seed,state=key[0],appearance=key[1],C=int(c['success']),D_center=int(d['success']),difference=int(d['success'])-int(c['success'])))
 table(out/'success_summary.csv',rates);table(out/'paired_results.csv',flips);table(out/'failure_stage_summary.csv',phases);table(out/'checkpoint_inventory.csv',inventory);table(out/'training_metrics.csv',metrics);table(out/'validation_metrics.csv',validation)
 save(out/'per_trial_results.json',trial_rows);save(out/'paired_matching_audit.json',checks)
 rng=np.random.RandomState(20261027);ix=rng.randint(0,20,(20000,20));stats=[];interactions=[]
 def stat(label,d):return dict(comparison=label,difference=float(d.mean()),physical_state_cluster_CI95=np.quantile(d[ix].mean(axis=tuple(range(1,d[ix].ndim))),[.025,.975]).tolist())
 diffs=[]
 for seed in SEEDS:
  C=arrays[seed,'C'];D=arrays[seed,'D-center'];diff=D-C;diffs.append(diff)
  stats.append(stat(f'{seed}: equal appearance D-C',diff))
  for j,c in enumerate(conditions):
   s=stat(f'{seed}: {c} D-C',diff[:,j]);s.update(D_wins=int((diff[:,j]>0).sum()),C_wins=int((diff[:,j]<0).sum()));stats.append(s)
  for j,c in enumerate(conditions[1:],1):
   interaction=(D[:,j]-D[:,0])-(C[:,j]-C[:,0]);s=stat(f'{seed}: {c} interaction D-C',interaction);s.update(C_background_minus_clean=float((C[:,j]-C[:,0]).mean()),D_background_minus_clean=float((D[:,j]-D[:,0]).mean()));interactions.append(s)
 # Collapse order axis before resampling: never pretend orders are new states.
 mean_diff=np.mean(diffs,axis=0)
 stats.append(stat('two orders equally weighted D-C',mean_diff))
 for j,c in enumerate(conditions):stats.append(stat(f'two orders: {c} D-C',mean_diff[:,j]))
 for j,c in enumerate(conditions[1:],1):interactions.append(stat(f'two orders: {c} interaction D-C',mean_diff[:,j]-mean_diff[:,0]))
 save(out/'paired_statistics.json',dict(results=stats,background_interactions=interactions,unit='20 physical states, all orders and appearances kept together',conditional_on_two_orders=True,not_independent_test=True))
 assert all(sha(p)==h for p,h in read(out/'protected_inputs.json').items())
 save(out/'delivery_verification.json',dict(status='COMPLETE_EXECUTION_NOT_METHOD',new_policy_updates=1200,actual_successful_computations=1200,replayed_updates=0,new_valid_rollouts=120,actual_new_attempts=120,invalid_trials=0,C_training_updates=0,C_new_rollouts=0,AE_updates=0,sealed_reads=0,protected_inputs_unchanged=True))
 lines=['# Balanced sampling D-center paired completion','',
  'C reused unchanged; new D-center uses matching C saved step0, exact training RNG and actual sample sequence. Historical AE-D, own mu_D and common s_C; no inv03, AE update or new coefficient.','',
  '| Order | Method | Clean | Seen | Unseen | Mean |','|---|---|---|---|---|---|']
 for seed in SEEDS:
  for method in ['C','D-center']:
   a=arrays[seed,method];lines.append(f'| {seed} | {method} | '+' | '.join(f'{n}/20' for n in a.sum(0))+f' | {a.mean():.1%} |')
 lines+=['','## Interpretation']
 effects=[float(d.mean()) for d in diffs]
 for seed,effect in zip(SEEDS,effects):lines.append(f'- Order {seed}: D-center minus C = {100*effect:+.1f} percentage points.')
 lines.append(f'- Two-order equally weighted difference: {100*np.mean(effects):+.1f} percentage points.')
 lines+=['- High absolute D success below C is recovery, not evidence that consistency improves control.',
         '- If only one order favors D, the effect is order dependent. Two orders cannot establish general stability.',
         '- Background interaction must be interpreted with absolute Clean/background rates; loss of Clean ability is not robustness improvement.',
         '- Single task, fixed AE and initialization, reused/tuned development states. C/D target scale/representation differs; common coefficient does not equate gradient strength.',
         '- Historical uniform-window results are references only; balanced sampling changes weights, not merely ordering.',
         '- Completed1200 new updates and120 new rollouts; C/AE updates0, no extra experiment or sealed data. Same GPU1 as C, previous ECC baseline retained, no increment permitted.']
 with (out/'report.md').open('x') as f:f.write('\n'.join(lines)+'\n')
 print('D_PAIRED_COMPLETION_FINISHED',flush=True)

def publish(out):
 dest=HERE/'delivery'/out.name;dest.mkdir(parents=True,exist_ok=False)
 names=['report.md','replication_protocol.json','configuration_matching_audit.json','frozen_target_audit.json','paired_matching_audit.json','success_summary.csv','paired_results.csv','paired_statistics.json','failure_stage_summary.csv','checkpoint_inventory.csv','delivery_verification.json','training_metrics.csv','execution_started.json']
 for name in names:
  text=(out/name).read_text().replace(str(R),'<PROJECT_ROOT>');text=re.sub(r'GPU-[0-9a-f-]{36}','<GPU_UUID>',text)
  assert len(text)<5_000_000 and not re.search(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN .*PRIVATE KEY',text)
  with (dest/name).open('x') as f:f.write(text)
 def git(*args,env=None):return subprocess.check_output(['git',*args],cwd=REPO,env=env,text=True).strip()
 branch='experiments/rla-sampling-stability';assert git('branch','--show-current')==branch
 rel=str(dest.relative_to(REPO));git('add','--',rel);assert all(p.startswith(rel+'/') for p in git('diff','--cached','--name-only').splitlines())
 git('-c','user.name=Lily021117','-c','user.email=Lily021117@users.noreply.github.com','commit','-m','Report balanced-sampling matched D-center completion')
 env=os.environ.copy();env.update(GH_CONFIG_DIR='/data/lyc/tools/github-cli/config',XDG_CACHE_HOME='/data/lyc/tmp/github-cli/cache',TMPDIR='/data/lyc/tmp/github-cli')
 auth=['-c','credential.helper=','-c','credential.helper=!/data/lyc/tools/github-cli/v2.103.0/gh_2.103.0_linux_amd64/bin/gh auth git-credential']
 git(*auth,'push','origin','HEAD:refs/heads/'+branch,env=env);commit=git('rev-parse','HEAD');assert git(*auth,'ls-remote','origin','refs/heads/'+branch,env=env).split()[0]==commit
 save(out/'github_sync_report.json',dict(status='PUSHED_VERIFIED',commit=commit,branch=branch,repository='https://github.com/Lily021117/rla-background-research'))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['prepare','run'],required=True);a=p.parse_args()
 if a.mode=='prepare':prepare(a.out)
 else:run(a.out)
