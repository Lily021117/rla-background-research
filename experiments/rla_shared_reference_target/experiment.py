"""Two frozen balanced orders; shared Clean AE-C targets, no AE training."""
import argparse,collections,csv,fcntl,json,os,re,subprocess,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_sampling_stability.experiment import read,save,sha,table,R,SEEDS
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
C_RUN=R/'experiments/rla_sampling_stability/run_20261027_01'
D_RUN=R/'experiments/rla_sampling_stability/run_20261027_02'
GPU='GPU-205a3f5a-d91b-6385-c588-4e4cf3505f7e'
CONDITIONS=['clean','seen_background','unseen_background']

def prepare(out):
 assert read(D_RUN/'delivery_verification.json')['new_valid_rollouts']==120
 protected={};configdiff=[]
 for seed in SEEDS:
  src=C_RUN/f'seed_{seed}';root=out/f'seed_{seed}';root.mkdir()
  old=read(src/'run_plan.json');plan=json.loads(json.dumps(old));step0=src/'C/formal/step0000.pt'
  plan.update(common_step0=str(step0),common_step0_sha256=sha(step0),arms=['C-shared-target'],target_read_mode='shared_clean',shared_reference_gate=True,
   target_source_by_arm={'C':'C','C-shared-target':'C','preflight':'C'},paired_C_sample_trace=str(src/'C/formal/actual_sample_ids.jsonl'),causal_design='Only background auxiliary target appearance changes; same frozen AE-C and statistics')
  save(root/'run_plan.json',plan);save(root/'normalization.json',read(src/'normalization.json'))
  save(root/'preflight6/isolated_smoke/preflight_acceptance.json',read(src/'preflight6/isolated_smoke/preflight_acceptance.json'))
  ep=read(src/'evaluation_protocol.json');ep.update(models=['C-shared-target_s600'],role='exploratory reused development; same frozen states and conditions')
  save(root/'evaluation_protocol.json',ep);save(root/'policy_registry.json',{'C-shared-target_s600':str(root/'C-shared-target/formal/step0600.pt')})
  save(root/'representation_registry.json',{'C':str(src/'C/formal/step0600.pt'),'C-shared-target':str(root/'C-shared-target/formal/step0600.pt')})
  configdiff.append(dict(order_seed=seed,difference_keys=[k for k in plan if plan[k]!=old.get(k)],shared_training_protocol_unchanged=plan['shared_protocol']==old['shared_protocol'],step0_sha256=sha(step0),sampling_sha256=sha(plan['inputs']['B20_sampling_order'])))
  for p in [step0,src/'run_plan.json',src/'normalization.json',src/'evaluation_protocol.json',src/'per_trial_results.json',src/'C/formal/actual_sample_ids.jsonl',Path(plan['inputs']['B20_sampling_order']),Path(plan['inputs']['data_manifest']),Path(plan['inputs']['action_proprio_statistics']),D_RUN/f'seed_{seed}/per_trial_results.json']:
   protected[str(p)]=sha(p)
 plan=read(out/f'seed_{SEEDS[0]}/run_plan.json');ae=Path(plan['previous']);normal=read(out/f'seed_{SEEDS[0]}/normalization.json');manifest=read(plan['inputs']['data_manifest']);provenance=read(ae/'sidecar_provenance.json')
 sys.path[:0]=[str(HERE.parent/'official_libero_rla_consistency'),str(R/'worktrees/rla-official-0fc6e97/rebuttal/exp2_vla_adapter/vla_adapter_overlay')]
 os.environ['CENTER_RUN']=str(out/f'seed_{SEEDS[0]}');os.environ['CENTER_ARM']='C-shared-target'
 from stove_centering.store import CenterStore
 from rla_shared_reference_target.targets import target_appearance
 stores={c:CenterStore(str(ae/'sidecars/C'/c),chunk=8,num_tokens=16,token_dim=64,norm='center') for c in ['clean','background']}
 rows=[];fixed=[];moments=collections.defaultdict(lambda:np.zeros(3,np.float64))
 for e in manifest['episodes']:
  assert e['split'] in ['train','validation'];root=Path(e['root']);N=e['valid_chunks'];lab=np.load(root/'trajectory_labels.npz');raw=lab['proprio'];valid=lab['valid_horizon_mask'];slots={c:stores[c].slot(raw) for c in stores};assert min(slots.values())>=0
  records={c:next(r for r in provenance if r['model']=='C' and r['episode']==e['episode'] and r['appearance']==c) for c in stores}
  for c,r in records.items():
   assert sha(r['file'])==r['sha256'];protected[r['file']]=r['sha256'];protected[str(Path(r['file']).parents[1]/'stats.json')]=sha(Path(r['file']).parents[1]/'stats.json')
   np.testing.assert_array_equal(np.load(r['mask_file'])[:N],valid[:N]);assert valid[:N].all();assert r['encoder_sha256']==records['clean']['encoder_sha256']
   np.testing.assert_array_equal(stores[c].offset,np.array(normal['mu_C'],np.float32));assert stores[c].scale==np.float32(normal['s_C'])
  for t in range(N):
   clean=stores['clean'].frame(slots['clean'],t);background=stores['background'].frame(slots['background'],t)
   shared=[stores[target_appearance(c,'shared_clean')].frame(slots[target_appearance(c,'shared_clean')],t) for c in stores]
   np.testing.assert_array_equal(shared[0],shared[1]);np.testing.assert_array_equal(shared[0],clean);assert clean.shape==(8,1024) and np.isfinite(clean).all()
   if e['split']=='train':
    for name,z in [('C_clean',clean),('C_background',background),('shared_clean',shared[0]),('shared_background',shared[1])]:
     moments[name]+=np.array([z.size,z.sum(dtype=np.float64),np.square(z.astype(np.float64)).sum()])
  for t in sorted(set([0,N//2])):fixed.append([e['episode'],t,e['split']])
  rows.append(dict(episode=e['episode'],split=e['split'],valid_chunks=N,shape='8x16x64 (flattened8x1024)',clean_key=records['clean']['key'],background_key=records['background']['key'],shared_reference_key=records['clean']['key'],clean_sidecar_sha256=records['clean']['sha256'],background_sidecar_sha256=records['background']['sha256'],encoder_sha256=records['clean']['encoder_sha256'],all_valid_masks_match=True,shared_targets_elementwise_equal=True))
 table(out/'target_mapping_audit.csv',rows);save(out/'fixed_representation_samples.json',fixed)
 save(out/'target_distribution.json',{k:dict(elements=int(v[0]),mean=v[1]/v[0],variance=v[2]/v[0]-(v[1]/v[0])**2,RMS=np.sqrt(v[2]/v[0]),weighting='each eligible physical training window once, all8x1024 valid values') for k,v in moments.items()})
 save(out/'configuration_diff.json',configdiff);save(out/'protected_inputs.json',protected)
 save(out/'protocol.json',dict(method='C-shared-target',only_change='Background input retains background image, auxiliary target is matching AE-C Clean target',seeds=SEEDS,updates_per_order=600,total_updates=1200,rollouts=120,lambda_aux=plan['shared_protocol']['lambda_aux'],mean='existing mu_C',scale='existing s_C',AE_updates=0,teacher=False,keep=False,sealed_reads=0,representation_samples='first and midpoint valid window of each train/development episode, fixed before outputs',statistics='20000 bootstrap draws; physical state clustered jointly across appearances and both orders',GPU=GPU,device_deviation='C historical evaluation GPU1; new evaluation GPU0; D reference GPU0',stop='no extra configurations or trials; infrastructure error stops, no silent retry'))
 (out/'method_definition.md').write_text('# C-shared-target\n\nFrozen AE-C; both appearances use the same episode/timestep Clean auxiliary target. Background images and all action/observation fields remain unchanged. Existing C mean and scale, official mean L1 losses, fixed lambda_aux. Eight cumulative horizons and joint dual-view16x64 latent retained; no deployment reference image. No AE-D, AE updates, teacher, keep or additional consistency loss.\n')
 print('SHARED_PREPARE_PASS',flush=True)

def run(out):
 lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 save(out/'execution_start.json',dict(pid=os.getpid(),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),GPU=GPU))
 jobs=[(SEEDS[0],'gate')]+[(s,'train') for s in SEEDS]+[(s,'evaluate') for s in SEEDS]+[(s,'representation') for s in SEEDS]
 for seed,stage in jobs:
  print('START',seed,stage,flush=True)
  subprocess.run([sys.executable,str(HERE/'launch.py'),'--out',str(out),'--seed',str(seed),'--stage',stage],cwd=REPO,check=True)
 analyze(out);publish(out)

def analyze(out):
 rates=[];flips=[];phases=[];trials=[];inventory=[];training=[];validation=[];representation=[];arrays={};checks=[]
 for seed in SEEDS:
  root=out/f'seed_{seed}';formal=root/'C-shared-target/formal'
  accept=read(formal/'training_acceptance.json');assert accept['successful_updates']==600 and accept['strict_reload']
  expected=[json.loads(x) for x in (C_RUN/f'seed_{seed}/C/formal/actual_sample_ids.jsonl').read_text().splitlines()]
  actual=[json.loads(x) for x in (formal/'actual_sample_ids.jsonl').read_text().splitlines()];assert actual==expected and len(actual)==600
  init=read(formal/'initialization_audit.json');assert init['all_trainable_parameters_equal_common_step0'] and init['CPU_CUDA_RNG_equal_common_step0']
  cc=read(C_RUN/f'seed_{seed}/C/formal/execution_configuration.json');new=read(formal/'execution_configuration.json');assert {k:v for k,v in cc.items() if k!='arm'}=={k:v for k,v in new.items() if k!='arm'}
  checks.append(dict(seed=seed,initial_parameters_RNG_match=True,actual600_sample_ID_rows_exact=True,execution_configuration_except_arm_exact=True))
  training.extend(dict(order_seed=seed,**r) for r in csv.DictReader((formal/'training_metrics.csv').open()));validation.extend(dict(order_seed=seed,**r) for r in csv.DictReader((formal/'validation_metrics.csv').open()))
  representation.extend(dict(order_seed=seed,**r) for r in csv.DictReader((root/'representation/formal/predicted_representation_metrics.csv').open()))
  groups={}
  for method,p in [('C',C_RUN/f'seed_{seed}'),('D-center',D_RUN/f'seed_{seed}'),('C-shared-target',root)]:
   rr=read(p/'per_trial_results.json');assert len(rr)==60 and all(r['status']=='COMPLETE' for r in rr)
   groups[method]={(r['seed'],r['condition']):r for r in rr};states=sorted({r['seed'] for r in rr});assert len(states)==20
   arrays[seed,method]=np.array([[int(groups[method][s,c]['success']) for c in CONDITIONS] for s in states])
   trials.extend(dict(order_seed=seed,method=method,**r) for r in rr)
   for c in CONDITIONS:
    q=[r for r in rr if r['condition']==c];rates.append(dict(order_seed=seed,method=method,appearance=c,successes=sum(r['success'] for r in q),trials=20,SR=sum(r['success'] for r in q)/20))
    for stage,n in collections.Counter(r['stage'] for r in q).items():phases.append(dict(order_seed=seed,method=method,appearance=c,stage=stage,count=n))
  for key,r in groups['C-shared-target'].items():
   for method in ['C','D-center']:
    ref=groups[method][key]
    for k in ['state_sha256','xml_sha256','language']:assert r[k]==ref[k]
    flips.append(dict(order_seed=seed,state=key[0],appearance=key[1],reference=method,shared=int(r['success']),reference_success=int(ref['success']),difference=int(r['success'])-int(ref['success'])))
  ck=formal/'step0600.pt';inventory.append(dict(order_seed=seed,path=str(ck),sha256=sha(ck),global_step=600,strict_reload=True))
 table(out/'success_summary.csv',rates);table(out/'paired_results.csv',flips);table(out/'failure_stage_summary.csv',phases);table(out/'checkpoint_inventory.csv',inventory);table(out/'training_metrics.csv',training);table(out/'validation_metrics.csv',validation);table(out/'predicted_representation_metrics.csv',representation);save(out/'per_trial_results.json',trials);save(out/'matching_audit.json',checks)
 rng=np.random.RandomState(20261027);ix=rng.randint(0,20,(20000,20));stats=[];interactions=[]
 def stat(label,d):
  sampled=d[ix];return dict(comparison=label,difference=float(d.mean()),CI95=np.quantile(sampled.mean(axis=tuple(range(1,sampled.ndim))),[.025,.975]).tolist())
 for reference in ['C','D-center']:
  diffs=[]
  for seed in SEEDS:
   diff=arrays[seed,'C-shared-target']-arrays[seed,reference];diffs.append(diff);stats.append(stat(f'{seed}: shared minus {reference}, equal appearance',diff))
   for j,c in enumerate(CONDITIONS):stats.append(stat(f'{seed}: shared minus {reference}, {c}',diff[:,j]))
   for j,c in enumerate(CONDITIONS[1:],1):interactions.append(stat(f'{seed}: {c} shared minus {reference} background interaction',diff[:,j]-diff[:,0]))
  mean=np.mean(diffs,axis=0);stats.append(stat(f'two orders: shared minus {reference}',mean))
  for j,c in enumerate(CONDITIONS[1:],1):interactions.append(stat(f'two orders: {c} shared minus {reference} background interaction',mean[:,j]-mean[:,0]))
 save(out/'paired_statistics.json',dict(results=stats,background_interactions=interactions,unit='20 physical states across all appearances and orders',conditional_on_two_orders=True,development_not_independent_test=True))
 rep=[]
 for seed in SEEDS:
  for method in ['C','C-shared-target']:
   for split in ['train','validation']:
    rr=[r for r in representation if r['order_seed']==seed and r['method']==method and r['split']==split]
    rep.append(dict(order_seed=seed,method=method,split=split,pairs=len(rr),**{k:float(np.mean([float(r[k]) for r in rr])) for k in ['cross_prediction_L1','cross_prediction_MSE','cross_prediction_relative_L2','clean_reference_L1','background_reference_L1','clean_reference_weighted_L1','background_reference_weighted_L1','clean_action_L1','background_action_L1']}))
 table(out/'representation_summary.csv',rep)
 assert all(sha(p)==h for p,h in read(out/'protected_inputs.json').items())
 save(out/'delivery_verification.json',dict(status='COMPLETE_EXECUTION_NOT_METHOD',effective_updates=1200,actual_update_computations=1200,replayed_updates=0,valid_rollouts=120,actual_attempts=120,invalid_trials=0,AE_updates=0,C_updates=0,C_new_rollouts=0,sealed_reads=0,protected_inputs_unchanged=True))
 lines=['# Shared Clean AE-C target: paired experiment','', '| Order | Method | Clean | Seen | Unseen | Mean |','|---|---|---|---|---|---|']
 for seed in SEEDS:
  for method in ['C','D-center','C-shared-target']:
   a=arrays[seed,method];lines.append(f'| {seed} | {method} | '+' | '.join(f'{n}/20' for n in a.sum(0))+f' | {a.mean():.2%} |')
 lines+=['','## Primary paired effects']
 for s in stats:
  if 'equal appearance' in s['comparison'] or 'two orders' in s['comparison']:lines.append(f"- {s['comparison']}: {100*s['difference']:+.2f} pp, state-cluster95%CI [{100*s['CI95'][0]:+.2f}, {100*s['CI95'][1]:+.2f}].")
 lines+=['','## Interpretation and limitations',
 '- Recovery relative to D is not an improvement over C. Target equality is imposed, not evidence of learned invariance: actual policy prediction differences and errors to the fixed Clean reference are in representation_summary.csv.',
 '- All successes/failures retained. Stage labels are frozen distance/lift proxies, not ground-truth causal failure diagnoses.',
 '- The two orders are not independent training seeds; bootstrap clusters the same physical state jointly across orders/appearances. These20 development states have been repeatedly used in development, not an independent test.',
 '- Only Background target appearance changes. Existing AE-C, mean/scale, auxiliary coefficient, observations, labels and deployment unchanged. AE/teacher/keep/additional losses absent.',
 '- New runs/evaluation use GPU0; historical C used GPU1, D reference evaluated on GPU0. This execution difference is disclosed.',
 '-1200 successful updates,120 valid new rollouts, no replay, no invalid trial, no AE/C update or sealed-set read. Historical protected hashes unchanged. No automatic extension.']
 (out/'report.md').write_text('\n'.join(lines)+'\n');print('SHARED_ALL_EXPERIMENTS_COMPLETE',flush=True)

def publish(out):
 dest=HERE/'delivery'/out.name;dest.mkdir(parents=True,exist_ok=False)
 names=['report.md','method_definition.md','protocol.json','configuration_diff.json','target_mapping_audit.csv','target_distribution.json','fixed_representation_samples.json','matching_audit.json','success_summary.csv','paired_results.csv','failure_stage_summary.csv','checkpoint_inventory.csv','training_metrics.csv','representation_summary.csv','predicted_representation_metrics.csv','paired_statistics.json','delivery_verification.json','execution_start.json']
 for name in names:
  text=(out/name).read_text().replace(str(R),'<PROJECT_ROOT>');text=re.sub(r'GPU-[0-9a-f-]{36}','<GPU_UUID>',text)
  assert len(text)<5_000_000 and not re.search(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_|-----BEGIN .*PRIVATE KEY',text)
  with (dest/name).open('x') as f:f.write(text)
 gate=read(out/'seed_20261101/preflight/formal/shared_reference_gate.json');save(dest/'shared_reference_gate.json',gate)
 def git(*args,env=None):return subprocess.check_output(['git',*args],cwd=REPO,env=env,text=True).strip()
 branch='experiments/rla-shared-reference-target';assert git('branch','--show-current')==branch;rel=str(dest.relative_to(REPO));git('add','--',rel);assert all(p.startswith(rel+'/') for p in git('diff','--cached','--name-only').splitlines())
 git('-c','user.name=Lily021117','-c','user.email=Lily021117@users.noreply.github.com','commit','-m','Report shared Clean target paired experiment')
 env=os.environ.copy();env.update(GH_CONFIG_DIR='/data/lyc/tools/github-cli/config',XDG_CACHE_HOME='/data/lyc/tmp/github-cli/cache',TMPDIR='/data/lyc/tmp/github-cli')
 auth=['-c','credential.helper=','-c','credential.helper=!/data/lyc/tools/github-cli/v2.103.0/gh_2.103.0_linux_amd64/bin/gh auth git-credential'];git(*auth,'push','origin','HEAD:refs/heads/'+branch,env=env)
 commit=git('rev-parse','HEAD');assert git(*auth,'ls-remote','origin','refs/heads/'+branch,env=env).split()[0]==commit
 save(out/'github_sync_report.json',dict(status='PUSHED_VERIFIED',branch=branch,commit=commit,repository='https://github.com/Lily021117/rla-background-research'))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['prepare','run'],required=True);a=p.parse_args()
 (prepare if a.mode=='prepare' else run)(a.out)
