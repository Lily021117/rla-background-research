"""Frozen episode-balanced sampling intervention; two C arms, no loss changes."""
import argparse,collections,csv,fcntl,hashlib,json,os,pickle,re,subprocess,time
from pathlib import Path
import numpy as np
import torch

R=Path('/data/lyc/projects/rla-wm')
PRE=R/'experiments/rla_policy_seed_replication/run_20261026_02'
CROSS=R/'experiments/rla_initialization_rng_cross/run_20261026_01'
HIST=R/'experiments/official_libero_rla_consistency/run_20261023_10/stove/policy_run'
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
GPU='GPU-205a3f5a-d91b-6385-c588-4e4cf3505f7e'
SEEDS=[20261101,20261102]
SEGMENTS={'all':(0,4800),'first200':(0,1600),'middle200':(1600,3200),'last200':(3200,4800),'last100':(4000,4800)}
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def table(p,rows):
 with Path(p).open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def balanced(lengths,seed):
 # Independent RNG object; no calls to any model/global RNG.
 rng=np.random.RandomState(seed);episodes=list(lengths);pending={e:[] for e in episodes};order=[]
 for cycle in range(240):
  for j in rng.permutation(len(episodes)):
   e=episodes[j]
   if not pending[e]:pending[e]=list(map(int,rng.permutation(lengths[e])))
   order.append([e,pending[e].pop()])
 assert len(order)==4800 and set(collections.Counter(e for e,t in order).values())=={240}
 for e,n in lengths.items():
  counts=collections.Counter(t for ep,t in order if ep==e)
  assert max(counts.get(t,0) for t in range(n))-min(counts.get(t,0) for t in range(n))<=1
 return order

def statistics(label,order,lengths):
 per_episode=[];per_window=[];timebins=[]
 for segment,(a,b) in SEGMENTS.items():
  rows=order[a:b];ec=collections.Counter(e for e,t in rows);wc=collections.Counter((e,t) for e,t in rows)
  for e,n in lengths.items():
   counts=[wc[(e,t)] for t in range(n)]
   per_episode.append(dict(sequence=label,segment=segment,episode=e,eligible_windows=n,physical_exposures=ec[e],clean_exposures=ec[e],background_exposures=ec[e],unique_windows=sum(v>0 for v in counts),min_repeat=min(counts),max_repeat=max(counts)))
   for t,v in enumerate(counts):per_window.append(dict(sequence=label,segment=segment,episode=e,timestep=t,relative_time=t/max(n-1,1),exposures=v))
   for k in range(10):
    ts=[t for t in range(n) if min(9,int(10*t/n))==k]
    timebins.append(dict(sequence=label,segment=segment,episode=e,relative_time_decile=k,eligible_windows=len(ts),exposures=sum(wc[(e,t)] for t in ts),semantic_stage='NOT_LABELED'))
 return per_episode,per_window,timebins

def prepare(out):
 base=read(PRE/'seed_20261029/run_plan.json');hist=read(HIST/'run_plan.json')
 manifest=read(base['inputs']['data_manifest']);train=[e for e in manifest['episodes'] if e['split']=='train'];dev=[e for e in manifest['episodes'] if e['split']=='validation']
 assert len(train)==20 and len(dev)==5;lengths={e['episode']:e['valid_chunks'] for e in train};N=sum(lengths.values());assert N==2555
 assert not(set(lengths)&{e['episode'] for e in dev})
 common=Path(base['original_common_step0']);assert sha(common)==base['original_common_step0_sha256']
 historical_orders={20261023:Path(hist['inputs']['B20_sampling_order']),20261029:Path(base['inputs']['B20_sampling_order'])}
 epstats=[];wstats=[];tstats=[];evidence=[]
 for seed,path in historical_orders.items():
  order=read(path);assert len(order)==4800
  windows=[(e,t) for e,n in lengths.items() for t in range(n)];rng=np.random.RandomState(seed);rebuilt=[]
  while len(rebuilt)<4800:rebuilt.extend([list(windows[i]) for i in rng.permutation(N)])
  assert rebuilt[:4800]==order
  root=HIST/'C/isolated_smoke' if seed==20261023 else PRE/'seed_20261029/C/formal'
  ck=torch.load(root/'step0600.pt',map_location='cpu');assert ck['global_step']==600 and ck['next_physical_sample']==4800
  config=ck['configuration'];assert read(config['inputs']['B20_sampling_order'])==order
  metrics=list(csv.DictReader((root/'training_metrics.csv').open()));assert [int(r['step']) for r in metrics]==list(range(1,601))
  evidence.append(dict(seed=seed,order_file=str(path),order_sha256=sha(path),final_checkpoint=str(root/'step0600.pt'),checkpoint_global_step=600,next_sample=4800,successful_metric_rows=600,
    full_actual_sample_ID_log_available=False,evidence_level='frozen index replay + terminal checkpoint configuration/cursor + update log; NOT an independently recorded full historical sample trace',
    algorithm_reconstruction_exact=True,direct_synchronous_dataset_sample=True,workers=0,shuffle_buffer=None,restore='checkpoint RNG and cursor; these historical C arms have uninterrupted 600-row logs',
    caveat='source code determines access order; cannot independently prove every historical decoded item from absent per-item logs'))
  del ck
  x,y,z=statistics(f'history_{seed}',order,lengths);epstats+=x;wstats+=y;tstats+=z
 save(out/'historical_sampling_evidence.json',evidence)
 weights=[dict(episode=e,valid_windows=n,old_long_run_episode_probability=n/N,new_episode_probability=1/20,old_per_window_probability=1/N,new_per_window_probability=1/(20*n),new_to_old_window_weight=N/(20*n),new_exact_episode_exposures=240) for e,n in lengths.items()]
 table(out/'sampling_weight_change.csv',weights)
 target=read(PRE/'frozen_target_audit.json');assert target['status']=='PASS';target['reused_from']=str(PRE/'frozen_target_audit.json');save(out/'frozen_target_audit.json',target)
 protocol=dict(experiment='C episode-balanced sampling intervention',seeds=SEEDS,model_initialization_and_training_RNG='same historical common step0 for both; sampler seeds ONLY',
    common_step0=str(common),common_step0_sha256=sha(common),sampler='240 independent episode-permutation cycles; per-episode shuffled windows exhausted before reshuffle',
    weighting='changes uniform-window to uniform-episode weighting; not merely exchanging order',updates_per_arm=600,total_updates=1200,total_rollouts=120,
    independent_train_episodes=20,development_episodes=5,eligible_windows=N,physical_samples_per_arm=4800,appearance_samples_per_arm=9600,
    no_development_sampling_selection=True,LR=.0002,lambda_aux=.22408536585365854,scheduler='MultiStepLR[600] gamma.1 AFTER optimizer, all600 updates2e-4',
    AE_updates=0,D_updates=0,keep=False,teacher=False,sealed_reads=0,selection='final600 only',stop='CUDA/ECC/nonfinite/identity mismatch; no silent retry or new seed',
    stages='relative temporal deciles only; no fabricated semantic phase labels',GPU='authorized GPU0 only, >=26GiB before each stage',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
 save(out/'replication_protocol.json',protocol)
 for seed in SEEDS:
  root=out/f'seed_{seed}';root.mkdir();before=pickle.dumps(np.random.get_state());order=balanced(lengths,seed);assert before==pickle.dumps(np.random.get_state())
  assert order==balanced(lengths,seed);save(root/'sampling_order.json',order)
  plan=json.loads(json.dumps(base));plan.update(common_step0=str(common),common_step0_sha256=sha(common),policy_seed=20261023,sampler_seed=seed,arms=['C'],closed_loop_trials=60,log_actual_sample_ids=True,
    causal_design='episode-balanced sampling scheme, order seed only differs between new arms',initialization_scope='exact historical policy parameters AND all saved training RNG')
  plan['shared_protocol'].update(seed=20261023,sampler='episode balanced cycles with independent order RNG; each episode exactly240 exposures',normalization='historical fixed C mean/scale; unchanged action/proprio')
  plan['inputs']['B20_sampling_order']=str(root/'sampling_order.json');plan['inputs']['B20_sampling_sha256']=sha(root/'sampling_order.json')
  save(root/'run_plan.json',plan);save(root/'normalization.json',read(PRE/'seed_20261029/normalization.json'))
  save(root/'preflight6/isolated_smoke/preflight_acceptance.json',read(PRE/'seed_20261029/preflight6/isolated_smoke/preflight_acceptance.json'))
  save(root/'initialization_gate.json',dict(status='PASS',basis='same existing accepted historical step0 file, exact SHA; no reconstructed model or RNG',source=str(common),sha256=sha(common),optimizer_updates=0))
  ep=read(PRE/'seed_20261029/evaluation_protocol.json');ep.update(models=['C_s600'],trials=60,role=f'balanced sampler order seed{seed}; reused development, NOT independent test')
  save(root/'evaluation_protocol.json',ep);save(root/'policy_registry.json',{'C_s600':str(root/'C/formal/step0600.pt')})
  x,y,z=statistics(f'balanced_{seed}',order,lengths);epstats+=x;wstats+=y;tstats+=z
 table(out/'episode_sampling_statistics.csv',epstats);table(out/'window_sampling_statistics.csv',wstats);table(out/'relative_time_sampling_statistics.csv',tstats)
 save(out/'protected_inputs.json',{str(p):sha(p) for p in [common,*historical_orders.values(),Path(base['inputs']['data_manifest']),Path(base['inputs']['action_proprio_statistics']),PRE/'seed_20261029/normalization.json']})
 print('SAMPLING_FROZEN_ZERO_UPDATES',flush=True)

def run(out):
 lock=(out/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 gpu=os.environ.get('SAMPLING_GPU_UUID',GPU)
 assert gpu in [GPU,'GPU-37d3f7d4-309e-2bb8-2d51-292dbbcce419']
 if gpu!=GPU:
  assert read(out/'gpu1_authorization.json')['user_authorized']
  assert read(out/f'seed_{SEEDS[0]}/preflight/gpu1_zero_update/device_preflight.json')['status']=='PASS'
  assert all(read(out/f'seed_{s}/resource_assignment.json')['GPU_UUID']==gpu for s in SEEDS)
 name='execution_started_gpu1.json' if gpu!=GPU else 'execution_started.json'
 save(out/name,dict(pid=os.getpid(),GPU_UUID=gpu,new_policy_updates_budget=1200,new_rollouts_budget=120))
 baseline=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True))
 assert baseline==(1 if gpu!=GPU else 0)
 for stage in ['C','evaluate']:
  for seed in SEEDS:
   while int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))<26624:
    print('WAIT_AUTHORIZED_GPU_CAPACITY',stage,seed,flush=True);time.sleep(30)
   assert int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True))==baseline
   print('START',seed,stage,flush=True)
   subprocess.run(['python',str(HERE.parent/'rla_policy_seed_replication/launch.py'),'--out',str(out),'--seed',str(seed),'--stage',stage],cwd=REPO,check=True)
 analyze(out);publish(out)

def analyze(out):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 rows=[];validation=[];trains=[];audit=[];phases=[]
 sources=[('history_I23_T23',HIST),('history_I29_T29',PRE/'seed_20261029'),('history_I30_T30',PRE/'seed_20261030'),
          ('cross_I23_T29',CROSS/'seed_20261029'),('cross_I29_T23',CROSS/'seed_20261023')]
 sources += [(f'balanced_{s}',out/f'seed_{s}') for s in SEEDS]
 trialsets={}
 for label,root in sources:
  rr=[r for r in read(root/'per_trial_results.json') if r['model'] in ['C','C_s600']];assert len(rr)==60 and all(r['status']=='COMPLETE' for r in rr)
  trialsets[label]={(r['seed'],r['condition']):r for r in rr}
  for c in ['clean','seen_background','unseen_background']:
   q=[r for r in rr if r['condition']==c];rows.append(dict(group=label,condition=c,successes=sum(r['success'] for r in q),trials=20,SR=sum(r['success'] for r in q)/20))
   for phase,n in collections.Counter(r['stage'] for r in q).items():phases.append(dict(group=label,condition=c,stage=phase,count=n))
 for seed in SEEDS:
  root=out/f'seed_{seed}';formal=root/'C/formal';order=read(root/'sampling_order.json')
  acceptance=read(formal/'training_acceptance.json');assert acceptance['successful_updates']==600 and acceptance['strict_reload']
  ids=[json.loads(s) for s in (formal/'actual_sample_ids.jsonl').read_text().splitlines()];assert len(ids)==600
  for step,row in enumerate(ids,1):assert row['step']==step and row['actual_sample_ids']==[[e,int(t),c] for e,t in order[(step-1)*8:step*8] for c in ['clean','background']]
  ck=torch.load(formal/'step0000.pt',map_location='cpu');init=torch.load(read(root/'run_plan.json')['common_step0'],map_location='cpu')
  assert not ck['optimizer']['state']
  assert all(torch.equal(v,init['parameters'][k][n]) for k,ps in ck['parameters'].items() for n,v in ps.items())
  for k in ['torch_rng','cuda_rng','python_rng','numpy_rng']:
   if k=='torch_rng':assert torch.equal(ck[k],init[k])
   elif k=='cuda_rng':assert all(torch.equal(a,b) for a,b in zip(ck[k],init[k]))
   else:assert pickle.dumps(ck[k])==pickle.dumps(init[k])
  del ck,init
  audit.append(dict(sampler_seed=seed,all600_actual_sample_ID_rows_exact=True,same_historical_parameters_RNG=True,optimizer_initially_empty=True))
  for name,dest in [('training_metrics.csv',trains),('validation_metrics.csv',validation)]:
   for row in csv.DictReader((formal/name).open()):dest.append(dict(group=f'balanced_{seed}',**row))
 table(out/'success_summary.csv',rows);table(out/'failure_stage_summary.csv',phases);table(out/'training_metrics.csv',trains);table(out/'validation_metrics.csv',validation)
 save(out/'actual_sampling_verification.json',audit)
 keys=sorted(next(iter(trialsets.values())));assert all(sorted(r)==keys for r in trialsets.values())
 for key in keys:
  for rr in trialsets.values():
   for field in ['state_sha256','xml_sha256','language']:assert rr[key][field]==trialsets['history_I23_T23'][key][field]
 states=sorted({s for s,c in keys});conditions=['clean','seen_background','unseen_background'];rng=np.random.RandomState(20261027);ix=rng.randint(0,20,(20000,20));arr={k:np.array([[int(v[(s,c)]['success']) for c in conditions] for s in states]) for k,v in trialsets.items()};comparisons=[]
 for new in [f'balanced_{s}' for s in SEEDS]:
  for old in ['history_I23_T23','cross_I23_T29']:
   diff=arr[new]-arr[old];comparisons.append(dict(new=new,reference=old,difference=float(diff.mean()),CI95=np.quantile(diff[ix].mean(axis=(1,2)),[.025,.975]).tolist(),gains=int((diff>0).sum()),losses=int((diff<0).sum())))
 save(out/'paired_statistics.json',dict(unit='physical state clustering, appearances jointly',comparisons=comparisons,not_independent_test=True,only_two_order_seeds=True))
 summaries=[]
 for group in sorted({r['group'] for r in validation}):
  for step in [0,100,300,600]:
   for appearance in ['clean','background']:
    q=[r for r in validation if r['group']==group and int(r['step'])==step and r['appearance']==appearance]
    for ep in sorted({r['episode'] for r in q}):
     es=[r for r in q if r['episode']==ep];summaries.append(dict(group=group,step=step,appearance=appearance,episode=ep,**{k:float(np.mean([float(r[k]) for r in es])) for k in ['action_L1','translation_L1','rotation_L1','gripper_L1','auxiliary_L1']}))
 table(out/'per_demo_validation_summary.csv',summaries)
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 for s in SEEDS:
  q=[r for r in trains if r['group']==f'balanced_{s}'];axes[0].plot([int(r['step']) for r in q],[float(r['action_loss']) for r in q],alpha=.6,label=str(s))
  for app in ['clean','background']:
   vals=[np.mean([r['action_L1'] for r in summaries if r['group']==f'balanced_{s}' and r['step']==st and r['appearance']==app]) for st in [0,100,300,600]];axes[1].plot([0,100,300,600],vals,label=f'{s} {app}')
 axes[0].set_title('Training batch action L1, raw');axes[1].set_title('Fixed development action L1, episode equal');[ax.legend() for ax in axes];fig.tight_layout();fig.savefig(out/'learning_curves.png');plt.close(fig)
 assert all(sha(p)==h for p,h in read(out/'protected_inputs.json').items())
 save(out/'delivery_verification.json',dict(status='COMPLETE_EXECUTION_NOT_METHOD',policy_updates=1200,actual_successful_computations=1200,replay=0,valid_rollouts=120,invalid_rollouts=0,AE_updates=0,D_updates=0,sealed_reads=0,protected_hashes_unchanged=True,authorized_device_override='GPU1, see gpu1_authorization.json' if (out/'gpu1_authorization.json').exists() else None,isolated_preflight_optimizer_updates=0))
 lines=['# C sampling stability intervention','',
 'Same historical policy step0 and model training RNG for both new arms. Only balanced sampler order seed differs. This changes episode/window weights versus historical uniform-window sampling; it is not order-only versus history.','',
 'Historical sampling counts are reconstructed from frozen index files, checked against saved terminal configuration/cursor and update logs. Full per-item historical trace is unavailable; new runs record actual IDs for all600 updates.','',
 '| Group | Clean | Seen | Unseen |','|---|---|---|---|']
 for group in trialsets:
  rr=[r for r in rows if r['group']==group];lines.append('| '+group+' | '+' | '.join(f"{r['successes']}/20" for r in rr)+' |')
 rates=[float(arr[f'balanced_{s}'].mean()) for s in SEEDS]
 lines+=['',f'New order-seed equal-appearance rates: {rates}; report both, no winner selection.',
 'Two order seeds, one task, reused/tuned development states. Good performance in both is preliminary stabilization evidence only. If performance remains highly variable, the intervention is insufficient; no automatic further seed/loss search.',
 'No semantic phase oversampling. Relative-time deciles are not task-stage ground truth. Rollout stages retain historical diagnostic proxies, official success remains primary.',
 'See paired_statistics.json, per_demo_validation_summary.csv and raw curves. History includes all relevant C conditions, not only the worst baseline.',
 'Completed1200 updates/120 rollouts; AE/D updates0, sealed reads0, original inputs unchanged. Stop after this experiment.']
 with (out/'report.md').open('x') as f:f.write('\n'.join(lines)+'\n')
 print('SAMPLING_STABILITY_COMPLETE',flush=True)

def publish(out):
 dest=HERE/'delivery'/out.name;dest.mkdir(parents=True,exist_ok=False)
 files=['report.md','replication_protocol.json','historical_sampling_evidence.json','episode_sampling_statistics.csv','sampling_weight_change.csv','success_summary.csv','failure_stage_summary.csv','paired_statistics.json','actual_sampling_verification.json','delivery_verification.json','training_metrics.csv','per_demo_validation_summary.csv']
 files += [n for n in ['gpu1_authorization.json','execution_provenance.json','sampling_audit.md'] if (out/n).exists()]
 for name in files:
  s=(out/name).read_text().replace(str(R),'<PROJECT_ROOT>');s=re.sub(r'GPU-[0-9a-f-]{36}','<GPU_UUID>',s)
  assert len(s)<5_000_000 and not re.search(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN .*PRIVATE KEY',s)
  with (dest/name).open('x') as f:f.write(s)
 def git(*args,env=None):return subprocess.check_output(['git',*args],cwd=REPO,env=env,text=True).strip()
 branch='experiments/rla-sampling-stability';assert git('branch','--show-current')==branch
 rel=str(dest.relative_to(REPO));git('add','--',rel);assert all(x.startswith(rel+'/') for x in git('diff','--cached','--name-only').splitlines())
 git('-c','user.name=Lily021117','-c','user.email=Lily021117@users.noreply.github.com','commit','-m','Report two-order episode-balanced C sampling intervention')
 env=os.environ.copy();env.update(GH_CONFIG_DIR='/data/lyc/tools/github-cli/config',XDG_CACHE_HOME='/data/lyc/tmp/github-cli/cache',TMPDIR='/data/lyc/tmp/github-cli')
 auth=['-c','credential.helper=','-c','credential.helper=!/data/lyc/tools/github-cli/v2.103.0/gh_2.103.0_linux_amd64/bin/gh auth git-credential']
 git(*auth,'push','origin','HEAD:refs/heads/'+branch,env=env);commit=git('rev-parse','HEAD');assert git(*auth,'ls-remote','origin','refs/heads/'+branch,env=env).split()[0]==commit
 save(out/'github_sync_report.json',dict(status='PUSHED_VERIFIED',commit=commit,branch=branch,repository='https://github.com/Lily021117/rla-background-research'))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['prepare','run'],required=True);a=p.parse_args()
 if a.mode=='prepare':prepare(a.out)
 else:run(a.out)
