"""Two bounded C crossover cells, queued after completed policy-seed replication.

No new seed selection: cross historical 20261023 and the previously identified
20261029 run. Parameters and training RNG are taken from verified step0 files,
never from trained policies. All original modules/optimizer/loader stay intact.
"""
import argparse, collections, csv, fcntl, hashlib, json, os, re, subprocess, time
from pathlib import Path
import numpy as np
import torch

R = Path('/data/lyc/projects/rla-wm')
PREVIOUS = R/'experiments/rla_policy_seed_replication/run_20261026_02'
HISTORY = R/'experiments/official_libero_rla_consistency/run_20261023_10/stove/policy_run'
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GPU = 'GPU-205a3f5a-d91b-6385-c588-4e4cf3505f7e'
RNG_KEYS = ['torch_rng', 'cuda_rng', 'python_rng', 'numpy_rng']

def read(p): return json.loads(Path(p).read_text())
def save(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f: json.dump(value, f, indent=2)
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()
def param_hash(d):
    h = hashlib.sha256()
    for module in sorted(d):
        for name, value in sorted(d[module].items()):
            h.update((module+'.'+name).encode())
            h.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def prepare(out):
    base = read(PREVIOUS/'seed_20261029/run_plan.json')
    oldplan = read(HISTORY/'run_plan.json')
    paths = {20261023: Path(base['original_common_step0']),
             20261029: PREVIOUS/'seed_20261029/common_seed_step0.pt'}
    expected = {20261023: base['original_common_step0_sha256'],
                20261029: base['common_step0_sha256']}
    checkpoints = {}
    for seed, path in paths.items():
        assert sha(path) == expected[seed]
        checkpoints[seed] = torch.load(path, map_location='cpu')
        assert checkpoints[seed]['global_step'] == 0
        assert all(k in checkpoints[seed] for k in RNG_KEYS)
    orders = {20261023: Path(oldplan['inputs']['B20_sampling_order']),
              20261029: Path(base['inputs']['B20_sampling_order'])}
    assert all(len(read(p)) == 4800 for p in orders.values())
    assert {k: {n: tuple(v.shape) for n,v in d.items()} for k,d in checkpoints[20261023]['parameters'].items()} == {k: {n: tuple(v.shape) for n,v in d.items()} for k,d in checkpoints[20261029]['parameters'].items()}
    cells = [(20261023, 20261029), (20261029, 20261023)]
    protocol = dict(experiment='C initialization x training randomness crossover',
        seeds=[20261029,20261023], cells=[dict(initialization=i,training_rng=t) for i,t in cells],
        fixed_before_new_results=True, methods=['C'], updates_per_cell=600,
        total_new_updates=1200, new_valid_rollouts=120, AE_updates=0,
        historical_cells=['I23_T23','I29_T29'], selection='previously discussed seed29; not selected using seed30 outcome',
        only_variables='initial step0 parameter donor and saved training RNG/sample-order donor',
        LR=0.0002, lambda_aux=0.22408536585365854,
        scheduler='MultiStepLR[600], gamma=.1, AFTER optimizer; all600 updates2e-4',
        AE_target='frozen historical AE-C, original C mean and scale',
        optimizer='AdamW', batch=16, microbatch=8, accumulation=2,
        backend='Math SDPA', dtype='BF16', activation_checkpointing=True,
        checkpoint_selection='final600 only', keep=False, teacher=False,
        sealed_reads=0, no_automatic_extra_runs=True,
        waiting_for=str(PREVIOUS/'github_sync_report.json'),
        interpretation='2x2 development diagnostic, not independent test; history may differ in hardware/time; no unique-root-cause claim')
    save(out/'replication_protocol.json',protocol)
    target_audit=read(PREVIOUS/'frozen_target_audit.json');assert target_audit['status']=='PASS'
    target_audit['reused_from']=str(PREVIOUS/'frozen_target_audit.json')
    save(out/'frozen_target_audit.json',target_audit)
    for i,t in cells:
        root=out/f'seed_{t}';root.mkdir()
        ck=dict(parameters=checkpoints[i]['parameters'],global_step=0,next_physical_sample=0,
                optimizer_updates=0,initialization_seed=i,training_rng_seed=t,
                **{k:checkpoints[t][k] for k in RNG_KEYS})
        dest=root/'cross_step0.pt';torch.save(ck,dest)
        loaded=torch.load(dest,map_location='cpu')
        assert param_hash(loaded['parameters'])==param_hash(checkpoints[i]['parameters'])
        assert torch.equal(loaded['torch_rng'],checkpoints[t]['torch_rng'])
        assert all(torch.equal(a,b) for a,b in zip(loaded['cuda_rng'],checkpoints[t]['cuda_rng']))
        import pickle
        for k in ['python_rng','numpy_rng']:assert pickle.dumps(loaded[k])==pickle.dumps(checkpoints[t][k])
        plan=json.loads(json.dumps(base));plan.update(common_step0=str(dest),common_step0_sha256=sha(dest),
            initialization_scope='exact_parameter_donor_crossed_with_exact_training_RNG_donor',
            initialization_seed=i,policy_seed=t,arms=['C'],closed_loop_trials=60)
        # Preserve audited trainer experiment tag to retain exact initialization assertions.
        plan['causal_design']='C initialization-training RNG crossover'
        save(root/'sampling_order.json',read(orders[t]))
        plan['inputs']['B20_sampling_order']=str(root/'sampling_order.json')
        plan['inputs']['B20_sampling_sha256']=sha(root/'sampling_order.json')
        plan['shared_protocol']['seed']=t
        save(root/'run_plan.json',plan)
        save(root/'normalization.json',read(PREVIOUS/'seed_20261029/normalization.json'))
        save(root/'preflight6/isolated_smoke/preflight_acceptance.json',read(PREVIOUS/'seed_20261029/preflight6/isolated_smoke/preflight_acceptance.json'))
        save(root/'initialization_gate.json',dict(status='PASS',check='CPU exact parameter/RNG donor composition; inherited accepted structure, not a new forward claim',
             parameter_donor=str(paths[i]),parameter_donor_sha256=expected[i],
             RNG_donor=str(paths[t]),RNG_donor_sha256=expected[t],parameter_hash=param_hash(loaded['parameters']),
             exact_RNG_equal=True,step0=True,no_optimizer_state_reused=True))
        ep=read(PREVIOUS/'seed_20261029/evaluation_protocol.json')
        ep.update(models=['C_s600'],trials=60,role=f'C crossover I{i}_T{t}, reused development, not independent test')
        save(root/'evaluation_protocol.json',ep)
        save(root/'policy_registry.json',{'C_s600':str(root/'C/formal/step0600.pt')})
    save(out/'source_protection.json',{str(p):sha(p) for p in [*paths.values(),*orders.values(),PREVIOUS/'frozen_target_audit.json']})
    print('CROSSOVER_PREPARED_ZERO_UPDATES',flush=True)

def wait_previous():
    while not (PREVIOUS/'github_sync_report.json').exists():
        print('WAIT_PREVIOUS_REPORT_AND_PUBLICATION',flush=True);time.sleep(30)
    v=read(PREVIOUS/'delivery_verification.json')
    assert v['policy_successful_updates']==2400 and v['valid_new_rollouts']==240
    assert read(PREVIOUS/'github_sync_report.json')['status']=='PUSHED_VERIFIED'

def run(out):
    lock=(out/'queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(out/'queue_started.json',dict(pid=os.getpid(),state='WAIT_PREVIOUS',dependency=str(PREVIOUS),GPU='GPU0 only',new_updates_budget=1200,new_rollouts_budget=120))
    wait_previous()
    save(out/'dependency_completion.json',read(PREVIOUS/'github_sync_report.json'))
    for stage in ['C','evaluate']:
        for seed in [20261029,20261023]:
            while int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))<26624:
                print('WAIT_GPU0_26GiB',stage,seed,flush=True);time.sleep(30)
            ecc=int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True))
            assert ecc==0,'GPU0 ECC changed: no automatic fallback'
            subprocess.run(['python',str(HERE.parent/'rla_policy_seed_replication/launch.py'),'--out',str(out),'--seed',str(seed),'--stage',stage],cwd=REPO,check=True)
    analyze(out)
    publish(out)

def publish(out):
    """Only small allowlisted sanitized metadata; never checkpoints or trajectories."""
    dest=HERE/'delivery'/out.name;dest.mkdir(parents=True,exist_ok=False)
    paths=[out/n for n in ['replication_protocol.json','dependency_completion.json','source_protection.json','paired_statistics.json','success_summary.csv','delivery_verification.json','report.md']]
    for seed in [20261029,20261023]:
        root=out/f'seed_{seed}'
        paths += [root/n for n in ['run_plan.json','initialization_gate.json','evaluation_protocol.json','per_trial_results.json','C/formal/initialization_audit.json','C/formal/execution_configuration.json','C/formal/training_acceptance.json','C/formal/training_metrics.csv']]
    for path in paths:
        assert path.stat().st_size<5_000_000
        s=path.read_text().replace(str(R),'<PROJECT_ROOT>')
        s=re.sub(r'GPU-[0-9a-f-]{36}','<GPU_UUID>',s)
        assert not re.search(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN .*PRIVATE KEY',s)
        target=dest/path.relative_to(out);target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('x') as f:f.write(s)
    def git(*args,env=None):return subprocess.check_output(['git',*args],cwd=REPO,env=env,text=True).strip()
    assert git('branch','--show-current')=='experiments/rla-initialization-rng-cross'
    rel=str(dest.relative_to(REPO));git('add','--',rel)
    assert all(p.startswith(rel+'/') for p in git('diff','--cached','--name-only').splitlines())
    git('-c','user.name=Lily021117','-c','user.email=Lily021117@users.noreply.github.com','commit','-m','Report C initialization and training randomness crossover')
    env=os.environ.copy();env.update(GH_CONFIG_DIR='/data/lyc/tools/github-cli/config',XDG_CACHE_HOME='/data/lyc/tmp/github-cli/cache',TMPDIR='/data/lyc/tmp/github-cli')
    auth=['-c','credential.helper=','-c','credential.helper=!/data/lyc/tools/github-cli/v2.103.0/gh_2.103.0_linux_amd64/bin/gh auth git-credential']
    branch='experiments/rla-initialization-rng-cross'
    git(*auth,'push','origin','HEAD:refs/heads/'+branch,env=env)
    commit=git('rev-parse','HEAD');assert git(*auth,'ls-remote','origin','refs/heads/'+branch,env=env).split()[0]==commit
    save(out/'github_sync_report.json',dict(status='PUSHED_VERIFIED',commit=commit,branch=branch,repository='https://github.com/Lily021117/rla-background-research'))

def analyze(out):
    definitions=[('I23_T23',HISTORY),('I29_T29',PREVIOUS/'seed_20261029'),
                 ('I23_T29',out/'seed_20261029'),('I29_T23',out/'seed_20261023')]
    cells={};summary=[]
    for cell,root in definitions:
        rows=[r for r in read(root/'per_trial_results.json') if r['model'] in ('C','C_s600')]
        assert len(rows)==60 and all(r['status']=='COMPLETE' for r in rows)
        cells[cell]={(r['seed'],r['condition']):r for r in rows}
        for cond in ['clean','seen_background','unseen_background']:
            rr=[r for r in rows if r['condition']==cond]
            summary.append(dict(cell=cell,condition=cond,successes=sum(r['success'] for r in rr),trials=len(rr)))
    keys=sorted(cells['I23_T23']);assert all(sorted(c)==keys for c in cells.values())
    for k in keys:
        for c in cells.values():
            for f in ['state_sha256','xml_sha256','language']:assert c[k][f]==cells['I23_T23'][k][f]
    states=sorted({s for s,c in keys});conds=['clean','seen_background','unseen_background']
    arrays={c:np.array([[int(rs[(s,k)]['success']) for k in conds] for s in states]) for c,rs in cells.items()}
    rng=np.random.RandomState(20261026);indices=rng.randint(0,len(states),(20000,len(states)));comparisons={}
    for name,a,b in [('init_effect_training23','I29_T23','I23_T23'),('init_effect_training29','I29_T29','I23_T29'),('training_effect_init23','I23_T29','I23_T23'),('training_effect_init29','I29_T29','I29_T23')]:
        d=arrays[a]-arrays[b];boot=d[indices].mean(axis=(1,2))
        comparisons[name]=dict(difference=float(d.mean()),state_cluster_CI95=np.quantile(boot,[.025,.975]).tolist(),gains=int((d>0).sum()),losses=int((d<0).sum()))
    d=arrays['I29_T29']-arrays['I23_T29']-arrays['I29_T23']+arrays['I23_T23']
    comparisons['interaction']=dict(difference=float(d.mean()),state_cluster_CI95=np.quantile(d[indices].mean(axis=(1,2)),[.025,.975]).tolist())
    save(out/'paired_statistics.json',comparisons)
    with (out/'success_summary.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    rows=[]
    for t in [20261029,20261023]:
        root=out/f'seed_{t}'
        assert read(root/'C/formal/training_acceptance.json')['successful_updates']==600
        rr=list(csv.DictReader((root/'C/formal/training_metrics.csv').open()));assert len(rr)==600
        assert all(float(r['lr'])==.0002 for r in rr);rows.extend(rr)
    assert all(sha(p)==h for p,h in read(out/'source_protection.json').items())
    save(out/'delivery_verification.json',dict(status='COMPLETE',policy_updates=1200,AE_updates=0,valid_new_rollouts=120,sealed_reads=0,source_hashes_unchanged=True,history_reused=True,automatic_additional_experiments=False))
    lines=['# C initialization / training randomness crossover','',
           'Two new crossover cells; historical diagonal cells reused. All results are reused-development diagnostics, not independent tests. Initial parameters and saved RNG/order were crossed, not regenerated. Other hyperparameters unchanged.','',
           '| Cell | Clean | Seen | Unseen |','|---|---|---|---|']
    for c in cells:
        rs=[r for r in summary if r['cell']==c];lines.append('| '+c+' | '+' | '.join(f"{r['successes']}/{r['trials']}" for r in rs)+' |')
    lines+=['','Effects are new minus historical level, conditional on the other factor. Negative initialization effects indicate the new initialization is worse at that fixed training RNG. Interaction means the two effects need not be additive.','',json.dumps(comparisons,indent=2),'',
            'Only two factor levels, one task and previously tuned development states. No unique-cause claim; hardware/time differences in historical cells remain a limitation. No AE updates, loss changes or post-result extension.']
    with (out/'report.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print('CROSSOVER_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['prepare','run'],required=True);a=p.parse_args()
    if a.mode=='prepare':prepare(a.out)
    else:run(a.out)
