"""Freeze coverage-only training and development trials before execution."""
import argparse,json,hashlib,subprocess
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False)
def save(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2)
manifest=a.dataset/'frozen_dataset_manifest.json';stats=a.dataset/'normalization_statistics_expanded_train_only.json';m=json.loads(manifest.read_text());assert m['status']=='PASS'
old=['sealed_163_demo_12','sealed_164_demo_13','sealed_165_demo_14','sealed_169_demo_18'];train=[e for e in m['episodes'] if e['split']=='train'];dev=[e for e in m['episodes'] if e['split']=='validation'];assert len(train)==20 and len(dev)==5 and set(old)<=set(e['episode'] for e in train)
assert not set(e['source_demo'] for e in train)&set(e['source_demo'] for e in dev)
trials=[]
for e in dev:
    root=Path(e['root']);pro=json.loads((root/'trajectory_provenance.json').read_text());xml=next(Path(p) for p in pro['source_hashes'] if Path(p).name=='clean.xml').parent;assert (xml/'background.xml').exists()
    with np.load(root/'trajectory_labels.npz') as lab:state=lab['states'][0].copy();assert np.isfinite(state).all()
    for appearance in ('clean','background'):
        for arm in ('B4','B20'):trials.append(dict(arm=arm,episode=e['episode'],source_demo=e['source_demo'],root=str(root),xml_directory=str(xml),appearance=appearance,state_index=0,state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),task=e['task']))
protocol=dict(frozen=True,trials=trials,total_trials=20,max_steps=220,seed=0,chunk_execution=8,reflection=1,MSAA=4,success='env.check_success()',initial_settling_steps=0,retries=0,invalid_trial_rule='stop and report; do not silently retry or count as failure',role='development_not_independent_test')
save(a.out/'evaluation_protocol.json',protocol)
for arm in ('B4','B20'):
    out=a.out/arm;out.mkdir();episodes=old if arm=='B4' else [e['episode'] for e in train];n=sum(e['valid_chunks'] for e in train if e['episode'] in episodes)
    plan=dict(frozen=True,arm=arm,seed=20261023,updates=600,checkpoint_steps=[100,300,600],validation_steps=[0,100,300,600],final_checkpoint=600,manifest=str(manifest),statistics=str(stats),statistics_sha256=hashlib.sha256(stats.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),train_episodes=episodes,common_B4=old,development_episodes=[e['episode'] for e in dev],batch_size=8,accumulation=2,physical_exposures=4800,appearance_exposures=9600,equivalent_window_epochs=4800/n,sampling='repeat seeded permutation of eligible physical windows; sequential clean/background per window; same algorithm/seed, different actual sequences',lr=2e-4,optimizer='AdamW',betas=[.9,.999],eps=1e-8,weight_decay=.01,scheduler='MultiStepLR milestone600 gamma0.1; no reduced-LR updates before final',backend='Math SDPA',dtype='bfloat16',loss='unweighted official mean L1,8x7',initialization='fresh official Prismatic and common seeded heads; no prior trained policy',budget_reason='previous270-step curve still falls in final segment;600 gives B20 2.54 window epochs and B4 11.14, fixed compute not convergence',stop_conditions=['OOM','nonfinite loss/gradient/parameters','ECC increment','checkpoint/data mismatch'],RLA=False)
    save(out/'training_plan.json',plan);save(out/'normalization_statistics_train_only.json',json.loads(stats.read_text()))
save(a.out/'training_plan.json',dict(frozen=True,arms=['B4','B20'],updates_per_arm=600,checkpoint_steps=[0,100,300,600],evaluation_protocol=protocol,primary_change='independent training episode coverage at fixed optimizer compute',common_statistics='20train only; B4 shares aggregate statistics, not strictly unseen aggregate information',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
print(a.out)
