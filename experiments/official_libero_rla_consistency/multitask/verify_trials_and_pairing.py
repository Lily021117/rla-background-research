"""CPU artifact/composite-state pairing audit; no rollout or model call."""
import argparse,csv,json,collections
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--history',type=Path,required=True);p.add_argument('--completion',type=Path,required=True);p.add_argument('--delivery',type=Path,required=True);a=p.parse_args();rows=[];paired=[];checks=[]
for task,root,historical in [('cookie_historical',a.history,True),('drawer',a.run/'drawer/policy_run',False),('stove',a.run/'stove/policy_run',False)]:
 trials=json.loads((root/'per_trial_results.json').read_text());assert len(trials)==180;groups=collections.defaultdict(list);model_hashes=collections.defaultdict(set)
 for r in trials:
  assert r['status']=='COMPLETE' and r['geometry_pair_check'];condition='unseen_background' if r['condition'].startswith('unseen') else r['condition'];model='B' if r['model']=='B20' else r['model'];groups[r['seed']].append((model,condition,r));model_hashes[model].add(r['checkpoint_sha256']);rows.append(dict(task=task,historical=historical,original_condition=r['condition'],**r))
  if not historical:
   video=Path(r['video']);trajectory=Path(r['trajectory']);assert video.is_file() and video.stat().st_size>0 and trajectory.is_file()
   with np.load(trajectory) as d:
    assert len(d['executed_actions'])==len(d['states'])==len(d['eef_positions'])==len(d['target_bowl_positions'])==r['steps']
    assert d['executed_actions'].shape==(r['steps'],7) and len(d['predicted_chunks'])==(r['steps']+7)//8
    assert all(np.isfinite(d[k]).all() for k in d.files)
   checks.append(dict(task=task,model=model,seed=r['seed'],condition=condition,steps=r['steps'],video_bytes=video.stat().st_size,trajectory_bytes=trajectory.stat().st_size,finite_trajectory=True))
 assert len(groups)==20 and all(len(x)==1 for x in model_hashes.values())
 for seed,rs in sorted(groups.items()):
  assert len(rs)==9 and len({(m,c) for m,c,r in rs})==9 and len({r['state_sha256'] for m,c,r in rs})==1
  paired.append(dict(task=task,seed=seed,physical_state_sha256=rs[0][2]['state_sha256'],**{m+'_'+c:int(r['success']) for m,c,r in rs}))
def csvout(name,values):
 fields=list(dict.fromkeys(k for r in values for k in r))
 with (a.completion/name).open('x') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(values)
csvout('per_trial_results_combined.csv',rows);csvout('paired_state_results.csv',paired);csvout('trial_artifact_audit.csv',checks)
receipt=dict(status='PASS',new_valid_trials=len(checks),historical_valid_trials=180,physical_states_new=40,physical_states_historical=20,pairing_completeness=True,same_state_hash_across9_trials=True,video_files_present_nonempty=True,new_numeric_trajectories_finite=True,video_full_decode_claim=False,model_calls=0,optimizer_updates=0,new_rollouts=0)
with (a.completion/'trial_artifact_verification.json').open('x') as f:json.dump(receipt,f,indent=2)
for name in ('per_trial_results_combined.csv','paired_state_results.csv','trial_artifact_audit.csv','trial_artifact_verification.json'):
 text=(a.completion/name).read_text().replace('/data/lyc/projects/rla-wm','${PROJECT_ROOT}').replace('/data/lyc','${DATA_ROOT}')
 with (a.delivery/name).open('x') as f:f.write(text)
print(json.dumps(receipt,indent=2))
