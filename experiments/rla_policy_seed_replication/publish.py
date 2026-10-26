"""Publish only allowlisted, sanitized small diagnostics after all trials finish."""
import argparse,json,re,hashlib,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();o=a.out
v=json.loads((o/'delivery_verification.json').read_text())
assert v['policy_successful_updates']==2400 and v['AE_updates']==0 and v['valid_new_rollouts']==240
dest=Path(__file__).parent/'delivery'/o.name;dest.mkdir(parents=True,exist_ok=False)
names=['report.md','replication_protocol.json','execution_provenance.json','cross_seed_initialization_audit.json','initialization_matching_audit.json','frozen_target_audit.json','success_rates.csv','paired_results.csv','paired_statistics.json','failure_stage_summary.csv','training_metrics.csv','validation_metrics.csv','checkpoint_inventory.csv','per_trial_results.csv','delivery_verification.json']
for seed in json.loads((o/'replication_protocol.json').read_text())['seeds']:
 prefix=f'seed_{seed}/'
 names += [prefix+n for n in ['run_plan.json','normalization.json','evaluation_protocol.json','sampling_audit.json','LR_curve.csv','initialization_gate.json','module_initialization_audit.json']]
 for arm in ['C','D-center']:
  names += [prefix+arm+'/formal/'+n for n in ['execution_configuration.json','initialization_audit.json','training_acceptance.json']]
inventory=[]
for name in names:
 s=(o/name).read_text()
 for old,new in [('/data/lyc/projects/rla-wm','${PROJECT_ROOT}'),('/data/lyc/projects/LIBERO-plus','${LIBERO_ROOT}'),('/data/lyc','${DATA_ROOT}')]:s=s.replace(old,new)
 s=re.sub(r'GPU-[a-fA-F0-9-]{36}','${GPU_UUID}',s);assert '/root/' not in s
 data=s.encode();assert len(data)<5_000_000
 assert not re.search(rb'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN .*PRIVATE KEY',data)
 target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
 with target.open('xb') as f:f.write(data)
 inventory.append(dict(path=name,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
(dest/'figures').mkdir()
for path in (o/'figures').glob('*.png'):
 data=path.read_bytes();assert len(data)<5_000_000
 with (dest/'figures'/path.name).open('xb') as f:f.write(data)
 inventory.append(dict(path='figures/'+path.name,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
with (dest/'publication_manifest.json').open('x') as f:json.dump(dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),files=inventory,excluded=['weights','raw images','raw trajectories','videos','sidecar arrays','credentials']),f,indent=2)
print('READY_TO_COMMIT',dest,len(inventory))
