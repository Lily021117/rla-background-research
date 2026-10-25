"""Allowlisted small sanitized continuation artifacts, no weights or raw rollouts."""
import argparse,json,re,hashlib,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();o=a.out
assert json.loads((o/'delivery_verification.json').read_text())['valid_rollouts']==120
dest=Path(__file__).parent/'delivery/extend600_run_20261025_01';dest.mkdir(parents=True,exist_ok=False)
names=['report.md','run_plan.json','continuation_protocol.json','evaluation_protocol.json','statistical_protocol.json','recovery_precheck.json','replay_equivalence_audit.json','preupdate_incident.json','attempt_registry.json','lr_curves.csv','training_metrics_all.csv','offline_continuation_summary.csv','offline_200_to600.csv','per_demo_metrics.csv','coarse_phase_metrics.csv','checkpoint_inventory.csv','per_trial_results.csv','per_episode_phase_summary.csv','success_rates.csv','historical_success_rates.csv','historical_comparability_audit.json','paired_results.csv','paired_statistics.json','stage_summary.csv','delivery_verification.json']+[p.name for p in o.glob('training_config_*.json')]
secret=re.compile(rb'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----');inventory=[]
names.append('execution_code_version.json')
names += ['attempt_registry_v3.json','policy_registry_v3.json','cuda_incident.json','gpu0_resume_authorization.json']
for name in names:
 text=(o/name).read_text()
 for old,new in [('/data/lyc/projects/rla-wm','${PROJECT_ROOT}'),('/data/lyc/projects/LIBERO-plus','${LIBERO_ROOT}'),('/data/lyc','${DATA_ROOT}')]:text=text.replace(old,new)
 text=re.sub(r'GPU-[0-9a-fA-F-]{36}','${GPU_UUID}',text);data=text.encode();assert not secret.search(data) and len(data)<5_000_000 and '/root/' not in text,name
 with (dest/name).open('xb') as f:f.write(data)
 inventory.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
(dest/'figures').mkdir()
for path in (o/'figures').iterdir():
 assert path.suffix in ('.png','.pdf');data=path.read_bytes();assert len(data)<5_000_000
 with (dest/'figures'/path.name).open('xb') as f:f.write(data)
 inventory.append(dict(path='figures/'+path.name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
with (dest/'publication_manifest.json').open('x') as f:json.dump(dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),files=inventory,excluded=['weights','raw images','raw trajectories','videos','sidecar arrays','credentials']),f,indent=2)
print('SANITIZED_CONTINUATION_READY',dest,len(inventory),flush=True)
