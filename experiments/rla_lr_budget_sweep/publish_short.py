"""Allowlisted sanitized small research artifacts; no weights or raw observations."""
import argparse,json,hashlib,re,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();o=a.out
assert json.loads((o/'delivery_verification.json').read_text())['new_closed_loop']==0
dest=Path(__file__).parent/'delivery/short200_run_20261025_01';dest.mkdir(parents=True,exist_ok=False)
names=['report.md','next_experiment_proposal.md','short_protocol.json','run_plan.json','interruption_audit.json','loss_definition_audit.json','fixed_sample_manifest.json','data_exposure.csv','training_metrics_all.csv','fixed_offline_summary.csv','per_demo_metrics.csv','horizon_metrics.csv','coarse_phase_metrics.csv','stage_label_limitations.json','gradient_contributions.csv','parameter_update_statistics.csv','learning_signals.csv','matched_C_D_offline_comparison.csv','checkpoint_inventory.csv','delivery_verification.json','AE_training_curves.csv','AE_development_reconstruction.csv','AE_latent_distribution.csv','AE_training_latent_means.csv','historical_motion_probe_summary.csv','historical_motion_probe_uncertainty.json','historical_optimizer_audit.json']
names += [x.name for x in o.glob('training_config_*.json')]
names += ['AE_summary.csv']
names += ['diagnostic_incident.json','diagnostic_registry.json']
names += ['optimizer_runtime_summary.csv']
names += ['interval_learning_signals.csv']
secret=re.compile(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')
inventory=[]
for n in names:
 text=(o/n).read_text();assert not secret.search(text),n
 for prefix,placeholder in [('/data/lyc/projects/rla-wm','${PROJECT_ROOT}'),('/data/lyc/projects/LIBERO-plus','${LIBERO_ROOT}'),('/data/lyc','${DATA_ROOT}')]:text=text.replace(prefix,placeholder)
 text=re.sub(r'GPU-[0-9a-fA-F-]{36}','${GPU_UUID}',text);assert '/root/' not in text
 data=text.encode();assert len(data)<5_000_000,n
 with (dest/n).open('xb') as f:f.write(data)
 inventory.append(dict(file=n,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
(dest/'figures').mkdir()
for source in (o/'figures').iterdir():
 assert source.suffix in ('.png','.pdf');data=source.read_bytes();assert len(data)<5_000_000
 with (dest/'figures'/source.name).open('xb') as f:f.write(data)
 inventory.append(dict(file='figures/'+source.name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
with (dest/'publication_manifest.json').open('x') as f:json.dump(dict(files=inventory,excluded=['weights','sidecars','original observations','raw trajectory','videos','credentials'],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),f,indent=2)
print('SANITIZED_PUBLICATION_READY',dest,len(inventory),flush=True)
