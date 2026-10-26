"""Small allowlisted reproducibility artifacts; no weights/data/videos/sidecar arrays."""
import argparse,json,re,hashlib,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();o=a.out
v=json.loads((o/'delivery_verification.json').read_text());assert v['AE_successful_updates']==v['policy_successful_updates']==600 and v['valid_rollouts']==60
dest=Path(__file__).parent/'delivery'/o.name;dest.mkdir(parents=True,exist_ok=False)
names=['report.md','experiment_protocol.json','run_plan.json','evaluation_protocol.json','historical_AE_training_plan.json','historical_policy_plan.json','normalization.json','policy_LR_curve.csv','target_readiness.json','historical_comparability.csv','success_rates.csv','paired_results.csv','paired_statistics.json','failure_stage_summary.csv','AE_reconstruction_comparison.csv','AE_distribution_comparison.csv','AE_paired_distance_comparison.csv','AE_decoder_ablation_comparison.csv','AE_representation_summary.csv','AE_training_curves.csv','policy_training_metrics.csv','policy_validation_metrics.csv','policy_validation_summary.csv','checkpoint_inventory.csv','delivery_verification.json','per_trial_results.csv','ae_run/training_plan.json','ae_run/official_initialization.json','ae_run/data_manifest.json','ae_run/sidecar_provenance.json','ae_run/sidecar_alignment_audit.json','ae_run/raw_source_recompute_audit.json','ae_run/representation_acceptance.json','ae_run/AE_D_inv03/sampling_identity_check.json','ae_run/AE_D_inv03/acceptance.json','D-inv03/formal/initialization_audit.json','D-inv03/formal/training_acceptance.json','execution_provenance.json']
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
with (dest/'publication_manifest.json').open('x') as f:json.dump(dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),files=inventory,excluded=['checkpoints','raw images','raw trajectories','videos','sidecar arrays','credentials']),f,indent=2)
print('READY_TO_COMMIT',dest,len(inventory))
