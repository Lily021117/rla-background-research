"""Allowlisted sanitized small artifacts, never checkpoints/data/credentials."""
import argparse,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stove_diagnosis'))
from common import save,sha
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();dest=Path(__file__).resolve().parent/'delivery'/a.out.name;dest.mkdir(parents=True,exist_ok=False)
allowed=['report.md','run_plan.json','training_config_D-old.json','training_config_D-center.json','target_center_audit.json','statistics_hashes.json','early_reproduction.json','initialization_and_reproduction_audit.json','paired_statistics.json','paired_results.csv','per_trial_results.csv','training_metrics.csv','offline_summary.csv','failure_stage_counts.csv','checkpoint_inventory.csv','evaluation_protocol.json','delivery_verification.json','gate_acceptance.json','mean_training_sources.csv']
allowed += ['diagnostics/isolated_smoke/'+n for n in ['gradient_contributions.csv','auxiliary_residual_energy.csv','verification.json']]
def sanitize(s):
 s=s.replace('/data/lyc/projects/rla-wm','${PROJECT_ROOT}').replace('/data/lyc/projects/vla-adapter-ref','${VLA_REFERENCE}').replace('/data/lyc','${DATA_ROOT}')
 assert not re.search(r'(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN .*PRIVATE KEY-----)',s)
 return s
inventory=[]
for name in allowed:
 src=a.out/name;assert src.is_file() and src.stat().st_size<5_000_000
 target=dest/src.name;assert not target.exists();target.write_text(sanitize(src.read_text()));inventory.append(dict(name=target.name,bytes=target.stat().st_size,sha256=sha(target)))
save(dest/'publication_manifest.json',dict(allowlist=inventory,excluded=['model weights','raw images','sidecar arrays','raw trajectories','videos','credentials'],historical_files_unchanged=True))
print(dest)
