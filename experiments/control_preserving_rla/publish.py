"""Allowlisted source-derived small summaries; never publish supervision arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import re

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
dest=Path(__file__).parent/'delivery'/a.out.parent.name;dest.mkdir(parents=True,exist_ok=False)
names=['report.md','run_plan.json','evaluation_protocol.json','source_lineage.json','initialization_acceptance.json','four_student_initialization_audit.json','gate_acceptance.json','teacher_cache_metadata.json','teacher_disagreement_summary.csv','success_rates.csv','paired_results.csv','paired_statistics.json','historical_protocol_audit.json','stage_metrics_five_arms.csv','stage_condition_summary.csv','stage_analysis_limitations.json','validation_summary.csv','validation_phase_metrics.csv','training_contribution_summary.csv','checkpoint_inventory.csv','delivery_verification.json']
names += ['training_config_'+arm+'.json' for arm in ('E-C','E-D','E-C-keep','E-D-keep')]
inventory=[]
for name in names:
 path=a.out/name;assert path.stat().st_size<2_000_000
 text=path.read_text().replace('/data/lyc/projects/rla-wm','${PROJECT_ROOT}').replace('/data/lyc/projects/LIBERO-plus','${LIBERO_ROOT}').replace('/data/lyc','${DATA_ROOT}')
 if name=='delivery_verification.json':
  value=json.loads(text);value.pop('raw_gpu_process_snapshot',None);text=json.dumps(value,indent=2)
 text=re.sub(r'GPU-[0-9a-fA-F-]{36}','${GPU_UUID_REDACTED}',text)
 assert not re.search(r'(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)',text)
 with (dest/name).open('x') as f:f.write(text)
 inventory.append(dict(name=name,bytes=len(text.encode()),sha256=hashlib.sha256(text.encode()).hexdigest()))
with (dest/'publication_manifest.json').open('x') as f:json.dump(dict(files=inventory,excluded=['checkpoints','raw images','teacher action cache','sidecars','videos','raw trajectories','credentials']),f,indent=2)
print(dest)
