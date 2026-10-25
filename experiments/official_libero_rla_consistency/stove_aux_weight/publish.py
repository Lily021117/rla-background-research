"""Allowlisted, sanitized small artifacts only; git push remains explicit."""
import argparse,re,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();dest=Path(__file__).parent/'delivery'/(a.out.name+'_completed');dest.mkdir(parents=True,exist_ok=False)
names=['report.md','run_plan.json','configuration_diff.json','evaluation_protocol.json','initialization_acceptance.json','target_identity_audit.json','success_rates.csv','paired_results.csv','paired_statistics.json','historical_protocol_audit.json','stage_metrics_four_arms.csv','stage_condition_summary.csv','stage_analysis_limitations.json','validation_phase_metrics.csv','action_dimension_summary.csv','training_contribution_summary.csv','checkpoint_inventory.csv','delivery_verification.json']
inventory=[]
for name in names:
 source={'report.md':'completion_report.md','checkpoint_inventory.csv':'checkpoint_inventory_completed.csv','delivery_verification.json':'delivery_verification_completed.json'}.get(name,name)
 p=a.out/source;assert p.stat().st_size<2_000_000
 s=p.read_text().replace('/data/lyc/projects/rla-wm','${PROJECT_ROOT}').replace('/data/lyc/projects/LIBERO-plus','${LIBERO_ROOT}').replace('/data/lyc','${DATA_ROOT}')
 if name=='delivery_verification.json':
  obj=json.loads(s);obj.pop('gpu_resources_after',None);s=json.dumps(obj,indent=2)
 s=re.sub(r'GPU-[0-9a-fA-F-]{36}', '${GPU_UUID_REDACTED}',s)
 assert not re.search(r'(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)',s)
 with (dest/name).open('x') as f:f.write(s)
 inventory.append(dict(name=name,bytes=len(s.encode()),sha256=hashlib.sha256(s.encode()).hexdigest()))
with (dest/'publication_manifest.json').open('x') as f:json.dump(dict(files=inventory,excluded=['weights','raw images','sidecars','videos','raw trajectories','credentials']),f,indent=2)
print(dest)
