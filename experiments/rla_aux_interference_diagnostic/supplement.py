"""Read-only D-center log context and explicit computation incident ledger."""
import csv,hashlib,json,sys
from pathlib import Path
out=Path(sys.argv[1]);root=Path('/data/lyc/projects/rla-wm/experiments/rla_sampling_stability/run_20261027_02')
records=[]
for name in ('training_metrics.csv','validation_metrics.csv'):
 p=root/name;content=p.read_bytes()
 with (out/('D_center_context_'+name)).open('xb') as f:f.write(content)
 records.append(dict(path=str(p),sha256=hashlib.sha256(content).hexdigest(),purpose='existing logs only; no D checkpoint forward or gradient',rows=len(content.splitlines())-1))
with (out/'supplemental_provenance.json').open('x') as f:json.dump(records,f,indent=2)
with (out/'precompute_incident.json').open('x') as f:json.dump(dict(attempt='diagnostic',failure='SyntaxError importing new worker; unmatched parenthesis',evidence='diagnostic.log',policy_forward=0,backward=0,gradient_extractions=0,optimizer_updates=0,AE_updates=0,rollouts=0,model_construction_only=True,retry='diagnostic_retry1',protocol_unchanged=True,initialization_restart_not_training_replay=True,infrastructure_retries=0,implementation_launch_retries=1),f,indent=2)
