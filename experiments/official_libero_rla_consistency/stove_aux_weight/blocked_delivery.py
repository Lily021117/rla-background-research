"""Preserve an honest partial delivery after CUDA failure, no retry."""
import argparse,json,hashlib,csv,subprocess,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();o=a.out
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
checks=[]
for r in json.loads((o/'protected_inputs.json').read_text()):
 unchanged=sha(Path(r['path']))==r['sha256'];assert unchanged;checks.append(dict(path=r['path'],unchanged=unchanged))
incident=json.loads((o/'cuda_incident.json').read_text())
v=dict(status='BLOCKED_CUDA_EXECUTION_FAILURE',policy_updates={'C-low':300,'D-center-low':300},actual_computed_updates={'C-low':300,'D-center-low':300},replayed_updates=0,AE_updates=0,new_closed_loop=0,final600_checkpoint_available=False,protected_inputs=checks,sealed_reads=0,ECC_increment=False,own_GPU_processes_released=True,automatic_retry=False,automatic_GPU_switch=False,incident=incident,source_changes='new stove_aux_weight directory only; historical tracked files unchanged')
with (o/'delivery_verification.json').open('x') as f:json.dump(v,f,indent=2)
with (o/'checkpoint_inventory.csv').open('x') as f:
 w=csv.DictWriter(f,fieldnames=['arm','path','sha256','global_step','role']);w.writeheader()
 for r in incident['checkpoints']:w.writerow(dict(arm=r['arm'],path=r['checkpoint'],sha256=r['sha256'],global_step=300,role='recovery_only_not_final_evaluation'))
dest=Path(__file__).parent/'delivery'/o.name;dest.mkdir(parents=True,exist_ok=False)
names=['report.md','run_plan.json','configuration_diff.json','evaluation_protocol.json','initialization_acceptance.json','target_identity_audit.json','cuda_incident.json','checkpoint_inventory.csv','delivery_verification.json']
manifest=[]
for name in names:
 s=(o/name).read_text().replace('/data/lyc/projects/rla-wm','${PROJECT_ROOT}').replace('/data/lyc/projects/LIBERO-plus','${LIBERO_ROOT}').replace('/data/lyc','${DATA_ROOT}')
 assert not re.search(r'(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)',s)
 with (dest/name).open('x') as f:f.write(s)
 manifest.append(dict(name=name,sha256=sha(dest/name),bytes=(dest/name).stat().st_size))
with (dest/'publication_manifest.json').open('x') as f:json.dump(dict(status='BLOCKED_PARTIAL_NOT_COMPLETED_EXPERIMENT',files=manifest,excluded=['weights','sidecars','images','videos','raw trajectories','credentials']),f,indent=2)
print(dest)
