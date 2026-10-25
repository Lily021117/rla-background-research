"""Publish allowlisted small GPU-pause evidence; exclude raw data and weights."""
import argparse,hashlib,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
dest=Path(__file__).parent/'delivery/extend600_gpu0_pause_20261025';dest.mkdir(parents=True,exist_ok=False)
names=['gpu0_driver_pause/report.md','gpu0_driver_pause/delivery_verification.json','gpu0_resume_authorization.json','cuda_incident.json','continuation_protocol.json','evaluation_protocol.json','attempt_registry_v3.json','policy_registry_v3.json']
inventory=[]
for name in names:
 text=(a.out/name).read_text()
 for old,new in [('/data/lyc/projects/rla-wm','${PROJECT_ROOT}'),('/data/lyc','${DATA_ROOT}')]:text=text.replace(old,new)
 text=re.sub(r'GPU-[0-9a-fA-F-]{36}','${GPU_UUID}',text)
 text=re.sub(r'PID[0-9]+','PID_REDACTED',text)
 assert not re.search(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN .*PRIVATE KEY',text)
 assert len(text.encode())<500_000 and '/root/' not in text
 target=dest/Path(name).name
 with target.open('x') as f:f.write(text)
 inventory.append(dict(path=target.name,sha256=hashlib.sha256(text.encode()).hexdigest()))
with (dest/'publication_manifest.json').open('x') as f:json.dump(dict(files=inventory,excluded=['weights','images','videos','trajectories','sidecars','credentials']),f,indent=2)
print(dest)
