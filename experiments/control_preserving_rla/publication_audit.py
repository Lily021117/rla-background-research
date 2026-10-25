"""Inspect staged files and all newly introduced history before private push."""
import argparse,ast,hashlib,json,re,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
base='a76fb794e96d12f6ec9c4c8faa44f07b1222f549';prefix='experiments/control_preserving_rla/'
names=subprocess.check_output(['git','diff','--cached','--name-only','-z']).decode().split('\0');names=[n for n in names if n]
secret=re.compile(r'(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
inventory=[]
for name in names:
 assert name.startswith(prefix) and Path(name).suffix in ('.py','.md','.json','.csv'),name
 content=subprocess.check_output(['git','show',':'+name]);assert len(content)<2_000_000,name;text=content.decode()
 assert not secret.search(text),name
 if name.endswith('.py'):ast.parse(text,filename=name)
 if '/delivery/' in name:
  assert '/data/lyc/' not in text and '/root/' not in text,name
  assert not re.search(r'GPU-[0-9a-fA-F-]{36}',text),name
 inventory.append(dict(path=name,bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
history=[]
for line in subprocess.check_output(['git','rev-list','--objects',base+'..HEAD'],text=True).splitlines():
 fields=line.split(' ',1);oid=fields[0]
 if subprocess.check_output(['git','cat-file','-t',oid],text=True).strip()!='blob':continue
 name=fields[1];assert name.startswith(prefix) and Path(name).suffix in ('.py','.md','.json','.csv'),name
 content=subprocess.check_output(['git','cat-file','blob',oid]);assert len(content)<2_000_000 and not secret.search(content.decode()),name
 history.append(dict(path=name,object=oid,bytes=len(content)))
assert sum(r['bytes'] for r in inventory)<5_000_000
with (a.out/'publication_safety_audit.json').open('x') as f:json.dump(dict(status='PASS',base_already_on_remote=base,new_history_blobs=history,staged_files=inventory,excluded=['checkpoints','teacher cache','source data','images','sidecars','videos','raw trajectories','credentials']),f,indent=2)
print('PUBLICATION_SAFETY_PASS',len(inventory),'staged files',len(history),'new history blobs')
