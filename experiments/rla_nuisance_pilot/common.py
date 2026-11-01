import os,json,hashlib,csv,xml.etree.ElementTree as ET
from pathlib import Path
P=Path('/data/lyc/projects/rla-wm')
HERE=Path(__file__).resolve().parent;CODE=HERE.parents[1]
ROOT=Path(os.environ.get('NUISANCE_ROOT',str(CODE.parent)))
OLD=P/'experiments/rla_frozen1800_independent_states/run_20261031_02'
BASE=P/'experiments/rla_scratch_policy_aux/run_20261029_01'
LIB=Path('/data/lyc/projects/LIBERO-plus')
GPU='GPU-205a3f5a-d91b-6385-c588-4e4cf3505f7e'
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def rank(s):return hashlib.sha256(('nuisance_pilot_v1:'+str(s)).encode()).hexdigest()
def table(p,rows):
 with Path(p).open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def xml_for(text,condition):
 tree=ET.fromstring(text);changes=[]
 if condition['kind']=='background':
  for name,path in condition['textures'].items():
   node=next(x for x in tree.iter('texture') if x.get('name')==name);changes.append((node,dict(node.attrib)));node.set('file',path)
 else:
  for attrs in condition['lights']:
   node=next(x for x in tree.iter('light') if x.get('name')==attrs['name']);changes.append((node,dict(node.attrib)));node.attrib.clear();node.attrib.update(attrs)
 result=ET.tostring(tree,encoding='unicode')
 for node,attrs in changes:node.attrib.clear();node.attrib.update(attrs)
 assert ET.tostring(tree)==ET.tostring(ET.fromstring(text))
 return result
