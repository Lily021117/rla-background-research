"""Read-only diagnostic helpers; all generated files are exclusive-create."""
import csv,hashlib,json
from pathlib import Path
import numpy as np

def read(p): return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False)
def csvsave(p,rows):
 with Path(p).open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def csvread(p):return list(csv.DictReader(Path(p).open()))
def digest_state(ck):
 h=hashlib.sha256()
 for module,params in sorted(ck['parameters'].items()):
  for name,p in sorted(params.items()):h.update((module+name).encode());h.update(p.contiguous().view(__import__('torch').uint8).numpy().tobytes())
 return h.hexdigest()
