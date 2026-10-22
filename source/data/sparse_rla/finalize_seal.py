"""Seal finished logs/report; supersedes intermediate pre-print log hashes."""
import os,json,hashlib
from pathlib import Path
OUT=Path(os.environ['RUN_DIR']);WT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for v in iter(lambda:f.read(8*1024**2),b''):h.update(v)
 return h.hexdigest()
assert json.loads((OUT/'delivery_verification.json').read_text())['optimizer_updates']==0
files={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='final_delivery_sha256.json' and p.suffix not in ['.pt','.safetensors']}
code={str(p):sha(p) for p in (WT/'sparse_rla').glob('*') if p.is_file()}
result=dict(authoritative=True,supersedes='artifact_sha256.json before final report amendment/log print',files=files,
 code_files=code,weight_reference_integrity='protected_checkpoints_before.json + protection_verification.json; symlinked originals not rewritten')
p=OUT/'final_delivery_sha256.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n')
print('FINAL_DELIVERY_SEALED',len(files))
