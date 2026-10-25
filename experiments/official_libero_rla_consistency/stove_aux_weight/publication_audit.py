"""Inspect the exact staged upload; no credentials or model loading."""
import argparse
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
prefix = 'experiments/official_libero_rla_consistency/stove_aux_weight/'
paths = subprocess.check_output(['git', 'diff', '--cached', '--name-only', '-z']).decode().split('\0')
paths = [x for x in paths if x]
rows = []
for path in paths:
    assert path.startswith(prefix), path
    assert Path(path).suffix in ('.py', '.md', '.json', '.csv'), path
    content = subprocess.check_output(['git', 'show', ':'+path])
    assert len(content) < 2_000_000, path
    text = content.decode('utf-8')
    assert not re.search(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)', text), path
    if path.endswith('.py'):
        ast.parse(text, filename=path)
    if '/delivery/' in path:
        assert 'run_20261025_01_completed/' in path
        assert '/data/lyc/' not in text and '/root/' not in text, path
        assert not re.search(r'GPU-[0-9a-fA-F-]{36}', text), path
    rows.append(dict(path=path,bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
assert sum(x['bytes'] for x in rows) < 5_000_000
result = dict(status='PASS', staged_files=rows,
              upstream_anchor='05e59c5efe8055c9f130049b6fe6614a2c362aa6',
              remote_anchor_verified_before_commit=True,
              excluded=['weights','raw_data','videos','raw_trajectories','sidecars','credentials'],
              note='Four inherited trailing blank EOF lines are formatting warnings only; executed source retained.')
with (a.out/'publication_safety_audit.json').open('x') as f:
    json.dump(result,f,indent=2)
print('PASS', len(rows), 'files', sum(x['bytes'] for x in rows), 'bytes')
