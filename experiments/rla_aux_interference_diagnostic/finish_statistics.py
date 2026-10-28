"""Algebraic effect size, no new forward/backward or gradient arrays."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
out=Path(sys.argv[1]);s=pd.read_csv(out/'target_switch_per_batch.csv')
units=json.loads((out/'C/diagnostic_scope_corrected/unit_audits.json').read_text())
assert all(len({tuple(v['unused_parameters']) for v in u['objective_audits'].values()})==1 for u in units)
s['delta_aux_over_action']=np.sqrt(np.maximum(0,s.norm_ratio_C**2+s.norm_ratio_shared**2-2*s.norm_ratio_C*s.norm_ratio_shared*s.aux_switch_cosine))
s.to_csv(out/'gradient_effect_size.csv',index=False,mode='x')
rows=[]
for r in json.loads((out/'supplemental_provenance.json').read_text()):
 actual=hashlib.sha256(Path(r['path']).read_bytes()).hexdigest();assert actual==r['sha256'];rows.append(dict(path=r['path'],unchanged=True))
for r in json.loads((out/'executed_source_inventory.json').read_text()):
 p=Path(r['path'])
 if p.name=='worker.py':p=out/'worker_attempt1.py'
 assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'];rows.append(dict(path=str(p),unchanged=True))
with (out/'additional_provenance_verification.json').open('x') as f:json.dump(dict(status='PASS',checks=rows,three_objective_unused_parameter_sets_equal=True,scope_correction_source=str(Path(__file__).parent/'worker.py')),f,indent=2)
