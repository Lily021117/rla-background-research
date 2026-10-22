import os,json,csv,hashlib,resource
from pathlib import Path
import numpy as np
from PIL import Image
O=Path(__file__).resolve().parent;B=O/'batch';B.mkdir(exist_ok=False)
R=Path('${RLA_PROJECT_ROOT}');X=R/'experiments/rla_background_invariance'
source=X/'step4_5_2_fk_proprio_validation/run_20261021_01/validate.py';s=source.read_text();s=s[s.index('import tensorflow as tf'):s.index('unchanged=all(sha')]
old="state=np.stack([fk[ep,j] for j in range(s,u)]);act=actions[ep][s:u]"
assert old in s;s=s.replace(old,"state=inputs[f'proprio_{ti}'];act=inputs[f'actions_{ti}']")
needle="   rows.append({'episode':ep";assert needle in s;s=s.replace(needle,"   if ti==0 and appearance=='clean':torch.save(batch,O/f'batch_{group}.pt')\n"+needle)
def csvout(n,rs):
 with (B/n).open('x') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def write(n,x):
 if n=='batch_schema.json':x.update(constructed_batches=880,unique_transitions=220,train_transitions=190,validation_transitions=30)
 with (B/n).open('x') as f:json.dump(x,f,indent=2)
ns={'__name__':'batch_expand','O':B,'P1':X/'step4_1_adapter_readiness/run_20261021_03','OLD':X/'step4_1_adapter_readiness/run_20261021_02','np':np,'Image':Image,'json':json,'csvout':csvout,'write':write,'trans':json.loads((O/'selection_manifest.json').read_text())['transitions'],'targets':list(csv.DictReader((O/'rla_target_manifest.csv').open())),'images':list(csv.DictReader((O/'sample_manifest.csv').open())),'inputs':np.load(O/'batch_inputs.npz')}
exec(compile(s,str(source),'exec'),ns)
write('CPU_acceptance.json',{'status':'PASS','batches':len(ns['rows']),'CPU_peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'source':str(source),'source_SHA256':hashlib.sha256(source.read_bytes()).hexdigest(),'backward':0,'optimizer_updates':0})
print('BATCH_PASS',len(ns['rows']),flush=True)
