"""CPU-only recovery inventory; no retry or GPU use."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,json,csv,hashlib,subprocess
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();rows=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
for arm in ('C-low','D-center-low'):
 d=a.out/arm/'isolated_smoke';ck=torch.load(d/'step0300.pt',map_location='cpu',weights_only=False)
 assert ck['global_step']==300 and ck['next_physical_sample']==2400
 assert all(torch.isfinite(t).all() for module in ck['parameters'].values() for t in module.values())
 assert ck['optimizer']['state'] and ck['scheduler']['last_epoch']==300
 assert all(k in ck for k in ('torch_rng','cuda_rng','python_rng','numpy_rng','AMP_scaler','configuration'))
 optsteps=sorted({int(v['step']) for v in ck['optimizer']['state'].values()});assert optsteps==[300]
 assert ck['configuration']==json.loads((a.out/'run_plan.json').read_text())
 logs=list(csv.DictReader((d/'training_metrics.csv').open()));assert len(logs)==300 and int(logs[-1]['successful_updates'])==300
 rows.append(dict(arm=arm,checkpoint=str(d/'step0300.pt'),sha256=sha(d/'step0300.pt'),successful_updates=300,actual_computed_updates=300,optimizer_steps=optsteps,next_physical_sample=2400,full_RNG=True,AMP_scaler=None,parameters_finite=True,scheduler_last_epoch=300,training_replay_needed=0,validation300_incomplete=True))
 del ck
gpu='GPU-37d3f7d4-309e-2bb8-2d51-292dbbcce419';ecc=subprocess.check_output(['nvidia-smi','-i',gpu,'-q','-d','ECC'],text=True)
with (a.out/'ecc_after_incident.txt').open('x') as f:f.write(ecc)
def counters(s):return [x.strip() for x in s.splitlines() if any(k in x for k in ('Correctable','Uncorrectable','SRAM SM','Threshold Exceeded'))]
assert counters(ecc)==counters((a.out/'C-low/isolated_smoke/ecc_before.txt').read_text())
with (a.out/'cuda_incident.json').open('x') as f:json.dump(dict(status='BLOCKED_PENDING_RESOURCE_DIRECTION',error='CUBLAS_STATUS_EXECUTION_FAILED in both processes during fixed validation at step300',GPU=gpu,ECC_increment=False,root_cause='UNRESOLVED; do not infer faulty hardware solely from CUDA exception',checkpoints=rows,new_closed_loop=0,all_own_GPU_processes_exited=True,no_automatic_retry=True),f,indent=2)
print(json.dumps(rows,indent=2))
