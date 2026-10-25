"""Read-only final optimizer/counter checks; no model construction or GPU."""
import argparse,json,gc
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
plan=json.loads((a.out/'run_plan.json').read_text());registry=json.loads((a.out/'policy_registry_resumed.json').read_text());rows=[]
for arm,path in registry.items():
 ck=torch.load(path,map_location='cpu');steps=[int(s['step']) for s in ck['optimizer']['state'].values()]
 assert ck['arm']==arm and ck['global_step']==600 and ck['next_physical_sample']==4800
 assert ck['scheduler']['last_epoch']==600 and set(steps)=={600}
 assert ck['AMP_scaler'] is None and ck['dtype']=='bfloat16' and ck['activation_checkpointing']
 assert ck['configuration']==plan
 rows.append(dict(model=arm,path=path,global_step=600,parent_C_updates=600,next_physical_sample=4800,optimizer_state_entries=len(steps),optimizer_step_min=min(steps),optimizer_step_max=max(steps),scheduler_last_epoch=600,AMP_scaler=None,dtype='bfloat16',activation_checkpointing=True,configuration_exact=True,model_scope='C initialized post-training student'))
 del ck;gc.collect()
with (a.out/'final_checkpoint_state_audit.json').open('x') as f:json.dump(dict(status='PASS',GPU_allocations=0,optimizer_updates=0,models=rows),f,indent=2)
print('FINAL_CHECKPOINT_STATE_PASS',len(rows))
