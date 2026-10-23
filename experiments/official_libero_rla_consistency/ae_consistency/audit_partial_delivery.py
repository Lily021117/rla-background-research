"""CPU-only audit after a CUDA failure; never substitutes partial D for final."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
def save(n,x):
    with (a.run/n).open('x') as f:json.dump(x,f,indent=2)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
inventory=[]
for arm in ('C','D'):
    root=a.run/('AE_'+arm);rows=list(csv.DictReader((root/'training_metrics.csv').open()));assert [int(x['step']) for x in rows]==list(range(1,len(rows)+1))
    for kind in ('encoder','decoder','resume'):
        for path in sorted(root.glob(kind+'_step*.pt')):inventory.append(dict(arm=arm,kind=kind,file=str(path),bytes=path.stat().st_size,sha256=sha(path),step=int(path.stem.split('step')[-1])))
    if arm=='D':d_updates=len(rows)
for kind in ('encoder','decoder'):
    x=torch.load(a.run/'AE_C'/f'{kind}_step0000.pt',map_location='cpu',weights_only=True);y=torch.load(a.run/'AE_D'/f'{kind}_step0000.pt',map_location='cpu',weights_only=True);assert x.keys()==y.keys() and all(torch.equal(x[k],y[k]) for k in x);del x,y
assert json.loads((a.run/'AE_C/sampling_order.json').read_text())==json.loads((a.run/'AE_D/sampling_order.json').read_text())
with (a.run/'checkpoint_inventory.csv').open('x') as f:w=csv.DictWriter(f,fieldnames=list(inventory[0]));w.writeheader();w.writerows(inventory)
protocol=json.loads((a.run/'development_protocol/evaluation_protocol.json').read_text());known=[]
for e in json.loads((a.run/'data_manifest.json').read_text())['episodes']:known.extend(np.load(Path(e['root'])/'trajectory_labels.npz')['states'])
known=np.stack(known);overlaps=[]
for x in protocol['new_development_states']:
    s=np.load(x['state_file']);matches=int(np.sum(np.max(abs(known[:,1:]-s[None,1:]),axis=1)<=1e-8));overlaps.append(dict(seed=x['seed'],allowed_source_state_matches_ignoring_time_at1e8=matches))
save('static_overlap_audit.json',dict(checks=overlaps,sealed_contents_read=False,independence_not_certified=True,policy_rollouts=0))
save('hardware_incident.json',dict(status='STOPPED_CUDA_FAILURE',GPU_UUID='GPU-37d3f7d4-309e-2bb8-2d51-292dbbcce419',error='CUDA unspecified launch failure during loss/backward after completed update380',completed_C_updates=600,completed_D_updates=d_updates,recoverable_D_step=300,unpersisted_successful_updates=d_updates-300,ECC_increment_observed=False,baseline_and_after='SRAM correctable2 / uncorrectable parity1 / SM1',GPU0_free_MiB_at_stop=25174,required_free_MiB=26624,automatic_GPU_failover=False,restart_performed=False))
save('delivery_verification.json',dict(status='BLOCKED_FINAL_AE_D_AND_GPU_RESOURCE',AE_C_successful_updates=600,AE_D_successful_updates=d_updates,AE_D_final600_exists=False,AE_initial_states_equal=True,sampling_orders_equal=True,DINO_cache_images=10224,new_static_development_states=20,static_conditions_per_state=3,final_representation_evaluation=False,sidecars_generated=False,policy_readiness_executed=False,policy_optimizer_updates=0,new_policy_rollouts=0,heldout_contents_read=False,automatic_loss_search=False))
