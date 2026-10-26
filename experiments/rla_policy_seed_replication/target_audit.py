"""Reuse frozen sidecars; validate both methods against existing normalization."""
import argparse,json,sys,os,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();protocol=json.loads((a.out/'replication_protocol.json').read_text());root=a.out/f"seed_{protocol['seeds'][0]}";plan=json.loads((root/'planned_run_plan.json').read_text());R=Path(plan['inputs']['project']);A=Path(plan['previous'])
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'official_libero_rla_consistency'),str(R/'experiments/official_libero_rla_consistency/run_20261023_02/vla-official'),str(R/'worktrees/rla-official-0fc6e97/rebuttal/exp2_vla_adapter/vla_adapter_overlay')]
os.environ['CENTER_RUN']=str(root)
from stove_centering.store import CenterStore
es=json.loads(Path(plan['inputs']['data_manifest']).read_text())['episodes'];prov=json.loads((A/'sidecar_provenance.json').read_text());normal=json.loads((root/'normalization.json').read_text());checks=[]
for arm,m,mean in [('C','C','mu_C'),('D-center','D','mu_D')]:
 os.environ['CENTER_ARM']=arm
 for c in ['clean','background']:
  store=CenterStore(str(A/'sidecars'/m/c),chunk=8,num_tokens=16,token_dim=64,norm='center');np.testing.assert_array_equal(store.offset,np.asarray(normal[mean],np.float32));assert store.scale==np.float32(normal['s_C'])
  for e in es:
   record=next(x for x in prov if x['episode']==e['episode'] and x['appearance']==c and x['model']==m);labels=np.load(Path(e['root'])/'trajectory_labels.npz');slot=store.slot(labels['proprio']);assert slot>=0;z=np.load(record['file'],mmap_mode='r');mask=np.load(record['mask_file']);np.testing.assert_array_equal(mask,labels['valid_horizon_mask']);assert mask[:e['valid_chunks']].all()
   for t in [0,e['valid_chunks']//2,e['valid_chunks']-1]:
    actual=store.frame(slot,t);expected=((z[t].astype(np.float32)-np.asarray(normal[mean],np.float32))/np.float32(normal['s_C'])).reshape(8,1024);np.testing.assert_array_equal(actual,expected);assert np.isfinite(actual).all();checks.append(dict(method=arm,episode=e['episode'],appearance=c,t=t,shape=list(actual.shape),encoder_sha256=record['encoder_sha256'],sidecar_sha256=record['sha256'],exact_reader=True))
with (a.out/'frozen_target_audit.json').open('x') as f:json.dump(dict(status='PASS',checks=checks,AE_updates=0,new_sidecars=0,own_C_D_centers=True,shared_historical_scale=True,normalization_source=protocol['normalization_source'],future_targets_not_policy_inputs=True),f,indent=2)
print('FROZEN_TARGET_PASS',len(checks))
