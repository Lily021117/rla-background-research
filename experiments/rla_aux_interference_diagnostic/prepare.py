"""Freeze metadata-only selection before any new model computation."""
import csv, hashlib, json, sys
from pathlib import Path
import torch

ROOT=Path('/data/lyc/projects/rla-wm/experiments')
OUT=Path(sys.argv[1])
def read(p): return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
 return h.hexdigest()
def save(n,x):
 with (OUT/n).open('x') as f: json.dump(x,f,indent=2)
def write(n,rows):
 with (OUT/n).open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
inventory=[];snapshots=[];batches=[];protected={};train=[];valid=[];step0_hash=None
for order in (20261101,20261102):
 bases={m:ROOT/r/f'seed_{order}'/arm/'formal' for m,r,arm in [('C','rla_sampling_stability/run_20261027_01','C'),('shared','rla_shared_reference_target/run_20261027_01','C-shared-target')]}
 traces={m:[json.loads(x) for x in (b/'actual_sample_ids.jsonl').read_text().splitlines()] for m,b in bases.items()}
 assert traces['C']==traces['shared'] and len(traces['C'])==600
 plan=read(bases['C'].parent.parent/'run_plan.json')
 if order==20261101:
  save('run_plan.json',plan)
  norm=read(bases['C'].parent.parent/'normalization.json');save('normalization.json',norm)
  dataset=read(Path(plan['inputs']['data_manifest']));meta={r['episode']:r for r in dataset['episodes']}
  for key in ('data_manifest','action_proprio_statistics'):
   p=Path(plan['inputs'][key]);protected[str(p)]=sha(p)
  for p in sorted((Path(plan['previous'])/'sidecars/C').rglob('*')):
   if p.is_file():protected[str(p)]=sha(p)
 # Deterministic positions span first, middle, late and final update; no outcomes read.
 for step in (1,200,400,600):
  ids=traces['C'][step-1]['actual_sample_ids'];assert len(ids)==16
  for i in range(0,16,2):
   e,t,c=ids[i];assert ids[i+1]==[e,t,'background'] and c=='clean'
   assert meta[e]['split']=='train' and 0<=t<meta[e]['valid_chunks']
  batches.append(dict(id=f'{order}_{step}',order=order,historical_update=step,samples=ids,sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),microbatch=8,accumulation=2))
 for method,base in bases.items():
  for name in ('execution_configuration.json','initialization_audit.json','training_acceptance.json','actual_sample_ids.jsonl','training_metrics.csv','validation_metrics.csv'):
   p=base/name;protected[str(p)]=sha(p)
  for filename,dest in [('training_metrics.csv',train),('validation_metrics.csv',valid)]:
   with (base/filename).open() as f:
    for r in csv.DictReader(f):
     r=dict(order=order,method=method,**r)
     if 'timestep' in r:r['time_third']=min(2,3*int(r['timestep'])//meta[r['episode']]['valid_chunks'])
     dest.append(r)
  for step in (0,1,100,300,600):
   p=base/f'step{step:04}.pt';row=dict(order=order,method=method,step=step,path=str(p),exists=p.exists())
   if p.exists():
    row['sha256']=sha(p);protected[str(p)]=row['sha256'];ck=torch.load(p,map_location='cpu',weights_only=False)
    assert ck['global_step']==step
    h=hashlib.sha256()
    for k,ps in sorted(ck['parameters'].items()):
     for n,v in sorted(ps.items()):h.update((k+n).encode());h.update(v.contiguous().view(torch.uint8).numpy().tobytes())
    row['parameter_sha256']=h.hexdigest();row['next_physical_sample']=ck.get('next_physical_sample');del ck
    if step==0:
     if step0_hash is None:step0_hash=row['parameter_sha256'];snapshots.append(dict(id='common_step0',**row))
     else:assert step0_hash==row['parameter_sha256']
    elif step in (100,300,600):snapshots.append(dict(id=f'{method}_{order}_{step}',**row))
   inventory.append(row)
assert len(snapshots)<=13 and len(batches)==8
save('checkpoint_inventory.json',inventory);save('snapshot_manifest.json',snapshots);save('batch_manifest.json',batches);save('protected_inputs.json',protected)
save('diagnostic_protocol.json',dict(base_commit='b404cde06d6f3f66524e685a68113b089dc9ecd9',snapshot_limit=13,batches=8,units=len(snapshots)*8,objectives_per_unit=3,acceptance_units_max=2,acceptance_extra_gradient_extractions_max=6,selection='actual successful update IDs 1,200,400,600 for each frozen order; no loss/outcome selection',randomness='restore common step0 Python/NumPy/CPU/CUDA RNG before each objective; preserve checkpoint recomputation RNG; independent of snapshot training RNG',mode='train',precision='bfloat16',backend='Math SDPA',microbatch=8,accumulation=2,lambda_aux=plan['shared_protocol']['lambda_aux'],loss='actual official wrapper action loss and nn.L1Loss auxiliary captured from graph; each micro objective divided by2',near_zero_norm=1e-10,action_equivalence='bitwise expected under Math SDPA and restored complete RNG',decomposition_relative_L2_tolerance=.02,decomposition_reason='BF16 shared backward summation can differ from separately accumulated branches; conservative predeclared2% gate, report actual',stage_labels='episode valid-window time thirds, NOT physical stages',optimizer_updates=0,AE_updates=0,new_rollouts=0,sealed_reads=0,stop_on_gate_failure=True))
write('historical_training_curves.csv',train);write('historical_validation_samples.csv',valid)
# Match exact source sample keys, never compare unpaired row means.
keys=['order','step','episode','timestep','appearance'];paired={}
for r in valid:paired.setdefault(tuple(r[k] for k in keys),{})[r['method']]=r
diff=[]
for key,v in paired.items():
 if set(v)!={'C','shared'}:continue
 r=dict(zip(keys,key));r['time_third']=v['C']['time_third']
 for m in ('action_L1','translation_L1','rotation_L1','gripper_L1','auxiliary_L1'):
  r[m+'_C']=float(v['C'][m]);r[m+'_shared']=float(v['shared'][m]);r[m+'_difference']=float(v['shared'][m])-float(v['C'][m])
 diff.append(r)
write('paired_validation_samples.csv',diff)
save('preparation_acceptance.json',dict(status='PASS',snapshots=len(snapshots),batches=len(batches),common_step0_parameters_equal=True,actual_training_traces_equal=True,paired_validation_rows=len(diff),missing=[r for r in inventory if not r['exists']],selection_frozen_before_new_computation=True))
print('PREPARED',len(snapshots),len(batches),flush=True)
