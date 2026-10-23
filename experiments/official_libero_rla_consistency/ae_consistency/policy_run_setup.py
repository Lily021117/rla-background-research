"""Freeze the already accepted policy experiment, without reading sealed data."""
import argparse,json,hashlib,csv,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
old=a.root/'run_20261023_08';a.output.mkdir(exist_ok=False)
def save(path,obj):path.write_text(json.dumps(obj,indent=2))
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
inputs=json.loads((old/'policy_readiness_retry1/readiness_inputs.json').read_text());common=old/'policy_readiness_retry1/isolated_smoke/common_policy_step0.pt'
protocol=json.loads((old/'policy_readiness_retry1/isolated_smoke/shared_training_protocol.json').read_text())
assert sha(inputs['B20_sampling_order'])==inputs['B20_sampling_sha256']
manifest=json.loads(Path(inputs['data_manifest']).read_text());assert sum(e['split']=='train' for e in manifest['episodes'])==20 and sum(e['split']=='validation' for e in manifest['episodes'])==5
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,memory.total','--format=csv'],text=True)
save(a.output/'run_plan.json',dict(frozen=True,previous=str(old),common_step0=str(common),common_step0_sha256=sha(common),inputs=inputs,shared_protocol=protocol,updates=600,checkpoints=[0,100,300,600],validation=[0,100,300,600],preflight='isolated full microbatch8 plus backward; no optimizer update; formal process reloads common RNG',memory_optimization_rule='only if necessary: activation checkpointing or matched microbatch reduction with same16 appearance effective batch; record numeric/RNG deviations',appearance_rules='same B20 order; each physical window clean then background; no extra augmentation',policy_selection='final600 only',closed_loop_trials=180,invalid_rule='task failure valid; infrastructure failure retained and experiment stops for diagnosis; no silent retry',sealed_read=False))
save(a.output/'initial_inventory.json',dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),GPU=gpu,sha256={str(x):sha(x) for x in [common,Path(inputs['data_manifest']),Path(inputs['action_proprio_statistics']),old/'final_checkpoint_inventory.csv',old/'sidecar_provenance.json',old/'development_protocol/evaluation_protocol.json']}))
for name in ('preflight','C','D','evaluation'):
 d=a.output/name;d.mkdir();save(d/'normalization_statistics_train_only.json',json.loads(Path(inputs['action_proprio_statistics']).read_text()))
save(a.output/'evaluation_protocol.json',dict(frozen=True,source=str(old/'development_protocol/evaluation_protocol.json'),conditions=['clean','seen_background','unseen_warm_teal'],models=['B20','C','D'],states=list(range(2026102300,2026102320)),order='state,condition,model',trials=180,max_steps=220,chunk_execution=8,reflection=1,MSAA=4,success='env.check_success()',initial_settling=0,invalid_rule='stop; retain all logs; no silent retry',role='development_not_independent_test'))
print(a.output,flush=True)
