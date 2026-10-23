"""Prepare fixed600 budgets and a common original policy initialization."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--task',type=Path,required=True);p.add_argument('--stage',choices=['ae','policy'],required=True);a=p.parse_args();base=a.project/'experiments/official_libero_rla_consistency';manifest=json.loads((a.task/'frozen_dataset_manifest.json').read_text());ae=a.task/'ae_run'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,x):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(x,f,indent=2)
if a.stage=='ae':
 plan=json.loads((base/'run_20261023_08/training_plan.json').read_text());windows=sum(e['valid_chunks'] for e in manifest['episodes'] if e['split']=='train');plan.update(train_window_equivalent_epochs=4800/windows,all_horizon_pool_epochs=4800/(8*windows),lambda_cons=.009127798272072366,lambda_selection='frozen from original task; no per-task calibration or development tuning',budget_reason='same600x8 physical transitions across all tasks; fixed compute, not convergence',task_id=manifest['task_id'])
 assert plan['updates']==600 and plan['lr']==1e-4 and plan['loss_scale']==256
 save(ae/'training_plan.json',plan)
else:
 assert (ae/'sidecar_alignment_audit.json').is_file();root=a.task/'policy_run';root.mkdir(exist_ok=False)
 init=base/'run_20261023_08/policy_readiness_retry1/isolated_smoke/common_policy_step0.pt';baseline=base/'run_20261023_06/B20/formal_B/step0000.pt';statistics=a.task/'normalization_statistics_expanded_train_only.json'
 windows=[(e['episode'],t) for e in manifest['episodes'] if e['split']=='train' for t in range(e['valid_chunks'])];rng=np.random.RandomState(20261023);order=[]
 while len(order)<4800:order.extend(windows[i] for i in rng.permutation(len(windows)))
 order=order[:4800];save(root/'sampling_order.json',order)
 shared=json.loads((base/'run_20261023_08/policy_readiness_retry1/isolated_smoke/shared_training_protocol.json').read_text());shared.update(status='FROZEN_NEW_TASK',sampling_order_reference='sampling_order.json',normalization='this task train-only action/proprio; common AE-C training-only target normalization',task_id=manifest['task_id'],RLA_auxiliary_baseline='B has original official action head, no auxiliary supervision',memory='same non-reentrant activation checkpointing for B/C/D; expandable allocator, cache release',window_equivalent_epochs=4800/len(windows))
 inputs=dict(project=str(a.project),B20_step0=str(baseline),B20_sampling_order=str(root/'sampling_order.json'),B20_sampling_sha256=sha(root/'sampling_order.json'),data_manifest=str(a.task/'frozen_dataset_manifest.json'),action_proprio_statistics=str(statistics),legacy_B20_sampling_field_means_new_task_common_order=True,no_trained_policy_initialization=True)
 plan=dict(frozen=True,previous=str(ae),common_step0=str(init),common_step0_sha256=sha(init),inputs=inputs,shared_protocol=shared,updates=600,checkpoints=[0,1,100,300,600],validation=[0,100,300,600],task_id=manifest['task_id'],appearance_rules='same physical order; clean/background eachonce; no unseen palette training',selection='final600 only',closed_loop_trials=180,sealed_read=False)
 save(root/'run_plan.json',plan)
 for name in ('preflight6','preflight_B','B','C','D','evaluation'):
  save(root/name/'normalization_statistics_train_only.json',json.loads(statistics.read_text()))
 save(root/'policy_registry.json',{arm:str(root/arm/'isolated_smoke/step0600.pt') for arm in ('B','C','D')})
 task=json.loads((a.task/'task_protocol.json').read_text());save(root/'evaluation_protocol.json',dict(frozen=True,source=str(ae/'development_protocol/evaluation_protocol.json'),conditions=['clean','seen_background','unseen_background'],models=['B','C','D'],states=task['development_seeds'],order='state,condition,model',trials=180,max_steps=220,chunk_execution=8,reflection=1,MSAA=4,bddl=task['bddl'],task=task['task'],role='new development, not independent final test',invalid_rule='stop, preserve logs; only identified infrastructure repairs may retry exact triplet, no task-failure retries'))
 print(root,flush=True)
