"""CPU-only preservation audit after GPU0 driver query stalled; no model forward."""
import argparse,csv,hashlib,json,math,subprocess
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();o=a.out
dest=o/'gpu0_driver_pause';dest.mkdir(exist_ok=False)
folder=o/'D-center_m0p1/formal_gpu0_retry1'
rows=list(csv.DictReader((folder/'training_metrics.csv').open()))
assert [int(r['step']) for r in rows]==list(range(201,397))
assert all(math.isfinite(float(r[k])) for r in rows for k in ['action_loss','auxiliary_loss','total_loss','gradient_vla','gradient_head'])
path=folder/'step0300.pt';ck=torch.load(path,map_location='cpu')
assert ck['global_step']==300 and ck['next_physical_sample']==2400
assert ck['scheduler']['last_epoch']==300 and ck['scheduler']['milestones']=={1200:1}
assert ck['AMP_scaler'] is None and ck['optimizer']['param_groups'][0]['lr']==2e-5
assert all(int(s['step'])==300 for s in ck['optimizer']['state'].values())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
protected=json.loads((o/'protected_inputs.json').read_text())
checks=[dict(path=r['path'],unchanged=sha(r['path'])==r['sha256']) for r in protected]
assert all(r['unchanged'] for r in checks)
result=dict(status='PARTIAL_BLOCKED_GPU_DRIVER',C_final_step=600,D_last_logged_success=396,D_latest_full_recovery_step=300,
 D_successful_computations_this_attempt=len(rows),D_unpersisted_successful_computations=96,
 C_continuation_successful_computations=200,total_continuation_successful_computations=396,
 known_C_replayed_computations=16,new_distinct_logged_update_indices=380,
 durable_extension_beyond_user_reported_C416_D200=284,
 failed_inflight_D397='not confirmed; not counted',valid_rollouts=0,actual_rollout_attempts=0,
 AE_updates=0,teacher_updates=0,additional_diagnostic_backward=0,sealed_reads=0,
 checkpoint=dict(path=str(path),sha256=sha(path),global_step=300,keys=sorted(ck),optimizer_states=len(ck['optimizer']['state']),
 full_recovery_cpu_audit=True,GPU_strict_reload_after_incident=False),
 incident=dict(observation='Training stopped advancing at396; nvidia-smi GPU0 query stuck in D state, os_acquire_rwlock_write; dmesg permission denied',
 cause='UNRESOLVED: evidence of driver query stall; no proof of OOM or new ECC',
 stop='SIGTERM sent only to own trainer PID2020445 and own query PID2021520; no GPU reset, no other task termination',
 last_observed_training_process='Zl; launcher awaiting process teardown',last_observed_query_process='D state',
 GPU_memory_release='UNVERIFIED because device query is blocked',
 checkpoint_397_or_396_saved=False),protected_inputs=checks,
 source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
(dest/'delivery_verification.json').write_text(json.dumps(result,indent=2))
report='''# GPU0 continuation pause — partial result

D-center resumed from full step200 with unchanged training semantics and successfully computed updates201–396. C remains complete at600. D's latest complete checkpoint is300, verified on CPU including optimizer/scheduler/sampling position; 301–396 are logged but not persisted and must be counted as replay if resumed. No substitute weight-only restart is permitted.

Training stopped advancing while a GPU0 status query blocked in the driver (`os_acquire_rwlock_write`). Only this task's trainer and status query received SIGTERM. The cause is unresolved: no evidence establishes OOM or a new ECC error. Device-memory release could not be verified while the query was blocked. No other process was terminated and no device reset was performed.

This attempt:196 successful training computations,0 rollout attempts. Across the continuation:C200+D196=396 successful computations, including16 known C replays;380 previously unfinished logged update indices. D has96 additional unpersisted computations. An in-flight397 is unconfirmed and excluded. AE updates0, teacher updates0, extra diagnostic backward0, sealed-test reads0. Training backward was performed as authorized.

The requested120 valid development rollouts have NOT run. No method conclusion, final600 D comparison, or success-rate claim is available. Resume only after GPU safety is re-established, from full D300 state with original1200-step scheduler and original sampling/RNG, targeting600. Do not repeat C or silently disregard the96 replays.

Historical protected inputs were rehashed unchanged; see delivery_verification.json. Failed attempts and all checkpoint/log evidence are retained.
'''
(dest/'report.md').write_text(report)
print(json.dumps({k:result[k] for k in ['status','C_final_step','D_last_logged_success','D_latest_full_recovery_step','D_successful_computations_this_attempt','valid_rollouts']}))
