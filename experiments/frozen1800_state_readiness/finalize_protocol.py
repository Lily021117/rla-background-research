"""Freeze new-state evaluation before any policy scoring."""
import argparse, json, hashlib, time
from pathlib import Path
from .generated_states import P,SOURCE,sha,read,save

def main(root):
 state_path=root/'route_b_v3/state_protocol.json';states=read(state_path)
 assert states['status']=='READY_FOR_PROTOCOL_COMMIT' and 20<=states['N']<=50
 assert read(root/'phase_aware_dedup.json')['pass_all']
 out=root/'evaluation';out.mkdir(exist_ok=False)
 protocol=read(SOURCE/'evaluation/evaluation_protocol.json')
 protocol.update(frozen_at_unix=time.time(),states=[s['seed'] for s in states['new_development_states']],trials=30*states['N'],
  frozen_state_protocol=str(state_path),role='按冻结初始化分布新生成的仿真状态验证; backgrounds previously used in development',
  generated_independent_states=True,low_lr_continuation=False,
  checkpoint_selection='six frozen step1800 policies only',
  statistical_plan='20000 paired physical-state bootstrap draws seed31001; all models,policy seeds and appearances jointly resampled; evaluation-data uncertainty only',
  decision_rule='Primary equal-four-background/equal-three-policy-seed D-low minus C-low; separately report Clean cost and per-seed/per-background; no adaptation',
  retry_rule='On infrastructure/hardware error preserve attempt and stop. No automated device switch or retry. A documented repair may resume identical triplet; never retry valid failure.')
 save(out/'evaluation_protocol.json',protocol)
 registry=read(SOURCE/'evaluation/policy_registry.json');assert len(registry)==6
 old=read(root.parent/'run_20261031_01/code/reports/frozen1800_independent_states_readiness/checkpoint_inventory.json')
 inventory=[]
 for name,path in registry.items():
  matching=next(x for x in old if str(P/x['path'])==path);actual=sha(path);assert actual==matching['actual_sha256']
  inventory.append(dict(model=name,path=path,sha256=actual,step=1800))
 save(out/'policy_registry.json',registry);save(out/'checkpoint_inventory_frozen.json',inventory)
 save(out/'run_plan.json',read(SOURCE/'evaluation/run_plan.json'))
 save(root/'frozen_inputs.json',dict(evaluation_protocol_sha256=sha(out/'evaluation_protocol.json'),state_protocol_sha256=sha(state_path),
  generation_protocol_sha256=sha(root/'route_b_v3/generation_protocol.json'),source_audit_sha256=sha(root/'source_content_audit.json'),
  candidate_results_sha256=sha(root/'route_b_v3/candidate_results.json'),checkpoint_inventory_sha256=sha(out/'checkpoint_inventory_frozen.json')))
 print('FINAL_PROTOCOL_READY',states['N'],protocol['trials'],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
