"""Wait for bounded CPU preparation, commit protocol, then score exactly once."""
import argparse,os,json,sys,time,subprocess,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from frozen1800_state_readiness.generated_states import read,save,sha

def main(root,generator_pid):
 state=root/'route_b_v3/state_protocol.json';start=time.time()
 try:
  while not state.exists():
   os.kill(generator_pid,0)
   assert time.time()-start<1800,'CPU generator readiness timeout; no scoring'
   time.sleep(5)
  manifest=read(state);assert manifest['status']=='READY_FOR_PROTOCOL_COMMIT' and 20<=manifest['N']<=50
  from frozen1800_state_readiness.phase_aware_dedup import main as dedup
  dedup(root)
  from frozen1800_state_readiness.finalize_protocol import main as finalize
  finalize(root)
  prior=list((root/'generated_states').glob('seed_*'))+list((root/'route_b_v2/generated_states').glob('seed_*'))
  valid=list((root/'route_b_v3/generated_states').glob('seed_*'))
  save(root/'pre_scoring_acceptance.json',dict(status='PASS_TECHNICAL_NOT_POLICY',N=manifest['N'],
   candidate_attempts_all_revisions=len(prior)+len(valid),candidate_cap=500,
   v1_candidate_attempts=len(list((root/'generated_states').glob('seed_*'))),v2_candidate_attempts=len(list((root/'route_b_v2/generated_states').glob('seed_*'))),v3_candidate_attempts=len(valid),
   prepared_states_role='按冻结初始化分布新生成的仿真状态验证',
   policy_forwards_before_freeze=0,policy_rollouts_before_freeze=0,backward=0,optimizer_updates=0,AE_updates=0,
   source_audit_reads_sealed_contents=True,old_candidate_states_scored=0,
   geometry_depth_segmentation_and_state_checks=True,phase_aware_layout_dedup=True,
   generation_physics_steps_separate_from_policy=True,
   source_adapter_exact_replacement_check='PASS',paired_statistics_synthetic_algebra_check='PASS; not experiment results'))
  from frozen1800_state_readiness.publish import publish
  publish(root,'ready')
  from frozen1800_state_readiness.run_evaluation import main as evaluate
  evaluate(root)
 except Exception:
  save(root/'orchestrator_stopped.json',dict(time=time.time(),error=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--generator-pid',type=int,required=True);a=p.parse_args();main(a.root,a.generator_pid)
