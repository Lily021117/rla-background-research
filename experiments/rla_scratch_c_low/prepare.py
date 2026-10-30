"""Freeze C-low plan and reuse identities without overwriting the completed run."""
import sys,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_scratch_c_low.common import *
def main():
 assert read(BASE/'completion.json')['policy_updates']==4800
 assert read(BASE/'delivery/delivery_verification.json')['actual_attempts']==800
 ROOT.mkdir(parents=True,exist_ok=False);protected={}
 for name in ['protocol.json','encoder_inventory.json','target_manifest.json','target_acceptance.json','targets/C/statistics.json','targets/D/statistics.json','delivery/per_trial_results.json','delivery/paired_statistics.json','delivery/delivery_verification.json','delivery/report.md','github_sync_final.json','evaluation_clean/evaluation_protocol.json','evaluation_background/evaluation_protocol.json']:
  p=BASE/name;protected[str(p)]=sha(p)
 for r in read(BASE/'target_manifest.json'):
  assert sha(r['path'])==r['sha256'];protected[r['path']]=r['sha256']
 for order in ORDERS:
  old=BASE/f'seed_{order}';root=ROOT/f'seed_{order}';plan=read(old/'run_plan.json')
  plan.update(arms=['C-low'],experiment='scratch_AE_C_low_completion',target_source_by_arm={'C-low':'C'},closed_loop_trials=100)
  plan['shared_protocol']['lambda_aux']=LAMBDA/2
  plan['extension_only_changes']=['new arm C-low reads existing new AE-C targets with existing own normalization','constant lambda0/2; no other execution setting changed']
  save(root/'run_plan.json',plan)
  gate=old/'preflight/formal/preflight_acceptance.json';evidence=read(gate);assert evidence['status']=='PASS'
  evidence.update(reused_evidence=True,source=str(gate),source_sha256=sha(gate),reason='same official graph, complete batch and frozen C/D targets; half coefficient already executed in D-low; runtime verifies new C-low initial identity')
  save(root/'preflight/formal/preflight_acceptance.json',evidence)
  for p in [old/'run_plan.json',gate,Path(plan['common_step0']),Path(plan['inputs']['B20_sampling_order']),Path(plan['inputs']['action_proprio_statistics']),Path(plan['inputs']['data_manifest'])]:protected[str(p)]=sha(p)
  for arm in ['C-fixed','D-low']:
   for name in ['step0600.pt','initialization_audit.json','actual_sample_ids.jsonl','execution_configuration.json','training_acceptance.json','training_metrics.csv','validation_metrics.csv']:
    p=old/arm/'formal'/name;protected[str(p)]=sha(p)
  source=read(old/'C-fixed/formal/initialization_audit.json')
  save(root/'expected_initialization.json',source)
  assert sha(plan['common_step0'])==plan['common_step0_sha256']
 ep=read(BASE/'evaluation_clean/evaluation_protocol.json');bg=read(BASE/'evaluation_background/evaluation_protocol.json')
 for k in ['states','max_steps','control_freq','chunk_execution','reflection','MSAA','bddl','task','frozen_state_protocol','success']:assert ep[k]==bg[k]
 registry={f'C-low_order{o}_s600':str(ROOT/f'seed_{o}/C-low/formal/step0600.pt') for o in ORDERS}
 ep.update(models=list(registry),conditions=['clean']+bg['conditions'],trials=200,C_low_completion=True,gate='explicit authorization: all Clean plus four backgrounds, no new gate',order='state,appearance,order',checkpoint_selection='both final600 only')
 save(ROOT/'evaluation/evaluation_protocol.json',ep);save(ROOT/'evaluation/policy_registry.json',registry);save(ROOT/'evaluation/run_plan.json',read(ROOT/f'seed_{ORDERS[0]}/run_plan.json'))
 frozen=read(ep['frozen_state_protocol']);protected[ep['frozen_state_protocol']]=sha(ep['frozen_state_protocol'])
 for s in frozen['new_development_states']:
  protected[s['state_file']]=sha(s['state_file'])
  for c in s['conditions']:
   p=Path(s['state_file']).parent/(c['condition']+'.xml');assert sha(p)==c['XML_sha256'];protected[str(p)]=sha(p)
 # Freeze all previous run code files, not only the final weights.
 for p in (HERE.parent/'rla_scratch_policy_aux').glob('*.py'):protected[str(p)]=sha(p)
 save(ROOT/'protected_inputs.json',protected)
 save(ROOT/'protocol.json',dict(base_commit='9f5b6fdb8e69dac379868b58ded894c3ec61dd2d',orders=ORDERS,arms=['C-low'],successful_updates_each=600,total_new_updates=1200,new_valid_rollouts=200,AE_updates=0,target='existing new AE-C formal2000 labels',normalization='existing AE-C own train per-position mean + raw global scale; no reestimation',lambda_RLA=LAMBDA/2,initialization='exact corresponding previous C-fixed step0 and RNG; no new auxiliary initialization',sampling='read original frozen full index, compare actual IDs against previous run',checkpoints=[0,1,100,300,600],validation=[0,100,300,600],diagnostics=[0,300,600],primary_comparisons=['C-low minus C-fixed','D-low minus C-low'],reference_results='all800 prior valid trials frozen/read-only; select matching C-fixed and D-low results without rerun',statistical_unit='physical state, all5 appearances and both orders resampled jointly; intervals exclude complete training seed/task uncertainty',GPU=GPU,resources='26GiB startup gate, UUID and ECC checks, no automatic retry/device switching',stop='only2 trainings and200 rollouts; no AE updates, no hyperparameter search or extension',sealed_read=False))
 table(ROOT/'auxiliary_weight_schedule.csv',[dict(order=o,k=k,successful_update=k+1,lambda_RLA=weight('C-low',k)) for o in ORDERS for k in range(600)])
 print('C_LOW_PREPARATION_PASS',flush=True)
if __name__=='__main__':main()
