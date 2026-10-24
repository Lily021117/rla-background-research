import argparse,subprocess
from common import *
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
plan=read(CENTER/'run_plan.json');save(a.out/'run_plan.json',plan)
selected=states()[:5];assert len(selected)==5
save(a.out/'diagnostic_protocol.json',dict(frozen=True,seed_order=[s['seed'] for s in selected],condition='clean',prefix_limit=10,continuation_limit=20,max_total_control_steps=220,lift_m=.03,lift_source='previous frozen phase criterion',switch='first action chunk boundary at/after first lift, empty queue',snapshot='MuJoCo integration state plus controller/robot/gripper/observable state and RNG; same-policy suffix versus historical record gate',restore_action_tolerance=1e-6,restore_state_tolerance=1e-8,stop_on_restore_failure=True,same_observation_demo_rule='each allowed episode: first lift frame, +8,+24 capped to last full chunk; both appearances; deduplicate',same_observation_rollout_rule='all120 existing trajectories: first lift chunk boundary,+16,+40 capped before end; same frozen input for both models',success='official On(bowl,plate): height AND contact AND xy<.03m',no_training=True,no_backward=True,no_sealed=True))
registry={'C':str(S/'policy_run/C/isolated_smoke/step0600.pt'),'D-center':str(CENTER/'D-center/isolated_smoke/step0600.pt')};save(a.out/'policy_registry.json',registry)
protected={str(p):sha(p) for p in [BDDL,CENTER/'report.md',CENTER/'checkpoint_inventory.csv',S/'policy_run/run_plan.json',S/'frozen_dataset_manifest.json',S/'ae_run/training_plan.json'] if p.is_file()}
save(a.out/'protected_inputs.json',protected);save(a.out/'source_version.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),policy_hashes={k:sha(Path(v)) for k,v in registry.items()}))
print(a.out,flush=True)
