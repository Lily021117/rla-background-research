"""Resume only frozen diagnostics after token-padding assertion; no retraining."""
import os,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_shared_reference_target.experiment import HERE,REPO,read,save,analyze,publish
out=Path(sys.argv[1])
assert read(out/'seed_20261101/representation_process.json')['returncode']!=0
for seed in (20261101,20261102):
 assert read(out/f'seed_{seed}/C-shared-target/formal/training_acceptance.json')['successful_updates']==600
 assert len(read(out/f'seed_{seed}/per_trial_results.json'))==60
env=os.environ.copy();env['SHARED_DIAGNOSTIC_ATTEMPT']='matched_preprocessing'
subprocess.run([sys.executable,str(HERE/'launch.py'),'--out',str(out),'--seed','20261101','--stage','representation'],cwd=REPO,env=env,check=True)
subprocess.run([sys.executable,str(HERE/'launch.py'),'--out',str(out),'--seed','20261102','--stage','representation'],cwd=REPO,check=True)
save(out/'diagnostic_retry_audit.json',dict(original_failure='assertion that independently randomized padding token IDs must match',source='official RLDSBatchTransform random.choices padding',resolution='verify differences only in action-query replacement positions; match preprocessing Python RNG per pair and across compared frozen policies',original_failed_output_preserved=True,extra_optimizer_updates=0,extra_backward=0,extra_rollouts=0,changed_training_or_deployment=False))
analyze(out);publish(out)
