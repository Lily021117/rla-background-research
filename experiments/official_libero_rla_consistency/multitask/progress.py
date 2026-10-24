"""Read-only concise progress across fixed task pipelines."""
import argparse,json,csv
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();out={}
for task in ('drawer','stove'):
 root=a.run/task;ae=root/('ae_run_recovery1' if task=='drawer' else 'ae_run');row={}
 for arm in ('C','D'):
  folder=ae/('AE_'+arm+('_recovery1' if task=='stove' else ''));path=folder/'training_metrics.csv';rs=list(csv.DictReader(path.open())) if path.exists() else [];row['AE_'+arm]=int(rs[-1]['step']) if rs else 0
 for arm in ('B','C','D'):
  path=root/'policy_run'/arm/'isolated_smoke/training_metrics.csv';rs=list(csv.DictReader(path.open())) if path.exists() else [];row['policy_'+arm]=int(rs[-1]['step']) if rs else 0
 trials=root/'policy_run/per_trial_results.json';results=json.loads(trials.read_text()) if trials.exists() else [];row['valid_rollouts']=sum(r['status']=='COMPLETE' for r in results);row['invalid_rollouts']=sum(r['status']!='COMPLETE' for r in results)
 out[task]=row
print(json.dumps(out))
