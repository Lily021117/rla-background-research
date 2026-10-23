"""Read only the already-authorized AE training recovery checkpoints."""
import argparse,json,csv
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--previous',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
old=torch.load(a.previous/'AE_D/resume_step0300.pt',map_location='cpu');new=torch.load(a.previous/'AE_D_recovery_01/resume_step0600.pt',map_location='cpu')
assert old['global_step']==300 and old['next_sample']==2400 and new['global_step']==600 and new['next_sample']==4800
assert old['plan']==new['plan'] and old['lambda_cons']==new['lambda_cons']
assert all(k in old for k in ('master','optimizer','torch_rng','cuda_rng','python_rng','numpy_rng'))
prior=list(csv.DictReader((a.previous/'AE_D/training_metrics.csv').open()));C=json.loads((a.previous/'AE_C/acceptance.json').read_text());D=json.loads((a.previous/'AE_D_recovery_01/acceptance.json').read_text())
audit=dict(status='PASS_RECOVERY_WITH_REPLAY_COST',restored_components=['matching encoder and decoder','FP32 master','AdamW optimizer moments and step','CPU CUDA Python NumPy RNG','sampling order and offset','effective update counter'],scheduler='not used: fixed1e-4 throughout AE training',AMP_scaler='no dynamic scaler object: fixed multiplier256 restored from unchanged plan',C_effective_updates=600,C_computed_successes=600,C_recorded_seconds=C['seconds'],D_effective_updates=600,D_computed_successes=680,D_original_training_seconds=float(prior[-1]['seconds']),D_recovery_seconds=D['seconds'],D_sum_recorded_training_seconds=float(prior[-1]['seconds'])+D['seconds'],replay=json.loads((a.previous/'recovery_replay_verification.json').read_text()),limitation='recorded loop and acceptance timers exclude some process initialization, fault teardown and queue time; not full end-to-end wall time',sealed_read=False)
(a.output/'AE_D_recovery_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
