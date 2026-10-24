"""Record exact source/statistics and early reproduction without GPU use."""
import argparse,sys,json,subprocess
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stove_diagnosis'))
from common import read,save,sha,digest_state
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.out;plan=read(r/'run_plan.json')
save(r/'implementation_hashes.json',{str(p):sha(p) for p in Path(__file__).parent.glob('*.py')})
save(r/'statistics_hashes.json',{str(p):sha(p) for p in [r/'normalization.json',r/'mean_training_sources.csv',Path(plan['previous'])/'sidecars/D/clean/stats.json',Path(plan['previous'])/'sidecars/D/clean/raw_target_statistics.json']})
for arm in plan['arms']:save(r/f'training_config_{arm}.json',dict(**plan,active_arm=arm,center='mu_C' if arm=='D-old' else 'mu_D',scale='s_C',raw_sidecar_model='D',auxiliary_lambda=plan['shared_protocol']['lambda_aux'],normalization_sha256=sha(r/'normalization.json')))
initial=torch.load(plan['common_step0'],map_location='cpu',weights_only=False);idigest=digest_state(initial);checks=[]
for arm in plan['arms']:
 ck=torch.load(r/arm/'isolated_smoke/step0000.pt',map_location='cpu',weights_only=False);assert digest_state(ck)==idigest
 assert torch.equal(ck['torch_rng'],initial['torch_rng']) and all(torch.equal(x,y) for x,y in zip(ck['cuda_rng'],initial['cuda_rng'])) and ck['python_rng']==initial['python_rng'] and np.array_equal(ck['numpy_rng'][1],initial['numpy_rng'][1])
 checks.append(dict(arm=arm,parameter_sha256=idigest,all_initial_parameters_equal=True,RNG_equal=True));del ck
del initial
new=torch.load(r/'D-old/isolated_smoke/step0001.pt',map_location='cpu',weights_only=False);old=torch.load(Path(plan['historical_source'])/'policy_run/D/isolated_smoke/step0001.pt',map_location='cpu',weights_only=False)
save(r/'early_reproduction.json',dict(initialization=checks,old_D_step1_parameter_bitwise_equal=digest_state(new)==digest_state(old),new_hash=digest_state(new),historical_hash=digest_state(old)))
print('INITIALIZATION VERIFIED',flush=True)
