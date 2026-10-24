"""Read saved real step0 states on CPU; no model forward or update."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,torch
p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);a=p.parse_args();root=a.task/'policy_run'
def load(arm):return torch.load(root/arm/'isolated_smoke/step0000.pt',map_location='cpu')
c=load('C');d=load('D');assert c['global_step']==d['global_step']==0 and c['next_physical_sample']==d['next_physical_sample']==0
digest=hashlib.sha256();count=0
for module,parameters in c['parameters'].items():
 assert parameters.keys()==d['parameters'][module].keys()
 for name,tensor in parameters.items():assert torch.equal(tensor,d['parameters'][module][name]);digest.update((module+'.'+name).encode());digest.update(tensor.view(torch.uint8).numpy().tobytes());count+=tensor.numel()
assert torch.equal(c['torch_rng'],d['torch_rng']) and len(c['cuda_rng'])==len(d['cuda_rng'])
assert all(torch.equal(x,y) for x,y in zip(c['cuda_rng'],d['cuda_rng'])) and c['python_rng']==d['python_rng']
for x,y in zip(c['numpy_rng'],d['numpy_rng']):np.testing.assert_array_equal(x,y)
assert c['configuration']==d['configuration'] and c['activation_checkpointing']==d['activation_checkpointing']==True and c['dtype']==d['dtype']=='bfloat16'
target_audit=json.loads((root/'preflight6/isolated_smoke/target_gradient_audit.json').read_text());assert target_audit['exact_C_D_action_only_match']
row=dict(status='PASS_SAVED_C_D_STEP0_EXACT',parameters=count,trainable_parameter_digest=digest.hexdigest(),CPU_CUDA_Python_NumPy_RNG_equal=True,configuration_equal=True,initial_optimizer_states_empty=not c['optimizer']['state'] and not d['optimizer']['state'],common_untrained_reference=c['configuration']['common_step0'],target_difference_only='C uses AE-C; D uses AE-D; fixed common auxiliary coefficient and normalization',preflight=target_audit,updates_by_this_audit=0)
with (root/'initialization_and_rng_audit.json').open('x') as f:json.dump(row,f,indent=2)
print(json.dumps({k:v for k,v in row.items() if k!='preflight'},indent=2))
