"""One zero-update shared initialization per seed; no AE/teacher/target generation."""
import json,hashlib,random
from pathlib import Path
import numpy as np
import torch
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def digest(state):
 h=hashlib.sha256()
 for k in sorted(state):
  for n,p in sorted(state[k].items()):h.update((k+'.'+n).encode());h.update(p.contiguous().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def execute(g):
 root=g['a'].run.parent;plan=json.loads((root/'planned_run_plan.json').read_text());seed=plan['policy_seed'];initial=torch.load(plan['original_common_step0'],map_location='cpu');assert initial['global_step']==0 and sha(plan['original_common_step0'])==plan['original_common_step0_sha256']
 scope=plan['initialization_scope'];mods=g['mods']
 if scope=='fixed_action_path_new_auxiliary':
  for k,m in mods.items():
   names={n for n,p in m.named_parameters() if p.requires_grad};assert names==set(initial['parameters'][k]);full=m.state_dict();full.update(initial['parameters'][k]);m.load_state_dict(full,strict=True)
 from control_preserving_rla.interface import Interface,reset_auxiliary
 interface=Interface(g,plan);order=json.loads(Path(plan['inputs']['B20_sampling_order']).read_text());examples=[(e,t,c) for e,t in order[:4] for c in ['clean','background']]
 for m in mods.values():m.eval()
 batch=interface.batch(examples)
 with torch.inference_mode():before=interface.forward(batch,'Inference')[0].detach().clone()
 if scope=='fixed_action_path_new_auxiliary':reset_auxiliary(g,seed)
 changed=[]
 for k,m in mods.items():
  for n,p in m.named_parameters():
   if p.requires_grad and not torch.equal(p.detach().cpu(),initial['parameters'][k][n]):changed.append(k+'.'+n)
 if scope=='fixed_action_path_new_auxiliary':assert set(changed).issubset({'head.model.layer_norm2_rla.weight','head.model.layer_norm2_rla.bias','head.model.fc2_rla.weight','head.model.fc2_rla.bias'})
 assert scope=='all_random_modules_new_seed' and g['initialization_recipe']
 rng=g['initialization_rng']
 inventory=[];frozen=hashlib.sha256()
 for k,m in mods.items():
  for n,p in m.named_parameters():
   raw=p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes();ph=hashlib.sha256(raw).hexdigest()
   if not p.requires_grad:
    source='strict pretrained';rule='loaded unchanged (BF16 cast)';frozen.update((k+'.'+n).encode());frozen.update(raw)
   elif 'lora_B' in n:
    source='new seed construction';rule='constant zero';assert torch.count_nonzero(p)==0
   elif 'lora_A' in n:source='new seed construction';rule='Gaussian std=1/r, r=64'
   elif 'action_queries' in n:source='new seed construction';rule='official constructor followed by HF post_init Embedding normal initializer; not pretrained'
   elif 'gating_factor' in n:
    source='new seed construction';rule='constant zero';assert torch.count_nonzero(p)==0
   elif 'layer_norm' in n.lower() or 'layernorm' in n.lower():source='new seed construction';rule='official LayerNorm constants'
   else:source='new seed construction';rule='official module constructor; baseline head copied into RLA overlay where shared'
   inventory.append(dict(module=k,name=n,numel=p.numel(),trainable=p.requires_grad,source=source,rule=rule,sha256=ph))
 with (root/'module_initialization_audit.json').open('x') as f:json.dump(dict(recipe=g['initialization_recipe'],parameters=inventory,frozen_pretrained_sha256=frozen.hexdigest(),historical_recipe='baseline seed20261023 plus forked RLA constructor seed20261024; new runs use seed and seed+1'),f,indent=2)
 outputs=[]
 from stove_centering.training import restore
 with torch.inference_mode():
  for repeat in range(2):
   restore(rng);action,z,loss=interface.forward(batch,'Inference');assert torch.isfinite(action).all() and torch.isfinite(z).all() and torch.isfinite(loss);outputs.append((action.detach().clone(),z.detach().clone()))
 assert all(torch.equal(x,y) for x,y in zip(outputs[0],outputs[1]))
 if scope=='fixed_action_path_new_auxiliary':assert torch.equal(before,outputs[0][0]),'aux readout reset altered action path'
 interface.clear();restore(rng);state=g['trainable_state']();path=root/'common_seed_step0.pt';assert not path.exists();torch.save(dict(parameters=state,global_step=0,next_physical_sample=0,**rng,initialization_scope=scope,policy_seed=seed,original_common_step0_sha256=plan['original_common_step0_sha256'],optimizer_updates=0),path)
 ck=torch.load(path,map_location='cpu');assert digest(state)==digest(ck['parameters']);plan['common_step0_sha256']=sha(path)
 with (root/'run_plan.json').open('x') as f:json.dump(plan,f,indent=2)
 audit=dict(status='PASS',seed=seed,scope=scope,frozen_pretrained_sha256=frozen.hexdigest(),common_seed_step0_sha256=plan['common_step0_sha256'],parameter_hash=digest(state),changed_vs_historical_step0=changed,action_unchanged_by_aux_reset='not applicable: full original initialization recipe',action_repeated_exact=True,aux_repeated_exact=True,action_hash=hashlib.sha256(outputs[0][0].cpu().view(torch.uint8).numpy().tobytes()).hexdigest(),aux_hash=hashlib.sha256(outputs[0][1].cpu().view(torch.uint8).numpy().tobytes()).hexdigest(),same_checkpoint_for_C_D=True,AE_updates=0,optimizer_updates=0,backward=0,sample_order_sha256=plan['inputs']['B20_sampling_sha256'],no_trained_policy_initialization=True)
 with (root/'initialization_gate.json').open('x') as f:json.dump(audit,f,indent=2)
 print('SEED_INITIALIZATION_PASS',seed,scope,len(changed),flush=True)
