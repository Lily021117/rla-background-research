"""Save newly constructed full initialization; no historical trainable loading."""
import json,hashlib,torch
from rla_scratch_low_seeds.common import read,save,sha
def execute(g):
 root=g['a'].run.parent;plan=read(root/'planned_run_plan.json');assert g['bootstrap_seed']==plan['policy_seed'] and g['initialization_recipe']
 inventory=[];frozen=hashlib.sha256()
 for k,m in g['mods'].items():
  for n,p in m.named_parameters():
   raw=p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes();h=hashlib.sha256(raw).hexdigest()
   if not p.requires_grad:source='strict pretrained loading, BF16 cast';frozen.update((k+'.'+n).encode());frozen.update(raw)
   else:
    source='official constructor random or constant rule under new seed'
    if 'lora_B' in n or 'gating_factor' in n:assert torch.count_nonzero(p)==0;source='official constant zero initialization'
   inventory.append(dict(module=k,name=n,trainable=p.requires_grad,numel=p.numel(),source=source,sha256=h))
 state=g['trainable_state']();path=root/'common_seed_step0.pt';assert not path.exists()
 torch.save(dict(parameters=state,global_step=0,next_physical_sample=0,**g['initialization_rng'],policy_seed=plan['policy_seed'],optimizer_updates=0),path)
 ck=torch.load(path,map_location='cpu')
 assert all(torch.equal(v,ck['parameters'][k][n]) for k,values in state.items() for n,v in values.items())
 plan['common_step0_sha256']=sha(path);save(root/'run_plan.json',plan)
 save(root/'module_initialization_audit.json',dict(recipe=g['initialization_recipe'],parameters=inventory,frozen_pretrained_sha256=frozen.hexdigest(),no_historical_trainable_weights_loaded=True,strict_roundtrip=True,seed=plan['policy_seed'],optimizer_updates=0))
 print('NEW_COMMON_STEP0_SAVED',plan['policy_seed'],flush=True)
