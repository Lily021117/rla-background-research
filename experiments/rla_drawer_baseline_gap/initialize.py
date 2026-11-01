"""One seed build supplies paired pretrained modules and compatible action-head keys."""
import torch,hashlib
from rla_drawer_baseline_gap.common import *
def execute(g):
 from prismatic.models.action_heads import L1RegressionActionHead
 root=g['a'].run.parent;plan=read(root/'planned_run_plan.json')
 assert g['bootstrap_seed']==plan['policy_seed'] and g['initialization_recipe']
 rla=g['trainable_state']();rng=g['initialization_rng']
 with torch.random.fork_rng(devices=[0]):
  head=L1RegressionActionHead(input_dim=g['model'].llm_dim,hidden_dim=g['model'].llm_dim,action_dim=7,use_pro_version=True).to(dtype=torch.bfloat16)
 full=head.state_dict();copied=[]
 for n,v in full.items():
  source=rla['head'][n]
  if source.shape==v.shape:full[n]=source
  else:
   assert n.endswith('gating_factor') and v.numel()==1 and torch.count_nonzero(source)==0
   full[n]=torch.zeros_like(v)
  copied.append(n)
 head.load_state_dict(full,strict=True)
 standard=dict(rla);standard['head']={n:p.detach().cpu().clone() for n,p in head.named_parameters() if p.requires_grad}
 inventory=[];frozen=hashlib.sha256()
 for mod,m in g['mods'].items():
  for n,p in m.named_parameters():
   raw=p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
   if not p.requires_grad:frozen.update((mod+'.'+n).encode());frozen.update(raw)
   if 'lora_B' in n:assert torch.count_nonzero(p)==0
   inventory.append(dict(module=mod,name=n,trainable=p.requires_grad,numel=p.numel(),sha256=hashlib.sha256(raw).hexdigest(),origin='strict pretrained' if not p.requires_grad else 'official seeded constructor / constant initializer'))
 for label,params in [('standard',standard),('rla',rla)]:
  path=root/(label+'_step0.pt');assert not path.exists()
  torch.save(dict(parameters=params,global_step=0,next_physical_sample=0,**rng,policy_seed=plan['policy_seed'],optimizer_updates=0),path)
  ck=torch.load(path,map_location='cpu');assert all(torch.equal(v,ck['parameters'][m][n]) for m,ps in params.items() for n,v in ps.items())
 plan['common_step0_sha256']=sha(root/'rla_step0.pt');plan['standard_step0_sha256']=sha(root/'standard_step0.pt')
 save(root/'run_plan.json',plan)
 save(root/'initialization_audit.json',dict(status='PASS',recipe=g['initialization_recipe'],shared_vla_proprio_exact=True,compatible_head_keys=copied,common_training_RNG=True,architecture_different=True,frozen_pretrained_sha256=frozen.hexdigest(),parameters=inventory,standard_sha256=plan['standard_step0_sha256'],RLA_sha256=plan['common_step0_sha256'],no_trained_policy_loaded=True,optimizer_updates=0))
