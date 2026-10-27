"""The existing deployed evaluator, with optimization/backward forbidden."""
import torch
from pathlib import Path
from rla_appearance_gap_eval.common import read,sha
def execute(g):
 root=g['a'].run.parent
 assert read(root/'static_acceptance.json')['status']=='PASS'
 assert read(root/'preview_review.json')['status']=='PASS'
 for p,h in read(root/'frozen_artifact_hashes.json').items():assert sha(p)==h
 for r in read(root/'checkpoint_inventory.json'):assert sha(r['checkpoint'])==r['sha256']
 def forbidden(*args,**kwargs):raise RuntimeError('appearance-gap evaluation forbids optimizer updates and backward')
 torch.Tensor.backward=forbidden;torch.autograd.backward=forbidden;torch.autograd.grad=forbidden
 torch.optim.Optimizer.step=forbidden;torch.optim.AdamW.step=forbidden;torch.optim.Adam.step=forbidden
 from rla_lr_budget_sweep.extend_evaluation import execute as existing_evaluation
 existing_evaluation(g)
 g['save']('no_update_audit.json',dict(optimizer_updates=0,AE_updates=0,backward=0,optimizer_and_autograd_calls_forbidden=True,existing_deployed_evaluator=True))
