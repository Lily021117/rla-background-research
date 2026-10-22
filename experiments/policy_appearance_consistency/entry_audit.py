"""Verify the public dataset/step_loss entry using per-task normalization."""
import os,json,hashlib,random,copy
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import PairedDataset
from entry import step_loss
from method import rng_state,restore_rng
from policies.policy.vla_bc_policy_rla_unified import VLABCPolicyRLAUnified
O=Path(os.environ['RUN_DIR']);R=Path(os.environ['RLA_PROJECT_ROOT'])
data=R/'experiments/rla_background_invariance/step4_4_1_dataset_expansion/run_20261021_02'
task='pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_demo'
torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
cfg=json.load(open(O/'config_P1.json'));train=PairedDataset(data,'train',task)
val=PairedDataset(data,'validation',task,train.normalizer)
assert not {x[0]['episode'] for x in train.entries}&{x[0]['episode'] for x in val.entries}
torch.manual_seed(42);random.seed(42);np.random.seed(42);initial_rng=rng_state()
m1=VLABCPolicyRLAUnified(**cfg['model']);m1.set_normalizer(train.normalizer)
restore_rng(initial_rng)
m2=VLABCPolicyRLAUnified(**cfg['model']);m2.set_normalizer(train.normalizer)
def digest(m):
    h=hashlib.sha256()
    for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().numpy().tobytes())
    return h.hexdigest()
initial_hash=digest(m1);assert initial_hash==digest(m2)
b=next(iter(DataLoader(train,batch_size=1,shuffle=False,num_workers=0)))
start=rng_state();l1=step_loss(m1,b,cfg)
g1=torch.autograd.grad(l1['loss'],[p for p in m1.parameters() if p.requires_grad])
restore_rng(start);cfg2=dict(cfg,arm='P2',lambda_cons=0.);l2=step_loss(m2,b,cfg2)
g2=torch.autograd.grad(l2['loss'],[p for p in m2.parameters() if p.requires_grad])
assert all(torch.equal(x,y) for x,y in zip(g1,g2)) and torch.equal(l1['loss'],l2['loss'])
assert digest(m1)==digest(m2)==initial_hash
out={'status':'PASS','two_separately_constructed_models_same_seed_hash':initial_hash,
     'entry_lambda0_loss_gradient_exact_equal':True,'optimizer_updates':0,'backward_calls':2,
     'task':task,'train_episodes':len({x[0]['episode'] for x in train.entries}),'train_windows':len(train),
     'validation_episodes':len({x[0]['episode'] for x in val.entries}),'validation_windows':len(val),
     'normalizer_fit':'selected task train only; reused unchanged for validation','parameters_unchanged':True,
     'loss':float(l1['loss'].detach()),'consistency':float(l1['consistency'].detach()),
     'note':'broader audit fits190 global training windows; public per-task entry fits only selected task. Both audited separately.'}
with (O/'entry_audit.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
