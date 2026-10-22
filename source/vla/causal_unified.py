"""One-way auxiliary supervision; native policy attention is never modified.

No target argument exists on predict_action. Eight native action features each
predict 4x64 latent coordinates: learned 896->256 projection, full 32x64 target.
This is a NEW auxiliary architecture, not the previous SparseHead experiment.
"""
from dataclasses import dataclass
import torch
from torch import nn

@dataclass(frozen=True)
class Arm:
    name: str
    frozen: bool
    aux_enabled: bool
    aux_loss_weight: float
    target_source: str

ARMS = {
    'A': Arm('A', True, False, 0., 'B'),
    'B': Arm('B', False, False, 0., 'B'),
    'C': Arm('C', False, True, .05, 'B'),
    'D': Arm('D', False, True, .05, 'C'),
}

class UnifiedHead(nn.Module):
    def __init__(self, native, seed=20261021):
        super().__init__()
        self.native = native
        dim = native.hidden_dim
        # Fork initialization RNG so construction cannot change policy RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.auxiliary = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4*64))
        self.auxiliary.to(device=next(native.parameters()).device, dtype=next(native.parameters()).dtype)
        self.last_hidden = None
        self.last_action = None
        self.last_aux = None
        self._hook = native.model.layer_norm2.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self.last_hidden = output  # No detach: auxiliary supervision reaches native trunk/VLM.

    def predict_action(self, *args, **kwargs):
        self.last_hidden = None
        self.last_action = self.native.predict_action(*args, **kwargs)
        assert self.last_hidden.shape[1] == 8
        self.last_aux = self.auxiliary(self.last_hidden).reshape(self.last_hidden.shape[0], 1, 2048)
        return self.last_action

    def clear(self):
        self.last_hidden = self.last_action = self.last_aux = None

def objective(action_loss, z_hat, target, arm):
    assert arm.aux_enabled == (arm.aux_loss_weight > 0)
    assert z_hat.shape == target.shape and z_hat.shape[1:] == (1, 2048)
    aux = torch.nn.functional.l1_loss(z_hat.float(), target.float())
    total = action_loss + arm.aux_loss_weight * aux
    if not torch.isfinite(total):
        raise FloatingPointError('nonfinite objective')
    return total, aux

def configure_trainability(model, head, proprio, arm):
    for name, p in model.named_parameters():
        p.requires_grad_(not arm.frozen and ('lora_' in name or 'action_queries' in name))
    head.requires_grad_(not arm.frozen)
    proprio.requires_grad_(not arm.frozen)

def named_trainables(model, head, proprio):
    return [(prefix+'.'+n,p) for prefix,m in [('vlm',model),('head',head),('proprio',proprio)]
            for n,p in m.named_parameters() if p.requires_grad]

def rng_state():
    import random,numpy as np
    return {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),
            'numpy':np.random.get_state(),'python':random.getstate()}

def set_rng(s):
    import random,numpy as np
    torch.set_rng_state(s['torch']);torch.cuda.set_rng_state_all(s['cuda'])
    np.random.set_state(s['numpy']);random.setstate(s['python'])

def save_checkpoint(path, model, head, proprio, optimizer, scheduler, step, provenance):
    from pathlib import Path
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    torch.save({'step':step,'vlm':{n:p.detach().cpu() for n,p in model.named_parameters()
               if 'lora_' in n or 'action_queries' in n},'head':head.state_dict(),
               'proprio':proprio.state_dict(),'optimizer':optimizer.state_dict(),
               'scheduler':scheduler.state_dict(),'rng':rng_state(),'provenance':provenance},path)

def restore_checkpoint(path, model, head, proprio, optimizer, scheduler):
    s=torch.load(path,map_location='cpu',weights_only=False)
    expected={n for n,p in model.named_parameters() if 'lora_' in n or 'action_queries' in n}
    assert expected==set(s['vlm'])
    with torch.no_grad():
        for n,p in model.named_parameters():
            if n in expected:p.copy_(s['vlm'][n].to(p.device))
    head.load_state_dict(s['head'],strict=True);proprio.load_state_dict(s['proprio'],strict=True)
    optimizer.load_state_dict(s['optimizer']);scheduler.load_state_dict(s['scheduler']);set_rng(s['rng'])
    return s['step']

def run_training(*, model, head, proprio, arm, optimizer, scheduler, forward,
                 load_batch, sampling_order, target_for, checkpoint, log,
                 start_step=0, budget=100, authorized=False):
    """Common B/C/D loop. Not invoked by Step5.6 acceptance.

    Any nonfinite/OOM exception stops. No hidden retry, no budget extension.
    Caller restores optimizer/RNG before resume, records lost computations.
    """
    if not authorized:raise PermissionError('Explicit later training authorization required')
    if arm.frozen:raise ValueError('A never trains')
    assert budget==100 and len(sampling_order)==100 and 0<=start_step<=budget
    configure_trainability(model,head,proprio,arm)
    model.train();head.train();proprio.train()
    for step in range(start_step+1,budget+1):
        optimizer.zero_grad(set_to_none=True)
        losses=[]
        for offset in [0,2]:
            import random,numpy as np
            seed=20261021+step*10+offset
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            ids=sampling_order[step-1][offset:offset+2]
            batch=load_batch(ids)
            action=forward(batch,'Training')
            total,aux=objective(action,head.last_aux,target_for(ids,arm.target_source),arm)
            (total/2).backward()
            losses.append({'action':float(action.detach()),'aux':float(aux.detach()),'total':float(total.detach())})
            head.clear()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for _,p in named_trainables(model,head,proprio)):
            raise FloatingPointError('nonfinite gradient; no optimizer step')
        optimizer.step();scheduler.step()
        if any(not torch.isfinite(p).all() for _,p in named_trainables(model,head,proprio)):
            raise FloatingPointError('nonfinite parameter; stop and retain failure evidence')
        log({'successful_update':step,'losses':losses})
        if step in [1,50,100]:checkpoint(step)
