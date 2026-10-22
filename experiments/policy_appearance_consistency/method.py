"""Paired appearance BC; use the pinned original BC-ResNet, with no RLA modules.

Inputs must already be normalized with training-only statistics. The only
allowed observations are current image and state. No optimizer is created here.
"""
import random
import numpy as np
import torch


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def restore_rng(s):
    random.setstate(s[0]); np.random.set_state(s[1]); torch.set_rng_state(s[2])
    if s[3] is not None:
        torch.cuda.set_rng_state_all(s[3])


def predict_normalized(policy, obs):
    assert set(obs) == {'image', 'state'}
    assert policy.use_latent_head is False
    assert policy.latent_head is policy.latent_encoder is policy.dino_extractor is None
    return policy._decode_action(policy._forward_trunk(obs))


def paired_forward(policy, clean, background):
    assert torch.equal(clean['state'], background['state'])
    before = rng_state()
    a = predict_normalized(policy, clean)
    after = rng_state()
    restore_rng(before)
    try:
        b = predict_normalized(policy, background)
    finally:
        restore_rng(after)  # consume one stochastic stream per physical pair
    return a, b


def masked_mean(x, mask):
    assert mask.dtype == torch.bool and mask.shape == x.shape[:2]
    expanded = mask[..., None].expand_as(x)
    assert expanded.any()
    return x.masked_select(expanded).mean()


def objective(a, b, target, mask, arm, lambda_cons):
    assert a.shape == b.shape == target.shape
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert torch.isfinite(target[mask]).all()
    assert arm in ['P0', 'P1', 'P2'] and lambda_cons >= 0
    bc_clean = masked_mean((a-target).square(), mask)
    bc_background = masked_mean((b-target).square(), mask)
    bc = bc_clean if arm == 'P0' else .5*(bc_clean+bc_background)
    consistency = masked_mean((a-b).square(), mask)
    weight = lambda_cons if arm == 'P2' else 0.
    return {'loss': bc+weight*consistency, 'bc': bc,
            'consistency': consistency, 'weighted_consistency': weight*consistency}
