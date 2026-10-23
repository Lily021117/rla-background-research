"""Minimal objective extension of the official multiview trainer.

No separate camera-latent correspondence: z is one joint 16x64 query sequence.
Current/future observations in each appearance retain their own reconstruction
target. Future information must never be passed into the VLA observation.
"""
import torch
import torch.nn.functional as F


def paired_ae_loss(trainer, models, clean, background, lambda_cons):
    assert lambda_cons >= 0
    assert trainer.num_views == 2
    for k in ('intrinsics','w2c'):
        assert torch.equal(clean[k], background[k]), k
    # Both branches use the same stochastic state; original AE has no intended
    # appearance-specific randomization within this forward.
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    c = trainer.inference_batch(models, clean, training=True)
    after_cpu = torch.get_rng_state()
    after_cuda = torch.cuda.get_rng_state_all() if cuda is not None else None
    torch.set_rng_state(cpu)
    if cuda is not None: torch.cuda.set_rng_state_all(cuda)
    try: b = trainer.inference_batch(models, background, training=True)
    finally:
        torch.set_rng_state(after_cpu)
        if after_cuda is not None: torch.cuda.set_rng_state_all(after_cuda)
    assert c['enc_tokens'].shape[-2:] == b['enc_tokens'].shape[-2:] == (16,64)
    def rec(o):
        assert o['x1_pred'].shape == o['x1_gt'].shape
        return trainer.lambda_l1*F.l1_loss(o['x1_pred'],o['x1_gt']) + trainer.lambda_mse*F.mse_loss(o['x1_pred'],o['x1_gt'])
    rc,rb=rec(c),rec(b)
    consistency=F.mse_loss(c['enc_tokens'],b['enc_tokens'])
    return {'loss':.5*(rc+rb)+lambda_cons*consistency,
            'reconstruction_clean':rc,'reconstruction_background':rb,
            'consistency':consistency,'z_clean':c['enc_tokens'],'z_background':b['enc_tokens']}


class AppearanceStores:
    """Variant-isolated official RlaStore objects; no colliding raw-state key.

    Use separate full sidecars per appearance and an explicit branch selector.
    The stock single global store cannot distinguish two equal proprio streams.
    Never concatenate variants into one state-key dictionary.
    """
    def __init__(self, clean, background):
        if clean is background: raise ValueError('Variant stores must be independent')
        self.stores={'clean':clean,'background':background}

    def frame(self,appearance,states,t):
        if appearance not in self.stores: raise KeyError(appearance)
        store=self.stores[appearance]
        slot=store.slot(states)
        return store.frame(slot,t)


def require_full_chunk(index,episode,anchor,cameras=('agentview','robot0_eye_in_hand')):
    missing=[(episode,anchor+h,camera) for h in range(9) for camera in cameras
             if (episode,anchor+h,camera) not in index]
    if missing: raise ValueError(f'Missing real images/geometry for cumulative h1..8: {missing}')
    return [[index[episode,anchor+h,camera] for camera in cameras] for h in range(9)]
