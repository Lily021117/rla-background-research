"""Explicit five-appearance CPU data reader; no feature/model computation."""
from common import *
def load_episode(episode,appearance):
 if appearance not in VARIANTS:raise ValueError(appearance)
 root=ROOT/'rendered'/episode
 rgb=np.load(root/(appearance+'.npz'))
 physical=np.load(root/'physical.npz')
 assert np.array_equal(rgb['timesteps'],physical['timesteps'])
 return rgb,physical
def transition(rgb,physical,t):
 n=len(rgb['timesteps']);assert 0<=t<n
 mask=np.array([t+h<n for h in range(1,9)],dtype=bool)
 # Missing futures are absent, NOT clamped copies. A downstream loss must use mask.
 return {'current_rgb_chw':rgb['rgb'][t].transpose(0,3,1,2),'future_rgb_by_horizon':{h:rgb['rgb'][t+h].transpose(0,3,1,2) for h in range(1,9) if t+h<n},'valid_horizon_mask':mask,'proprio':physical['proprio'][t],'source_action':physical['actions'][t],'source_time':physical['states'][t,0]}
