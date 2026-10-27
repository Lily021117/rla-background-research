"""CPU-only deterministic sampler tests; zero optimizer/backward/rollouts."""
import pickle,unittest
import numpy as np
from experiments.rla_sampling_stability.experiment import balanced

class SamplerTests(unittest.TestCase):
 def test_balanced_without_global_rng_effect(self):
  lengths={f'episode_{i}':112+i for i in range(20)}
  before=pickle.dumps(np.random.get_state())
  a=balanced(lengths,20261101);b=balanced(lengths,20261102)
  self.assertEqual(before,pickle.dumps(np.random.get_state()))
  self.assertEqual(a,balanced(lengths,20261101));self.assertNotEqual(a,b)
  for order in [a,b]:
   for start,end,count in [(0,4800,240),(0,1600,80),(1600,3200,80),(3200,4800,80),(4000,4800,40)]:
    for e in lengths:self.assertEqual(sum(ep==e for ep,t in order[start:end]),count)
   for e,n in lengths.items():
    times=[t for ep,t in order if ep==e]
    self.assertEqual(set(times),set(range(n)))
    for start in range(0,len(times),n):self.assertEqual(len(set(times[start:start+n])),len(times[start:start+n]))

if __name__=='__main__':unittest.main()
