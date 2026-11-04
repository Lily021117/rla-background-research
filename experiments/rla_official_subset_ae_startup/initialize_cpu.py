"""Official constructors on CPU, no AE checkpoint inputs and no neural forward."""
from prepare import *
import random,torch

def digest(mods):
 h=hashlib.sha256()
 for name,m in sorted(mods.items()):
  for k,v in sorted(m.state_dict().items()):
   h.update((name+'.'+k).encode());h.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()

def main():
 torch.set_num_threads(4);seed=20261104;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 sys.path.insert(0,str(O));from utils.misc import load_config
 from src import models as zoo
 cfg=load_config(str(O/'rebuttal/exp2_vla_adapter/configs/rla/16x64_libero.yaml'))
 real_load=torch.load
 def deny(*args,**kwargs):raise RuntimeError('AE checkpoint load forbidden during constructors')
 torch.load=deny
 try:mods={k:getattr(zoo,cfg['models'][k]['name'])(**cfg['models'][k]['args']) for k in ['encoder','decoder']}
 finally:torch.load=real_load
 assert not torch.count_nonzero(mods['decoder'].out_layer.weight) and not torch.count_nonzero(mods['decoder'].out_layer.bias)
 initial=digest(mods);rng=dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=None,cuda_pending_seed=20261105)
 path=R/'common_untrained_step0.pt';assert not path.exists()
 torch.save(dict(models={k:m.state_dict() for k,m in mods.items()},rng=rng,global_step=0,origin='按官方实现创建的共同未训练初始化',seed=seed),path)
 state=torch.load(path,map_location='cpu',weights_only=False)
 for k,m in mods.items():m.load_state_dict(state['models'][k],strict=True)
 assert digest(mods)==initial
 assert torch.equal(rng['cpu'],torch.get_rng_state())
 save(R/'initialization_audit.json',dict(status='CPU_CONSTRUCTION_AND_STATE_DICT_STRICT_RELOAD_PASS_GPU_FORWARD_PENDING',path=str(path),sha256=sha(path),parameter_sha256=initial,seed=seed,parameters={k:sum(v.numel() for v in m.parameters()) for k,m in mods.items()},pretrained_AE_load=False,torch_load_blocked_during_construction=True,decoder_output_zero=True,load_dir=None,auto_resume=False,image_decoder_constructed=False,DINO_constructed=False,model_forwards=0,updates=0,backward=0,cuda_rng='Not initialized on CPU. Capture new matching training RNG on authorized GPU before any updates; do not claim full GPU resume tested.',code_hashes={str(p):sha(p) for p in [O/'src/models/simple_token_transformer.py',O/'rebuttal/src/models/multiview_token_transformer.py',O/'rebuttal/src/models/plucker_embedding.py',Path(__file__)]}))
 print('CPU common step0 and strict reload PASS',initial,flush=True)

if __name__=='__main__':main()
