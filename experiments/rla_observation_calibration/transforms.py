"""CPU float32 RGB perturbation after original geometry and before normalization.
No global RNG consumption and no uint8 round trip in policy inputs.
"""
import hashlib,math,types
import numpy as np
import torch
from torchvision.transforms import functional as F,InterpolationMode

def noise_seed(task,state_hash,step,camera):
 key=f'rla_observation_calibration_v1:noise:{task}:{state_hash}:{int(step)}:{camera}'
 return int.from_bytes(hashlib.sha256(key.encode()).digest()[:16],'big')
def apply(x,condition,task,state_hash,step,camera):
 assert x.shape==(3,224,224) and x.dtype==torch.float32 and x.min()>=0 and x.max()<=1
 if condition=='clean':return x
 y=x
 if condition.startswith('blur_') or condition=='blur2_noise007':
  sigma=float(condition.split('_')[1]) if condition!='blur2_noise007' else 2.
  y=F.gaussian_blur(y,2*math.ceil(3*sigma)+1,sigma) # torchvision reflect padding
 if condition.startswith('noise_') or condition=='blur2_noise007':
  sigma={'noise_003':.03,'noise_007':.07,'noise_012':.12,'blur2_noise007':.07}[condition]
  rng=np.random.Generator(np.random.PCG64(noise_seed(task,state_hash,step,camera)))
  z=torch.from_numpy(rng.standard_normal((3,224,224),dtype=np.float32))
  y=(y+sigma*z).clamp(0,1)
 if condition.startswith('crop_'):
  ratio={'crop_095':.95,'crop_085':.85,'crop_075':.75}[condition];n=math.floor(224*ratio);offset=(224-n)//2
  y=F.resize(y[:,offset:offset+n,offset:offset+n],[224,224],interpolation=InterpolationMode.BILINEAR,antialias=True)
 assert torch.isfinite(y).all() and y.min()>=0 and y.max()<=1
 return y

def intrinsic(K,condition):
 A=np.eye(3)
 if condition.startswith('crop_'):
  n=math.floor(224*{'crop_095':.95,'crop_085':.85,'crop_075':.75}[condition]);o=(224-n)//2;s=224/n
  A=np.array([[s,0,s*(.5-o)-.5],[0,s,s*(.5-o)-.5],[0,0,1.]])
 return A@K,A

class Hook:
 def __init__(self,processor):
  self.processor=processor;self.original=processor.apply_transform;self.context=None;self.camera_index=0;self.calls=0
  def transform(proc,img):return self.transform(img)
  processor.apply_transform=types.MethodType(transform,processor)
 def set_context(self,condition,task,state_hash,step):self.context=(condition,task,state_hash,step);self.camera_index=0
 def transform(self,img):
  p=self.processor;assert self.context is not None and self.camera_index<2
  camera=['agentview','robot0_eye_in_hand'][self.camera_index];self.camera_index+=1;self.calls+=1
  assert not p.tvf_do_letterbox,'Unvalidated geometry: no silent letterbox substitution'
  bases=[]
  for i in range(len(p.input_sizes)):
   b=F.to_tensor(F.center_crop(F.resize(img,**p.tvf_resize_params[i]),**p.tvf_crop_params[i]));bases.append(b)
  assert all(torch.equal(bases[0],b) for b in bases),'backbone geometries differ'
  y=apply(bases[0],*self.context,camera) # exactly once, shared by both backbone normalizations
  return torch.vstack([F.normalize(y.clone(),**p.tvf_normalize_params[i]) for i in range(len(bases))])
