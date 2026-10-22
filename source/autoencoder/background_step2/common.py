import hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np
ROOT=Path('${RLA_PROJECT_ROOT}')
OUT=Path(os.environ['RUN_DIR'])
WT=Path(os.environ['STEP2_WT'])
READY=ROOT/'experiments/rla_background_invariance/step2_readiness'
STEP1=READY.parent/'baseline_eval/official_attempt'
MODEL=Path('${DATA_ROOT}/models/rla-wm/official/rla/maniskill/20260323_21-41-25')
UUID='${GPU_UUID}'
DINO=ROOT/'downloads/dinov3-vitl16-pretrain-lvd1689m'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),p
 p.write_text(json.dumps(x,indent=2)+'\n')
def append(p,x):
 with open(p,'a') as f:f.write(json.dumps(x)+'\n');f.flush()
def shell(args):return subprocess.check_output(args,text=True)
def ecc():
 raw=shell(['nvidia-smi','-i',UUID,'-q','-d','ECC'])
 fields=[l.strip() for l in raw.splitlines() if any(k in l for k in ('Correctable','Uncorrectable','Threshold Exceeded','Repair Pending','Unrepairable'))]
 return dict(raw=raw,fields=fields)
def health():
 current=ecc();assert current['fields']==read(OUT/'gpu_ecc_baseline.json')['fields'],'ECC increment/change: stop'
 return current
def gpu_gate():
 import torch
 assert os.environ['CUDA_VISIBLE_DEVICES']==UUID and torch.cuda.device_count()==1
 p=torch.cuda.get_device_properties(0);assert str(p.uuid).lower().replace('gpu-','')==UUID.lower().replace('gpu-','')
 free,total=torch.cuda.mem_get_info()
 # Latest user explicitly permits a measured smoke test below old 32GiB line.
 assert free>12*1024**3,('Insufficient smoke headroom',free)
 cap=min(24*1024**3,free-4*1024**3)
 torch.cuda.set_per_process_memory_fraction(cap/total,0)
 health()
 return dict(uuid=str(p.uuid),name=p.name,free_bytes=free,total_bytes=total,torch_cap_bytes=cap,safety_reserve_bytes=4*1024**3)
def resources():
 import resource
 vals={}
 for l in Path('/proc/self/status').read_text().splitlines():
  if l.startswith(('VmRSS:','VmHWM:')):k,v=l.split(':');vals[k]=v.strip()
 st=os.statvfs(OUT)
 return dict(cpu=vals,disk_free_bytes=st.f_bavail*st.f_frsize,io=Path('/proc/self/io').read_text(),timestamp=time.time())
def model_hash(m):
 h=hashlib.sha256()
 for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def models():
 import torch
 from src.models.simple_token_transformer import SimpleTokenTransformer
 from omegaconf import OmegaConf
 import src.models.simple_token_transformer as impl
 assert str(WT) in impl.__file__
 cfg=OmegaConf.to_container(OmegaConf.load(MODEL/'config.yaml'),resolve=True)
 nets={}
 for name in ('encoder','decoder'):
  net=SimpleTokenTransformer(**cfg['models'][name]['args'])
  p=MODEL/'ckpts'/f'{name}_step0100000.pt'
  state=torch.load(p,map_location='cpu',weights_only=True)
  net.load_state_dict(state,strict=True);del state
  nets[name]=net.cuda()
 return nets
def batch(indices):
 import torch
 records=read(OUT/'training_records.json')
 samples=[]
 for index in indices:
  r=records[index];streams=[]
  for appearance in ('clean','background'):
   a=np.load(r['cache'][appearance],mmap_mode='r')
   streams.append(torch.from_numpy(a[[r['current_cache_index'],r['future_cache_index']]].copy()).float())
  samples.append(torch.stack(streams))
 # B, appearance, time, single-camera, patch, feature
 return dict(features=torch.stack(samples).unsqueeze(3))
