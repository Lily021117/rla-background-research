"""Official double-view AE math and isolated on-disk feature cache."""
import sys,json,importlib.util,hashlib,os,subprocess,random
from pathlib import Path
import numpy as np
import torch
from functools import lru_cache

def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def seed():
    random.seed(20261023);np.random.seed(20261023);torch.manual_seed(20261023);torch.cuda.manual_seed_all(20261023)
def ecc():return subprocess.check_output(['nvidia-smi','-i',os.environ['RLA_GPU_UUID'],'-q','-d','ECC'],text=True)
def counters(s):return [x.strip() for x in s.splitlines() if ':' in x and any(k in x for k in ('Correctable','Uncorrectable','SRAM SM','Threshold Exceeded'))]
def load(project):
    project=Path(project);w=project/'worktrees/rla-official-0fc6e97';sys.path.insert(0,str(w))
    assert str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')==os.environ['RLA_GPU_UUID'].removeprefix('GPU-')
    assert torch.cuda.mem_get_info()[0]>=float(os.environ.get('RLA_MIN_FREE_GIB','26'))*1024**3
    torch.cuda.set_per_process_memory_fraction(.25);torch.set_num_threads(4);seed()
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    p=w/'rebuttal/exp2_vla_adapter/sidecar/extract_rla_sidecar.py';spec=importlib.util.spec_from_file_location('official_extract_consistency',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod.get_dinov3_model_for_channels=lambda c:str(project/'downloads/dinov3-vitl16-pretrain-lvd1689m')
    root=project/'experiments/official_libero_rla_consistency/run_20261023_01/downloads/extracted/runs/16x64_libero/20260725_00-39-52'
    ext=mod.RlaSidecarExtractor(str(root/'config.yaml'),str(root),step='90000',need_decoder=True,need_image_decoder=False,dino_compile=False)
    return ext,root
def entries(run):return json.loads((Path(run)/'data_manifest.json').read_text())['episodes']
def armdir(run,arm):
    registry=Path(run)/'active_models.json'
    return Path(json.loads(registry.read_text())[arm]) if registry.exists() else Path(run)/('AE_'+arm)
@lru_cache(maxsize=128)
def mapped(path):return np.load(path,mmap_mode='r')
def cached(run,e,c):return mapped(str(Path(run)/'features'/e['episode']/(c+'.npy')))
def rays(run,e):return mapped(str(Path(run)/'features'/e['episode']/'plucker.npy'))
def forward(ext,models,x0,x1,r):
    b,c,l,d=x0.shape;x0f=x0.reshape(b,c*l,d);x1f=x1.reshape(b,c*l,d)
    inverse=ext._compose_inverse_input(x_t_flat=x0f,x_T_flat=x1f)
    z,_=ext._forward_views(models['encoder'],inverse,None,2,plucker=r)
    _,pred=ext._forward_views(models['decoder'],x0f,z,2,plucker=r)
    return z,pred.reshape_as(x1)
def pair(run,e,t,h,c):
    f=cached(run,e,c);r=rays(run,e)
    return tuple(torch.from_numpy(np.array(x,copy=True)).cuda() for x in (f[t:t+1],f[t+h:t+h+1],r[t:t+1]))
