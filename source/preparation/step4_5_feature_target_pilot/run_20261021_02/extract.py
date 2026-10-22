import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
import csv,json,hashlib,subprocess,sys,time,gc,traceback
from pathlib import Path
import numpy as np,torch,decord
from PIL import Image
from omegaconf import OmegaConf
O=Path(__file__).resolve().parent;X=O.parents[1];R=O.parents[3];P=X/'step4_4_pilot/run_20261021_01';PRE=O.parent/'run_20261021_01';WT=R/'worktrees/rla-official-0fc6e97';DINO=R/'downloads/dinov3-vitl16-pretrain-lvd1689m';UUID='${GPU_UUID}'
sys.path.insert(0,str(WT));from utils.dino import DINOv3FeatureExtractor
from src.models.simple_token_transformer import SimpleTokenTransformer
torch.set_num_threads(2)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def mh(m):
 h=hashlib.sha256()
 for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def js(n,v):
 with (O/n).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False)
def tab(n,rs):
 with (O/n).open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rs for k in r)));w.writeheader();w.writerows(rs)
def ecc():
 raw=subprocess.check_output(['nvidia-smi','-i',UUID,'-q','-d','ECC'],text=True)
 return dict(raw=raw,fields=[s.strip() for s in raw.splitlines() if any(k in s for k in ['Correctable','Uncorrectable','Threshold Exceeded','Repair Pending','Unrepairable'])])
def main():
 start=time.time();protected=json.loads((PRE/'delivery_verification.json').read_text())['protected_sha256'];git=subprocess.check_output(['git','status','--short'],cwd=R,text=True)
 for p,h in protected.items():assert sha(p)==h,p
 rows=list(csv.DictReader((P/'sample_manifest.csv').open()));trans=json.loads((P/'selection_manifest.json').read_text())['transitions'];assert len(rows)==32 and len(trans)==8
 # Source reader conventions and independent existing-frame orientation evidence.
 renderer=Path('${VLA_REFERENCE_ROOT}/.venv-cpu/lib/python3.10/site-packages/mujoco/renderer.py');assert 'np.flipud(out)' in renderer.read_text()
 orient=[]
 for eid in ['sealed_163_demo_12','sealed_164_demo_13']:
  pair=R/'artifacts/action_conditioned_wm/TZ_three_seed_500/independent_data/pairs'/eid;c=json.loads((pair/'complete.json').read_text())
  for ci,cam in enumerate(['agentview','robot0_eye_in_hand']):
   vr=decord.VideoReader(c['videos']['clean'][ci],width=128,height=128,num_threads=1);v=vr[c['video_source_output_indices'].index(0)].asnumpy().astype(float);old=np.load(X/'step4_3_2_mask_validation/run_20261021_01/arrays'/f'{eid}_0_{cam}_legacy.npz')['clean'].astype(float)
   scores={k:float(abs(a-old).mean()) for k,a in [('identity',v),('vertical',v[::-1]),('horizontal',v[:,::-1]),('rotate180',v[::-1,::-1])]};assert min(scores,key=scores.get)=='vertical';orient.append(dict(episode=eid,camera=cam,MAE=scores,scope='orientation corroboration only; different renderer/codec is not pixel identity'))
 js('preprocessing_audit.json',dict(orientation=orient,renderer_source=str(renderer),renderer_SHA256=sha(renderer),pipeline='PNG top-left -> vertical flip to legacy native OpenGL rows -> lossless FFV1 -> same decord resize512 -> rotate180 -> RGB/255 -> official mean/std. Net orientation is horizontal flip of PNG; no second arbitrary rotation.',resolution=[512,512],feature_shape=[1024,1024],mask='full image, same Step2; no invented segmentation masking',camera_handling='single-view separately; no concatenation or Plucker retrofit'))
 # Preserve exact RGB pixels in a bounded streamed intermediate, then use the
 # same resize implementation as historical extraction rather than guessing PIL.
 items=[];cmd=['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','128x128','-r','10','-i','pipe:0','-c:v','ffv1','-pix_fmt','bgr0',str(O/'native_orientation_lossless.mkv')]
 proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
 for r in rows:
  for app in ['clean','background']:
   a=np.asarray(Image.open(r[app]).convert('RGB'));assert a.shape==(128,128,3);proc.stdin.write(a[::-1].copy().tobytes());items.append(dict(row=r,appearance=app,image=r[app]))
 proc.stdin.close();err=proc.stderr.read();assert proc.wait()==0,err
 native=decord.VideoReader(str(O/'native_orientation_lossless.mkv'),num_threads=1);assert len(native)==64
 for i,it in enumerate(items):assert np.array_equal(native[i].asnumpy(),np.asarray(Image.open(it['image']))[::-1])
 vr=decord.VideoReader(str(O/'native_orientation_lossless.mkv'),width=512,height=512,num_threads=1)
 assert os.environ['CUDA_VISIBLE_DEVICES']==UUID
 base=ecc();js('gpu_before.json',base);assert torch.cuda.device_count()==1;actual=str(torch.cuda.get_device_properties(0).uuid);assert actual.lower().replace('gpu-','')==UUID.lower().replace('gpu-','')
 free,total=torch.cuda.mem_get_info();assert free>=10*1024**3,('insufficient batch1 preflight headroom',free)
 cap=min(8*1024**3,free-5*1024**3);torch.cuda.set_per_process_memory_fraction(cap/total);torch.cuda.reset_peak_memory_stats();torch.manual_seed(20261021)
 js('resource_gate.json',dict(UUID=actual,free_before=free,cap_bytes=cap,reserve_min_bytes=5*1024**3,authorization='latest user explicitly authorizes GPU0 measured risk preflight below former26GiB gate; not a permanent global threshold change',physical_GPU=0,logical_GPU=0,batch=1))
 def health():
  f,_=torch.cuda.mem_get_info();assert f>=4*1024**3,('free reserve too low',f)
  assert ecc()['fields']==base['fields'],'ECC changed; stop'
 def tensor(i):return torch.from_numpy(vr[i].asnumpy()[::-1,::-1].copy()).permute(2,0,1).unsqueeze(0).float().cuda()/255
 dh=sha(DINO/'model.safetensors');assert dh=='dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179'
 dino=DINOv3FeatureExtractor(model_name=str(DINO),use_compile=False).cuda().eval().requires_grad_(False);assert dino.model_name_str==str(DINO) and dino.embed_dim==1024 and dino.patch_size==16 and 'dinov3' in dino.config.model_type;before=mh(dino)
 (O/'features').mkdir();(O/'targets').mkdir();frows=[];lookup={}
 with torch.inference_mode():
  a=tensor(0);_,test=dino(a,return_spatial_grid=False);_,again=dino(a,return_spatial_grid=False);torch.cuda.synchronize();repeat=float((test-again).abs().max());assert repeat==0;health()
  js('smoke.json',dict(stage='DINO',repeat_max=repeat,shape=list(test.shape),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),free_after=torch.cuda.mem_get_info()[0]));del a,test,again
  for i,it in enumerate(items):
   _,feat=dino(tensor(i),return_spatial_grid=False);assert feat.shape==(1,1024,1024) and torch.isfinite(feat).all();arr=feat[0].float().cpu().numpy().astype(np.float16);p=O/'features'/f'{i:03d}.npy';np.save(p,arr);r=it['row'];lookup[(r['episode'],int(r['source_state_id']),r['camera'],it['appearance'])]=str(p)
   frows.append(dict(index=i,episode=r['episode'],state_id=r['source_state_id'],output_frame=r['output_frame'],camera=r['camera'],appearance=it['appearance'],transition_ids=r['transition_ids'],image=it['image'],image_SHA256=sha(it['image']),path=str(p),SHA256=sha(p),shape='1024x1024',dtype='float16',finite=True,DINO_SHA256=dh));del feat,arr
   if i%8==0:health();print('FEATURE',i+1,'/64',flush=True)
  assert mh(dino)==before
  js('dino_hashes.json',dict(before=before,after=before));del dino;gc.collect();torch.cuda.empty_cache();health()
  tab('feature_manifest.csv',frows)
  cfgpath=Path('${DATA_ROOT}/models/rla-wm/official/rla/maniskill/20260323_21-41-25/config.yaml');cfg=OmegaConf.to_container(OmegaConf.load(cfgpath),resolve=True);targetrows=[];pairs=[];models=[]
  for group in ['B','C']:
   cp=X/f'step2_training/{group}/ckpts/encoder_step0000200.pt';h=sha(cp);assert h==protected[str(cp)];model=SimpleTokenTransformer(**cfg['models']['encoder']['args']);state=torch.load(cp,map_location='cpu',weights_only=True);model.load_state_dict(state,strict=True);del state;model=model.cuda().eval().requires_grad_(False);bh=mh(model);health();repeated=False
   for ti,t in enumerate(trans):
    for cam in ['agentview','robot0_eye_in_hand']:
     zz={};ff={}
     for app in ['clean','background']:
      ps=[lookup[(t['episode'],t[k],cam,app)] for k in ['source_start','source_future']];features=[torch.from_numpy(np.load(p).copy()).float().unsqueeze(0).cuda() for p in ps];residual=features[1]-features[0];z,_=model(residual);assert z.shape==(1,32,64) and torch.isfinite(z).all();za=z[0].float().cpu().numpy()
      if not repeated:
       z2,_=model(residual);assert float((z-z2).abs().max())==0;repeated=True
      p=O/'targets'/f'{group}_{ti}_{cam}_{app}.npy';np.save(p,za);zz[app]=za.astype(float).ravel();ff[app]=[x.cpu().numpy().astype(float) for x in features]
      targetrows.append(dict(group=group,transition_id=t['transition_id'],episode=t['episode'],camera=cam,appearance=app,state_t=t['source_start'],state_future=t['source_future'],action_start=t['action_start_inclusive'],action_stop=t['action_stop_exclusive'],path=str(p),SHA256=sha(p),encoder=str(cp),encoder_SHA256=h,shape='32x64',norm=float(np.linalg.norm(za)),variance=float(za.var()),finite=True,current_feature=ps[0],future_feature=ps[1]));del z,residual,features
     a,b=zz['clean'],zz['background'];pairs.append(dict(group=group,transition_id=t['transition_id'],episode=t['episode'],camera=cam,mse=float(np.mean((a-b)**2)),relative_l2=float(np.linalg.norm(a-b)/(np.linalg.norm(a)+1e-12)),cosine_similarity=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12)),clean_norm=float(np.linalg.norm(a)),background_norm=float(np.linalg.norm(b)),current_DINO_MSE=float(np.mean((ff['clean'][0]-ff['background'][0])**2)),future_DINO_MSE=float(np.mean((ff['clean'][1]-ff['background'][1])**2))))
    health();print('TARGET',group,ti+1,'/8',flush=True)
   assert mh(model)==bh;models.append(dict(group=group,checkpoint=str(cp),SHA256=h,strict=True,parameter_hash_before=bh,parameter_hash_after=bh,repeat_max=0));del model;gc.collect();torch.cuda.empty_cache()
 tab('rla_target_manifest.csv',targetrows);tab('per_transition_metrics.csv',pairs);js('model_verification.json',models)
 # Preserve previously verified temporal/source audit, explicitly NOT a VLA-state acceptance.
 props=list(csv.DictReader((PRE/'proprio_alignment_audit.csv').open()));tab('proprio_alignment_audit.csv',props)
 for r in frows+targetrows:assert sha(r['path'])==r['SHA256'];assert np.isfinite(np.load(r['path'])).all()
 for p,h in protected.items():assert sha(p)==h,p
 assert subprocess.check_output(['git','status','--short'],cwd=R,text=True)==git;health();js('gpu_after.json',ecc())
 js('delivery_verification.json',dict(status='PARTIAL',feature_target_stage='PASS',features=64,physical_transitions=8,target_arrays=len(targetrows),metric_rows=len(pairs),shape='32x64',proprio_alignment=False,VLA_batch_ready=False,protected_unchanged=True,GPU=actual,training_updates=0,backward=0,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),elapsed_seconds=time.time()-start,OOM=0,NaN_Inf=False,DINO_repeat_max=repeat))
 print('COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as exc:
  js('failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),stop_no_other_GPU=True));raise
