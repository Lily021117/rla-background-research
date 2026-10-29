"""Bounded, paired AE feasibility training from constructors, never pretrained AE weights.

Only cached frozen DINO features and accepted camera rays are read. No policy or simulator.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import argparse, csv, hashlib, json, random, subprocess, time, math, sys
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
import torch.nn.functional as F

P=Path('/data/lyc/projects/rla-wm')
OFF=P/'worktrees/rla-official-0fc6e97'
ROOT=P/'experiments/rla_ae_from_scratch/run_20261029_01'
GPU='GPU-205a3f5a-d91b-6385-c588-4e4cf3505f7e'
SEED=20261029
UPDATES=2000
BATCH=8
CHECKS=[0,50,250,500,1000,1500,2000]
SOURCES={
 'cookie':P/'experiments/official_libero_rla_consistency/run_20261023_08',
 'drawer_bowl':P/'experiments/official_libero_rla_consistency/run_20261023_10/drawer/ae_run_recovery1',
 'stove':P/'experiments/official_libero_rla_consistency/run_20261023_10/stove/ae_run',
}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2)
def csvsave(p,rows):
 with Path(p).open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def seeds():
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def setrng(r):
 random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['cpu']);torch.cuda.set_rng_state_all(r['cuda'])
def ecc():return subprocess.check_output(['nvidia-smi','-i',GPU,'-q','-d','ECC'],text=True)
def ec(s):return [x.strip() for x in s.splitlines() if ':' in x and any(k in x for k in ['Correctable','Uncorrectable','SRAM SM','Threshold Exceeded'])]
def digest(mods):
 h=hashlib.sha256()
 for name,m in sorted(mods.items()):
  for k,v in sorted(m.state_dict().items()):h.update((name+'.'+k).encode());h.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def prep():
 episodes=[];inventory=[];protected={}
 for task,src in SOURCES.items():
  mp=src/'data_manifest.json';d=json.loads(mp.read_text());assert d['status']=='PASS'
  protected[str(mp)]=sha(mp)
  provenance='frozen_input_inventory.json' if task=='cookie' else 'source_provenance.json'
  for fn in ['feature_acceptance.json',provenance,'feature_manifest.json']:
   p=src/fn;assert p.exists();protected[str(p)]=sha(p)
  accept=json.loads((src/'feature_acceptance.json').read_text());assert accept['status']=='PASS'
  counts={}
  for e0 in d['episodes']:
   e=dict(e0,task_key=task,feature_root=str(src/'features'/e0['episode']))
   assert e['frames']-8==e['valid_chunks']
   split=e['split'];assert split in ['train','validation'];counts.setdefault(split,[0,0,0])
   counts[split][0]+=1;counts[split][1]+=e['frames'];counts[split][2]+=e['valid_chunks']
   for appearance in ['clean','background']:
    x=np.load(Path(e['feature_root'])/(appearance+'.npy'),mmap_mode='r')
    assert x.shape==(e['frames'],2,196,1024) and np.isfinite(x[[0,-1]]).all()
   rays=np.load(Path(e['feature_root'])/'plucker.npy',mmap_mode='r');assert rays.shape==(e['frames'],392,6) and np.isfinite(rays[[0,-1]]).all()
   episodes.append(e)
  assert counts['train'][0]==20 and counts['validation'][0]==5
  for split,(n,frames,windows) in counts.items():inventory.append(dict(task=task,split=split,episodes=n,frames=frames,complete_anchors=windows,h1_to_h8_transitions=windows*8,selected=True))
 # Inventory already accepted new operation task, but deliberately do not mix its separately
 # prepared cache protocol into this first bounded run.
 q=P/'experiments/rla_new_task_baseline/run_20261028_02/dataset_v2/frozen_dataset_manifest.json'
 if q.exists():
  other=json.loads(q.read_text());protected[str(q)]=sha(q)
  save(ROOT/'other_task_inventory_metadata.json',other)
 ids=[(e['source_hdf5'],e['source_demo']) for e in episodes];assert len(ids)==len(set(ids))
 save(ROOT/'data_manifest.json',dict(episodes=episodes,development_role='exploratory, repeatedly used historically',new_data=False,heldout_read=False))
 csvsave(ROOT/'data_inventory.csv',inventory);save(ROOT/'protected_inputs.json',protected)
 pool=[(e['episode'],t,h) for e in episodes if e['split']=='train' for t in range(e['valid_chunks']) for h in range(1,9)]
 rg=np.random.default_rng(SEED);order=[]
 while len(order)<UPDATES*BATCH:order.extend([pool[int(i)] for i in rg.permutation(len(pool))])
 order=order[:UPDATES*BATCH];save(ROOT/'sampling_order.json',order)
 fixed=[]
 for task in SOURCES:
  es=[e for e in episodes if e['task_key']==task]
  for split in ['train','validation']:
   chosen=sorted([e for e in es if e['split']==split],key=lambda e:e['episode'])
   if split=='train':chosen=chosen[:2]
   for e in chosen:
    for t in sorted(set([0,e['valid_chunks']//2,e['valid_chunks']-1])):
     for h in [1,4,8]:fixed.append(dict(episode=e['episode'],task=task,split=split,t=t,h=h))
 save(ROOT/'fixed_evaluation_samples.json',fixed)
 calib=[]
 for task in SOURCES:
  es=sorted([e for e in episodes if e['task_key']==task and e['split']=='train'],key=lambda e:e['episode'])
  for e,h in zip(es[:2],[1,8]):calib.append((e['episode'],e['valid_chunks']//2,h))
 save(ROOT/'calibration_samples.json',calib)
 plan=dict(name='official-constructor common untrained AE initialization; NOT author original step0',official_commit='0fc6e97212442ebccffa06f1d42f5481e5f22950',base_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),seed=SEED,updates_each=UPDATES,pilot_updates=60,pilot_nodes=[0,10,30,60],effective_physical_batch=BATCH,appearance_batch=16,microbatch_physical=1,accumulation=8,lr=1e-4,optimizer='AdamW',weight_decay=0.,betas=[.9,.999],eps=1e-8,scheduler=None,precision='official mixed FP16 torso + FP32 masters; fixed loss scale256',backend='Math SDPA',checks=CHECKS,save_every=250,calibration_rule='train-only rec-only pilot60; lambda=min(median(0.1*encoder_rec_grad/encoder_cons_grad at10,30,60), min(0.1*rec/cons at10,30,60)); fail if nonfinite/zero poststartup gradients. No development-based choice. Step0 measured but excluded because zero decoder output blocks encoder rec gradient.',lambda_frozen_before_formal=True,rec='0.5*(mean_L1_clean+mean_MSE_clean+mean_L1_bg+mean_MSE_bg)',cons='mean MSE over B/16/64, raw latent',random_matching='restore RNG before background forward; retain one branch advancement; C/D identical order and initial RNG',horizons=list(range(1,9)),physical_exposures_per_arm=UPDATES*BATCH,unique_training_demonstrations=60,window_equivalent_traversals=UPDATES*BATCH/sum(e['valid_chunks'] for e in episodes if e['split']=='train'),sample_rule='uniform permutation over complete episode/anchor/horizon tuples, cycle if needed; not task-balanced',budget_rationale='bounded 2000 updates, not inherited600; 16000 physical pair exposures across60 demos; compared with official batch128 and1M maximum is only feasibility, not convergence; expected historical ~1.6s/update, pilot measures actual throughput',stop=['nonfinite loss/grad/parameters','CUDA/ECC increment','OOM','cache or identity mismatch','6 hour process wall cap; partial not autoextended'],policy_updates=0,rollouts=0,AE_pretrained_load=False,DINO='reuse accepted frozen pretrained DINOv3 ViT-L16 features; DINO no model update, no new extraction',geometry='accepted current-frame per-view Plucker rays; agentview then wrist',limits=['three tasks all bowl pick-place contexts, not broad operation coverage','h1..8 rather than official h2..64','not all official LIBERO suites','development already used historically','new open-middle-drawer source inventoried but not mixed in this cached pipeline'])
 save(ROOT/'precalibration_plan.json',plan)
 print(json.dumps(inventory),flush=True)

def setup():
 import ctypes,uuid
 driver=ctypes.CDLL('libcuda.so.1');assert driver.cuInit(0)==0
 dev=ctypes.c_int();assert driver.cuDeviceGet(ctypes.byref(dev),0)==0
 raw=(ctypes.c_ubyte*16)();assert driver.cuDeviceGetUuid_v2(ctypes.byref(raw),dev)==0
 assert 'GPU-'+str(uuid.UUID(bytes=bytes(raw)))==GPU
 assert torch.cuda.mem_get_info()[0]>=26*1024**3
 torch.cuda.set_per_process_memory_fraction(.30);torch.set_num_threads(4)
 torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
 seeds();sys.path.insert(0,str(OFF))
 from utils.misc import load_config
 from src import models as zoo
 cfg=load_config(str(OFF/'rebuttal/exp2_vla_adapter/configs/rla/16x64_libero.yaml'))
 # No Trainer/Extractor constructor: those load_dir/checkpoint paths are never called.
 # Explicitly reject all torch.load calls during model construction.
 old=torch.load
 def forbidden(*args,**kw):raise RuntimeError('Unexpected checkpoint load during untrained construction')
 torch.load=forbidden
 try:mods={k:getattr(zoo,cfg['models'][k]['name'])(**cfg['models'][k]['args']).cuda().train() for k in ['encoder','decoder']}
 finally:torch.load=old
 return mods,cfg

@lru_cache(maxsize=9)
def array(p):return np.load(p,mmap_mode='r')
def inputs(e,t,h,c):
 r=Path(e['feature_root']);x=array(str(r/(c+'.npy')));rays=array(str(r/'plucker.npy'))
 assert 0<=t<e['valid_chunks'] and 1<=h<=8 and t+h<e['frames']
 return [torch.from_numpy(np.array(v,copy=True)).cuda() for v in [x[t:t+1],x[t+h:t+h+1],rays[t:t+1]]]
def encode(mods,x,y,r):
 return mods['encoder']((y-x).reshape(1,392,1024),num_views=2,plucker=r)[0]
def decode(mods,x,z,r):
 return mods['decoder'](x.reshape(1,392,1024),tokens=z,num_views=2,plucker=r)[1].reshape_as(x)
def paired(mods,e,t,h):
 state=rng();x,y,r=inputs(e,t,h,'clean');zc=encode(mods,x,y,r);pc=decode(mods,x,zc,r);after=rng()
 setrng(state)
 try:
  xb,yb,rb=inputs(e,t,h,'background');assert torch.equal(r,rb)
  zb=encode(mods,xb,yb,rb);pb=decode(mods,xb,zb,rb)
 finally:setrng(after)
 rec=.5*(F.l1_loss(pc,y)+F.mse_loss(pc,y)+F.l1_loss(pb,yb)+F.mse_loss(pb,yb))
 con=F.mse_loss(zc,zb);assert torch.isfinite(rec) and torch.isfinite(con)
 return rec,con
def gradnorm(ps):return math.sqrt(sum(float(p.grad.detach().float().square().sum()) for p in ps if p.grad is not None))/256
def clean(ps):
 for p in ps:p.grad=None
def gradient_check(mods,byid,lam):
 state=rng();ps=list(mods['encoder'].parameters());allps=[p for m in mods.values() for p in m.parameters()]
 norms=[];losses=[]
 samples=json.loads((ROOT/'calibration_samples.json').read_text())
 for which in [0,1]:
  clean(allps);setrng(state);vals=[]
  for ep,t,h in samples:
   vals2=paired(mods,byid[ep],t,h);v=vals2[which];vals.append(float(v.detach()));(v*256/len(samples)).backward()
  assert all(p.grad is None or torch.isfinite(p.grad).all() for p in allps)
  norms.append(gradnorm(ps));losses.append(float(np.mean(vals)))
 clean(allps);setrng(state)
 return dict(rec=losses[0],cons=losses[1],encoder_rec_gradient=norms[0],encoder_cons_gradient=norms[1],lambda_cons=lam,weighted_encoder_cons_gradient=lam*norms[1],weighted_gradient_ratio=None if norms[0]<1e-12 else lam*norms[1]/norms[0],scope='all encoder trainable params, same6 fixed train transitions; matched RNG; no optimizer update')

def optimizer(mods):
 from src.trainers.utils import make_master_params
 ps=[p for m in mods.values() for p in m.parameters()];masters=make_master_params(ps)
 return ps,masters,torch.optim.AdamW(masters,lr=1e-4,weight_decay=0.,betas=(.9,.999),eps=1e-8)
def update(mods,ps,masters,opt,batch,byid,lam):
 from src.trainers.utils import model_grads_to_master_grads,master_params_to_model_params
 clean(ps);opt.zero_grad(set_to_none=True);rs=[];cs=[]
 for ep,t,h in batch:
  r,c=paired(mods,byid[ep],t,h);((r+lam*c)*256/len(batch)).backward();rs.append(float(r.detach()));cs.append(float(c.detach()))
 assert all(p.grad is None or torch.isfinite(p.grad).all() for p in ps),'Nonfinite gradient'
 ng=[gradnorm(list(mods[k].parameters())) for k in ['encoder','decoder']]
 model_grads_to_master_grads(ps,masters);masters[0].grad.div_(256);assert torch.isfinite(masters[0].grad).all()
 opt.step();master_params_to_model_params(ps,masters);assert torch.isfinite(masters[0]).all()
 return dict(reconstruction=float(np.mean(rs)),consistency=float(np.mean(cs)),weighted_consistency=lam*float(np.mean(cs)),total=float(np.mean(rs))+lam*float(np.mean(cs)),gradient_encoder=ng[0],gradient_decoder=ng[1])
def snapshot(out,step,mods,masters,opt,lam):
 p=out/f'step{step:04d}.pt';assert not p.exists()
 ck=dict(models={k:{n:v.detach().cpu() for n,v in m.state_dict().items()} for k,m in mods.items()},master=masters[0].detach().cpu(),optimizer=opt.state_dict(),rng=rng(),global_step=step,next_sample=step*BATCH,lambda_cons=lam,scheduler=None,AMP_scaler=None,fixed_loss_scale=256,order_sha256=sha(ROOT/'sampling_order.json'),initial_sha256=sha(ROOT/'common_untrained_step0.pt'))
 torch.save(ck,p);return p
def reload_check(path,mods,masters,opt):
 from src.trainers.utils import master_params_to_model_params
 before=digest(mods);state=rng();ck=torch.load(path,map_location='cpu',weights_only=False)
 for k,m in mods.items():m.load_state_dict(ck['models'][k],strict=True)
 masters[0].data.copy_(ck['master'].cuda());opt.load_state_dict(ck['optimizer']);master_params_to_model_params([p for m in mods.values() for p in m.parameters()],masters)
 assert before==digest(mods);setrng(ck['rng']);assert torch.equal(state['cpu'],torch.get_rng_state())
 assert torch.equal(state['cuda'][0],torch.cuda.get_rng_state())
 return dict(strict_reload=True,parameter_sha256=before,file_sha256=sha(path),global_step=ck['global_step'],optimizer_steps=sorted(set(int(v['step']) for v in opt.state.values())),rng_exact=True,scheduler=None,AMP='none; fixed loss scaling; FP32 masters restored')

@torch.no_grad()
def evaluate(mods,byid,out,step):
 state=rng();modes={k:m.training for k,m in mods.items()}
 for m in mods.values():m.eval()
 samples=json.loads((ROOT/'fixed_evaluation_samples.json').read_text());zs={}
 for i,s in enumerate(samples):
  for c in ['clean','background']:
   x,y,r=inputs(byid[s['episode']],s['t'],s['h'],c);zs[i,c]=encode(mods,x,y,r).cpu()
 rows=[]
 for i,s in enumerate(samples):
  group=[j for j,v in enumerate(samples) if v['task']==s['task'] and v['split']==s['split'] and v['h']==s['h']]
  j=group[(group.index(i)+3)%len(group)] # bijective rotation by one episode (3 anchors), same horizon
  assert samples[j]['episode']!=s['episode']
  for c in ['clean','background']:
   x,y,r=inputs(byid[s['episode']],s['t'],s['h'],c);z=zs[i,c].cuda()
   for kind,zv in [('real',z),('shuffled',zs[j,c].cuda()),('zero',torch.zeros_like(z))]:
    p=decode(mods,x,zv,r);assert torch.isfinite(p).all()
    for cam in [0,1]:
     l1=float(F.l1_loss(p[:,cam],y[:,cam]));mse=float(F.mse_loss(p[:,cam],y[:,cam]))
     rows.append(dict(**s,appearance=c,camera=['agentview','wrist'][cam],decoder_z=kind,l1=l1,mse=mse,rec=l1+mse,copy_current_l1=float(F.l1_loss(x[:,cam],y[:,cam])),copy_current_mse=float(F.mse_loss(x[:,cam],y[:,cam]))))
 csvsave(out/f'evaluation_step{step:04d}.csv',rows)
 reps=[]
 for task in SOURCES:
  for split in ['train','validation']:
   ii=[i for i,s in enumerate(samples) if s['task']==task and s['split']==split]
   a=np.stack([zs[i,'clean'].float().numpy().ravel() for i in ii]);b=np.stack([zs[i,'background'].float().numpy().ravel() for i in ii])
   na=np.linalg.norm(a,axis=1);nb=np.linalg.norm(b,axis=1);ua=a/np.maximum(na[:,None],1e-12);ub=b/np.maximum(nb[:,None],1e-12)
   for c,z in [('clean',a),('background',b)]:
    sv=np.linalg.svd(z-z.mean(0),compute_uv=False);v=sv**2;prob=v/max(float(v.sum()),1e-30);rank=0. if v.sum()<1e-20 else float(np.exp(-np.sum(prob[prob>0]*np.log(prob[prob>0]))))
    reps.append(dict(task=task,split=split,appearance=c,n=len(ii),latent_norm=float(np.linalg.norm(z,axis=1).mean()),latent_mean=float(z.mean()),mean_coordinate_variance=float(z.var(0).mean()),entropy_effective_rank=rank,rank_sample_ceiling=len(ii)-1,paired_MSE=float(np.mean((a-b)**2)),paired_unit_L2=float(np.linalg.norm(ua-ub,axis=1).mean()),cosine=float(np.mean(np.sum(ua*ub,axis=1)))))
 csvsave(out/f'representation_step{step:04d}.csv',reps)
 for k,m in mods.items():m.train(modes[k])
 setrng(state)
 print('EVAL',out.name,step,flush=True)

def run(mode,resume=None,output_name=None):
 start=time.time();baseline=ecc();mods,cfg=setup();byid={e['episode']:e for e in json.loads((ROOT/'data_manifest.json').read_text())['episodes']}
 if mode=='init':
  assert torch.count_nonzero(mods['decoder'].out_layer.weight)==0 and torch.count_nonzero(mods['decoder'].out_layer.bias)==0
  state=rng();h=digest(mods);p=ROOT/'common_untrained_step0.pt';assert not p.exists()
  torch.save(dict(models={k:{n:v.detach().cpu() for n,v in m.state_dict().items()} for k,m in mods.items()},rng=state,seed=SEED,global_step=0,origin='official constructors only; no AE weight loaded'),p)
  audit=dict(parameter_sha256=h,file_sha256=sha(p),official_constructor_only=True,torch_load_forbidden_during_construction=True,trainer_constructor_not_called=True,extractor_constructor_not_called=True,load_dir=None,auto_resume=False,decoder_output_zero_init=True,DINO='frozen pretrained cached features; not initialized or updated here',parameters={k:sum(p.numel() for p in m.parameters()) for k,m in mods.items()},config=cfg,source_hashes={str(p):sha(p) for p in [OFF/'src/models/simple_token_transformer.py',OFF/'rebuttal/src/models/multiview_token_transformer.py',OFF/'rebuttal/src/models/plucker_embedding.py']},gpu=GPU,ecc=baseline)
  save(ROOT/'initialization_audit.json',audit);print('INIT',audit['parameters'],h,flush=True);return
 ck=torch.load(ROOT/'common_untrained_step0.pt',map_location='cpu',weights_only=False)
 for k,m in mods.items():m.load_state_dict(ck['models'][k],strict=True)
 setrng(ck['rng']);initial=digest(mods);assert initial==json.loads((ROOT/'initialization_audit.json').read_text())['parameter_sha256'];del ck
 ps,masters,opt=optimizer(mods)
 order=json.loads((ROOT/'sampling_order.json').read_text());out=ROOT/(output_name or mode);out.mkdir(exist_ok=False)
 lam=0. if mode in ['pilot','C'] else json.loads((ROOT/'frozen_plan.json').read_text())['lambda_cons']
 total=60 if mode=='pilot' else UPDATES
 start_step=0
 if resume is not None:
  assert mode in ['C','D'] and output_name and output_name!=mode
  ck=torch.load(resume,map_location='cpu',weights_only=False)
  assert ck['initial_sha256']==sha(ROOT/'common_untrained_step0.pt') and ck['order_sha256']==sha(ROOT/'sampling_order.json') and ck['lambda_cons']==lam
  assert ck['fixed_loss_scale']==256 and ck['scheduler'] is None and ck['AMP_scaler'] is None
  start_step=int(ck['global_step']);assert 0<start_step<total and ck['next_sample']==start_step*BATCH
  for k,m in mods.items():m.load_state_dict(ck['models'][k],strict=True)
  masters[0].data.copy_(ck['master'].cuda());opt.load_state_dict(ck['optimizer'])
  from src.trainers.utils import master_params_to_model_params
  master_params_to_model_params(ps,masters);setrng(ck['rng']);del ck
 save(out/'start_audit.json',dict(initial_parameter_sha256=initial,common_step0_sha256=sha(ROOT/'common_untrained_step0.pt'),order_sha256=sha(ROOT/'sampling_order.json'),lambda_cons=lam,gpu=GPU,ecc=baseline,resume=None if resume is None else str(resume),restored_step=start_step,source_code_sha256=sha(Path(__file__)),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
 checks=[]
 g=gradient_check(mods,byid,lam);checks.append(dict(step=start_step,**g));save(out/f'gradient_step{start_step:04d}.json',g)
 if mode=='pilot':assert g['encoder_rec_gradient']==0,'Expected official zero output to block initial rec gradient'
 if mode!='pilot':evaluate(mods,byid,out,start_step)
 f=(out/'training_metrics.csv').open('x');w=None
 for step in range(start_step+1,total+1):
  if (ROOT/'STOP_FORMAL.json').exists():
   if step-1>start_step and not (out/f'step{step-1:04d}.pt').exists():snapshot(out,step-1,mods,masters,opt,lam)
   save(out/'stopped.json',dict(last_successful_update=step-1,reason='paired process error stop request',updates_this_process=step-1-start_step));raise SystemExit(2)
  v=update(mods,ps,masters,opt,order[(step-1)*BATCH:step*BATCH],byid,lam)
  v=dict(step=step,**v,lr=opt.param_groups[0]['lr'],seconds=time.time()-start,peak_allocated=torch.cuda.max_memory_allocated(),sample_start=(step-1)*BATCH,sample_stop=step*BATCH)
  if w is None:w=csv.DictWriter(f,fieldnames=list(v));w.writeheader()
  w.writerow(v);f.flush()
  if step%10==0:
   print('UPDATE',mode,step,v['reconstruction'],v['consistency'],round(v['seconds'],1),flush=True)
   assert ec(baseline)==ec(ecc()),'ECC changed';assert time.time()-start<21600,'Bounded six-hour process cap'
  if (mode=='pilot' and step in [10,30,60]) or (mode!='pilot' and step in CHECKS):
   path=snapshot(out,step,mods,masters,opt,lam);save(out/f'reload_step{step:04d}.json',reload_check(path,mods,masters,opt))
   g=gradient_check(mods,byid,lam);checks.append(dict(step=step,**g));save(out/f'gradient_step{step:04d}.json',g)
   if mode!='pilot':evaluate(mods,byid,out,step)
  elif mode!='pilot' and step%250==0:snapshot(out,step,mods,masters,opt,lam)
 f.close();csvsave(out/'gradient_checkpoints.csv',checks)
 if mode=='pilot':
  gs=checks[1:];assert all(x['encoder_rec_gradient']>1e-12 and x['encoder_cons_gradient']>1e-12 and x['cons']>1e-12 for x in gs)
  grad_candidates=[.1*x['encoder_rec_gradient']/x['encoder_cons_gradient'] for x in gs]
  loss_caps=[.1*x['rec']/x['cons'] for x in gs];lam=min(float(np.median(grad_candidates)),min(loss_caps));assert math.isfinite(lam) and lam>0
  plan=json.loads((ROOT/'precalibration_plan.json').read_text());plan.update(lambda_cons=lam,gradient_candidates=grad_candidates,loss_caps=loss_caps,pilot_seconds=time.time()-start,pilot_total_updates=60,formal_initialization='reload common_untrained_step0.pt; discard pilot optimizer/models/RNG',pilot_weighted_gradient_ratios=[lam*x['encoder_cons_gradient']/x['encoder_rec_gradient'] for x in gs]);save(ROOT/'frozen_plan.json',plan)
 save(out/'completion.json',dict(successful_updates=total,actual_update_computations=total-start_step,replayed_updates=0 if resume is None else 'audit prior unpersisted logs separately',seconds=time.time()-start,lambda_cons=lam,final_parameter_sha256=digest(mods),peak_allocated=torch.cuda.max_memory_allocated(),ecc_unchanged=ec(baseline)==ec(ecc()),policy_updates=0,rollouts=0))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','init','pilot','C','D']);p.add_argument('--resume-from',type=Path);p.add_argument('--output-name');args=p.parse_args()
 if args.mode=='prepare':prep()
 else:run(args.mode,args.resume_from,args.output_name)
