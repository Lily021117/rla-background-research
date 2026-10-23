"""Official paired reconstruction plus the sole latent MSE extension."""
import argparse,time,csv,math,resource
from common import *
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--arm',choices=['calibrate','C','D'],required=True);p.add_argument('--resume-from',type=Path);p.add_argument('--output-name');a=p.parse_args();before=ecc();ext,init=load(a.project)
ext.dino_extractor.cpu();del ext.dino_extractor;torch.cuda.empty_cache()
mods=ext.models
for m in mods.values():m.train().requires_grad_(True)
params=[p for m in mods.values() for p in m.parameters()];train=[e for e in entries(a.run) if e['split']=='train']
def losses(e,t,h):
    clean=pair(a.run,e,t,h,'clean');bg=pair(a.run,e,t,h,'background');cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all()
    zc,pc=forward(ext,mods,*clean);after_cpu=torch.get_rng_state();after_cuda=torch.cuda.get_rng_state_all();torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(cuda)
    try:zb,pb=forward(ext,mods,*bg)
    finally:torch.set_rng_state(after_cpu);torch.cuda.set_rng_state_all(after_cuda)
    yc,yb=clean[1],bg[1]
    rc=F.l1_loss(pc,yc)+F.mse_loss(pc,yc);rb=F.l1_loss(pb,yb)+F.mse_loss(pb,yb)
    return .5*(rc+rb),F.mse_loss(zc,zb),rc,rb
def gradnorm(ps,scale=1):return math.sqrt(sum(float(p.grad.detach().float().norm().square()) for p in ps if p.grad is not None))/scale
if a.arm=='calibrate':
    first=train[0];fixed=[(0,1),(first['frames']//3,2),(first['frames']//2,4),(first['frames']-9,8)];rec=0;cons=0;start=time.time()
    for t,h in fixed:
        r,c,_,_=losses(first,t,h);rec=rec+r/4;cons=cons+c/4
    (rec*256).backward(retain_graph=True);rnorm=gradnorm(params,256)
    for p in params:p.grad=None
    (cons*256).backward();cnorm=gradnorm(params,256)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
    lam=min(.01,.1*float(rec.detach())/max(float(cons.detach()),1e-8));assert lam>0
    plan=dict(frozen=True,seed=20261023,updates=600,microbatch_physical=1,accumulation=8,effective_physical_batch=8,effective_appearance_batch=16,physical_exposures=4800,appearance_exposures=9600,train_window_equivalent_epochs=4800/sum(e['valid_chunks'] for e in train),all_horizon_pool_epochs=4800/(8*sum(e['valid_chunks'] for e in train)),horizons=list(range(1,9)),horizon_sampling='seeded permutation of (episode,valid_anchor,h1..8); identical C/D sequence',optimizer='AdamW',lr=1e-4,weight_decay=0.,betas=[.9,.999],eps=1e-8,master_parameters='official inflat_all utility, FP32 master over official mixed-FP16 model',loss_scale=256.,loss_scale_growth=0.,loss_scale_deviation='fixed conservative2^8 rather than official initial2^20 dynamic growth; train-only finite-gradient calibration; identical arms',lambda_cons=lam,lambda_selection='min(0.01,0.1*train_batch_reconstruction/train_batch_raw_consistency), once, no development results',reconstruction='0.5*((L1+MSE)(clean_future_pred,clean_future_DINO)+(L1+MSE)(background_future_pred,background_future_DINO)); mean across B/views/patches/channels',consistency='mean((z_clean-z_background)^2) over B/16queries/64dims',freeze=['DINO','image_decoder_not_loaded'],unfreeze=['all official encoder','all matching decoder'],checkpoint_steps=[0,300,600],validation_steps=[0,600],final_checkpoint=600,initial_step=90000,reference='frozen step90000, not paperstep40000',budget_reason='same4800 physical-transition exposure scale as B20 pilot; resource-bounded600x8 vs official128 batch/1M max; not convergence claim',horizon_deviation='paired pilot h1..8 instead of official training h2..64; matches policy target support',stop_conditions=['nonfinite gradients/loss','OOM','ECC increment','identity mismatch'])
    save(a.run/'training_plan.json',plan);save(a.run/'consistency_scale_calibration.json',dict(episode=first['episode'],fixed_samples=fixed,reconstruction=float(rec.detach()),raw_consistency=float(cons.detach()),lambda_cons=lam,weighted_consistency=lam*float(cons.detach()),reconstruction_gradient_norm=rnorm,raw_consistency_gradient_norm=cnorm,weighted_consistency_gradient_norm=lam*cnorm,weighted_gradient_ratio=lam*cnorm/max(rnorm,1e-20),optimizer_updates=0,seconds=time.time()-start,peak_allocated=torch.cuda.max_memory_allocated()))
    print('CALIBRATION',plan,flush=True);raise SystemExit
plan=json.loads((a.run/'training_plan.json').read_text());assert plan['frozen'];out=a.run/(a.output_name or ('AE_'+a.arm));out.mkdir(exist_ok=False)
from src.trainers.utils import make_master_params,model_grads_to_master_grads,master_params_to_model_params,zero_grad
masters=make_master_params(params);opt=torch.optim.AdamW(masters,lr=plan['lr'],weight_decay=0.);lam=plan['lambda_cons'] if a.arm=='D' else 0.;seed()
pool=[(e['episode'],t,h) for e in train for t in range(e['valid_chunks']) for h in range(1,9)];rng=np.random.RandomState(plan['seed']);order=[pool[i] for i in rng.permutation(len(pool))[:4800]];save(out/'sampling_order.json',order);byid={e['episode']:e for e in train}
start_step=0
if a.resume_from:
    assert a.arm=='D' and a.output_name
    ck=torch.load(a.resume_from,map_location='cpu',weights_only=False);start_step=ck['global_step'];assert start_step==300 and ck['plan']==plan and ck['lambda_cons']==lam
    for k,m in mods.items():m.load_state_dict(torch.load(a.resume_from.parent/f'{k}_step{start_step:04}.pt',map_location='cuda',weights_only=True),strict=True)
    masters[0].data.copy_(ck['master'].cuda());opt.load_state_dict(ck['optimizer']);master_params_to_model_params(params,masters)
    torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng'])
    assert order==[tuple(x) for x in json.loads((a.resume_from.parent/'sampling_order.json').read_text())]
    save(out/'recovery_provenance.json',dict(resume=str(a.resume_from),restored_step=start_step,restored_master_and_optimizer_and_rng=True,original_logged_successful_updates=380,replayed_unpersisted_successful_updates=80,remaining_persisted_trajectory_updates=300,final_effective_updates=600,budget_unchanged=True,GPU_UUID=os.environ['RLA_GPU_UUID'],minimum_free_GiB=float(os.environ.get('RLA_MIN_FREE_GIB','26'))));del ck
def snapshot(step):
    for k,m in mods.items():torch.save(m.state_dict(),out/f'{k}_step{step:04}.pt')
    torch.save(dict(master=masters[0].detach().cpu(),optimizer=opt.state_dict(),global_step=step,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),plan=plan,lambda_cons=lam,next_sample=step*8),out/f'resume_step{step:04}.pt')
snapshot(start_step);f=(out/'training_metrics.csv').open('x');w=csv.DictWriter(f,fieldnames=['step','reconstruction','consistency','weighted_consistency','total','gradient_encoder','gradient_decoder','lr','seconds','allocated','reserved','cpu_maxrss_kib']);w.writeheader();start=time.time()
for step in range(start_step+1,601):
    zero_grad(params);opt.zero_grad(set_to_none=True);rs=[];cs=[]
    for ep,t,h in order[(step-1)*8:step*8]:
        rec,con,_,_=losses(byid[ep],t,h);loss=(rec+lam*con)/8;assert torch.isfinite(loss);(loss*256).backward();rs.append(float(rec.detach()));cs.append(float(con.detach()))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    ne=gradnorm(list(mods['encoder'].parameters()),256);nd=gradnorm(list(mods['decoder'].parameters()),256)
    model_grads_to_master_grads(params,masters);masters[0].grad.div_(256);assert torch.isfinite(masters[0].grad).all();opt.step();master_params_to_model_params(params,masters);assert torch.isfinite(masters[0]).all()
    w.writerow(dict(step=step,reconstruction=np.mean(rs),consistency=np.mean(cs),weighted_consistency=lam*np.mean(cs),total=np.mean(rs)+lam*np.mean(cs),gradient_encoder=ne,gradient_decoder=nd,lr=1e-4,seconds=time.time()-start,allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved(),cpu_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));f.flush()
    if step%25==0:print('AE',a.arm,step,np.mean(rs),np.mean(cs),time.time()-start,flush=True);assert counters(before)==counters(ecc())
    if step in (300,600):snapshot(step)
f.close();assert counters(before)==counters(ecc())
for k,m in mods.items():m.load_state_dict(torch.load(out/f'{k}_step0600.pt',map_location='cuda',weights_only=True),strict=True)
save(out/'acceptance.json',dict(status='PASS_TRAINING_NOT_METHOD',successful_updates=600,lambda_cons=lam,strict_matching_encoder_decoder_reload=True,ECC_increment=False,peak_allocated=torch.cuda.max_memory_allocated(),seconds=time.time()-start))
if a.resume_from:save(a.run/'active_models.json',dict(C=str(a.run/'AE_C'),D=str(out)))
