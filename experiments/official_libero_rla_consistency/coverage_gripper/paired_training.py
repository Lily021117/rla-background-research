"""Fixed-compute B4/B20 experiment; official batch and action loss unchanged."""
import json,csv,random,time,os,hashlib,resource
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from coverage_gripper.dataset import DerivedEpisodeDataset

def execute(g):
    a,o=g['a'],g['o'];plan=json.loads((a.run/'training_plan.json').read_text());assert plan['frozen']
    mods=g['mods'];model,head,prop=mods['vla'],mods['head'],mods['proprio'];opt=g['optimizer']
    ds=DerivedEpisodeDataset(plan['manifest'],plan['statistics'],g['transform'],g['collate'])
    train=[(e,t) for e,(_,_,meta) in ds.episodes.items() if e in plan['train_episodes'] for t in range(meta['valid_chunks'])]
    common=[(e,t,c) for e,(_,_,m) in ds.episodes.items() if e in plan['common_B4'] for t in range(m['valid_chunks']) for c in ('clean','background')]
    dev=[(e,t,c) for e,(_,_,m) in ds.episodes.items() if m['split']=='validation' for t in range(m['valid_chunks']) for c in ('clean','background')]
    rng=np.random.RandomState(plan['seed']);order=[]
    while len(order)<plan['updates']*8:order.extend(train[i] for i in rng.permutation(len(train)))
    order=order[:plan['updates']*8];g['save']('sampling_order.json',order)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[plan['updates']],gamma=.1)
    capture={};original=head.predict_action
    def predict(*args,**kwargs):
        p=original(*args,**kwargs);capture['pred']=p.detach();return p
    head.predict_action=predict
    def batch(part):
        b=ds.batch(part);b['labels']=b['labels'].to('cuda');b['proprio']=b['proprio'].to('cuda');assert int(model._process_action_masks(b['labels']).sum())==len(part)*g['NUM_TOKENS'];return b
    def forward(b,phase):
        return g['ft'].run_forward_pass(vla=model,action_head=SimpleNamespace(module=head),proprio_projector=prop,batch=b,action_tokenizer=g['at'],device_id='cuda',use_l1_regression=True,use_proprio=True,use_film=False,num_patches=model.vision_backbone.get_num_patches()*2,use_pro_version=True,cfg=SimpleNamespace(phase=phase))
    def snapshot(step):
        state=g['trainable_state']();h=hashlib.sha256()
        for group in sorted(state):
            for n,x in sorted(state[group].items()):h.update((group+n).encode());h.update(x.view(torch.uint8).numpy().tobytes())
        path=o/f'step{step:04}.pt';assert not path.exists()
        torch.save(dict(parameters=state,optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),global_step=step,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),next_physical_sample=step*8,configuration=plan,scope='PAIRED_B4_B20'),path)
        g['save'](f'checkpoint_step{step:04}_audit.json',dict(parameter_hash=h.hexdigest(),global_step=step,path=str(path)))
    vf=(o/'validation_metrics.csv').open('x');vw=csv.DictWriter(vf,fieldnames=['step','set','episode','timestep','appearance','position','action_L1','translation_L1','rotation_L1','gripper_L1','gripper_pred','gripper_target']);vw.writeheader()
    def validate(step):
        states=(torch.get_rng_state(),torch.cuda.get_rng_state_all(),random.getstate(),np.random.get_state())
        for m in mods.values():m.eval()
        for name,examples in [('development',dev),('common_B4',common),('own_train',[(e,t,c) for e,t in train[::max(1,len(train)//100)] for c in ('clean','background')])]:
            predictions=[];targets=[]
            with torch.inference_mode():
                for start in range(0,len(examples),8):
                    part=examples[start:start+8];b=batch(part);loss,_=forward(b,'Inference');pr=capture['pred'].float().cpu().numpy();gt=b['actions'].numpy();assert np.isfinite(pr).all() and torch.isfinite(loss);predictions.append(pr);targets.append(gt)
                    for i,(e,t,c) in enumerate(part):
                        err=abs(pr[i]-gt[i])
                        for pos in range(8):vw.writerow(dict(step=step,set=name,episode=e,timestep=t,appearance=c,position=pos,action_L1=float(err[pos].mean()),translation_L1=float(err[pos,:3].mean()),rotation_L1=float(err[pos,3:6].mean()),gripper_L1=float(err[pos,6]),gripper_pred=float(pr[i,pos,6]),gripper_target=float(gt[i,pos,6])))
            np.savez_compressed(o/f'{name}_step{step:04}.npz',prediction=np.concatenate(predictions),target=np.concatenate(targets))
            if step==0:g['save'](f'{name}_samples.json',examples)
        vf.flush();os.fsync(vf.fileno())
        for m in mods.values():m.train()
        torch.set_rng_state(states[0]);torch.cuda.set_rng_state_all(states[1]);random.setstate(states[2]);np.random.set_state(states[3])
    snapshot(0);validate(0)
    f=(o/'training_metrics.csv').open('x');w=csv.DictWriter(f,fieldnames=['step','loss','lr','gradient_vla','gradient_head','gradient_proprio','allocated','reserved','cpu_maxrss_kib','seconds','successful_updates']);w.writeheader();start=time.time()
    for step in range(1,plan['updates']+1):
        opt.zero_grad(set_to_none=True);examples=[(e,t,c) for e,t in order[(step-1)*8:step*8] for c in ('clean','background')];losses=[]
        for offset in (0,8):
            b=batch(examples[offset:offset+8]);loss,_=forward(b,'Training');assert torch.isfinite(loss);losses.append(float(loss.detach()));(loss/2).backward()
        norms={k:float(torch.stack([p.grad.detach().float().norm().square() for p in m.parameters() if p.grad is not None]).sum().sqrt()) for k,m in mods.items()};assert all(np.isfinite(v) for v in norms.values())
        lr=opt.param_groups[0]['lr'];opt.step();scheduler.step();assert all(torch.isfinite(p).all() for p in g['params'])
        w.writerow(dict(step=step,loss=np.mean(losses),lr=lr,gradient_vla=norms['vla'],gradient_head=norms['head'],gradient_proprio=norms['proprio'],allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved(),cpu_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,seconds=time.time()-start,successful_updates=step));f.flush()
        if step%25==0:print('UPDATE',plan['arm'],step,'loss',np.mean(losses),flush=True)
        if step in plan['checkpoint_steps']:
            snapshot(step);validate(step);assert g['counts'](g['before'])==g['counts'](g['ecc']())
    f.close();vf.close();ck=torch.load(o/f"step{plan['updates']:04}.pt",map_location='cpu')
    for k,m in mods.items():
        full=m.state_dict();full.update(ck['parameters'][k]);m.load_state_dict(full,strict=True)
        for n,p in m.named_parameters():
            if p.requires_grad:assert torch.equal(p.cpu(),ck['parameters'][k][n])
    g['save']('training_acceptance.json',dict(status='PASS',successful_updates=plan['updates'],strict_final_reload=True,ECC_increment=False,physical_exposures=len(order),appearance_exposures=len(order)*2))
