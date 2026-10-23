"""Bounded BC-only pilot using upstream normalization, chunking, batch and loss.
Entry point is baseline_update_smoke.py --mode formal, always fresh Prismatic.
"""
import csv,json,time,random,resource,hashlib,os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

def execute(g):
    a,o,mods,params=g['a'],g['o'],g['mods'],g['params']
    tf=g['tf']; model,head,prop=mods['vla'],mods['head'],mods['proprio']
    plan=json.loads((a.run/'training_plan.json').read_text())
    assert plan['frozen'] and plan['batch_size']==a.batch_size==8
    assert json.loads((a.run/'accepted_dataset_manifest.json').read_text())['status']=='PASS_RENDERED_POOL'
    assert json.loads((a.run/'isolated_smoke_batch8/result.json').read_text())['status']=='PASS_ISOLATED_UPDATE_RELOAD'
    optimizer=g['optimizer'];scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[plan['updates']],gamma=.1)
    rows=json.loads((a.run/'episode_manifest.json').read_text()); datasets={}; train=[]; dev=[]
    for e in rows:
        root=a.run/'episodes'/e['episode']; provenance=json.loads((root/'trajectory_provenance.json').read_text())
        assert provenance['split']==e['split'] and provenance['frames']==e['frames']
        lab=np.load(root/'trajectory_labels.npz'); actions=lab['actions'].astype(np.float32).copy();actions[:,-1]=1-np.clip(actions[:,-1],0,1)
        trajectory={'action':tf.constant(actions),'observation':{'proprio':tf.constant(lab['proprio']),'timestep':tf.range(len(actions))},'task':{'language_instruction':tf.constant([e['task']]*len(actions))},'dataset_name':tf.constant(['single_task_dual224_fk_v3']*len(actions)),'absolute_action_mask':tf.constant(np.tile([False]*6+[True],(len(actions),1)))}
        chunk=g['chunk_act_obs'](g['normalize_action_and_proprio'](trajectory,g['stats'],g['ACTION_PROPRIO_NORMALIZATION_TYPE']),window_size=1,future_action_window_size=7)
        datasets[e['episode']]=(root,tf.nest.map_structure(lambda x:x.numpy(),chunk))
        assert lab['valid_horizon_mask'][:e['complete_h8_chunks']].all()
        (train if e['split']=='train' else dev).extend((e['episode'],t) for t in range(e['complete_h8_chunks']))
    assert len(train)==plan['train_physical_windows'] and len(dev)==plan['development_physical_windows']
    rng=np.random.RandomState(plan['seed']); order=[]
    needed=plan['updates']*plan['batch_size']*plan['accumulation']//2
    while len(order)<needed: order.extend([train[i] for i in rng.permutation(len(train))])
    order=order[:needed];g['save']('sampling_order.json',order)
    def make_batch(examples):
        items=[]
        for episode,t,appearance in examples:
            root,ch=datasets[episode]; raw=tf.nest.map_structure(lambda x:x[t].copy() if isinstance(x[t],np.ndarray) else x[t],ch)
            with np.load(root/'frames'/f'{t:04}.npz') as frame:
                rgb=frame[appearance].transpose(0,2,3,1)
            raw['observation']['image_primary']=rgb[0:1];raw['observation']['image_wrist']=rgb[1:2]
            item=g['transform'](raw);pl=len(item['input_ids'])-g['NUM_TOKENS']
            item['input_ids']=torch.cat([item['input_ids'],torch.tensor([g['STOP_INDEX']])]);item['labels']=item['input_ids'].clone();item['labels'][:pl]=g['IGNORE_INDEX'];items.append(item)
        batch=g['collate'](items);batch['labels']=batch['labels'].to('cuda');batch['proprio']=batch['proprio'].to('cuda')
        assert int(model._process_action_masks(batch['labels']).sum())==len(examples)*g['NUM_TOKENS']
        assert all(torch.isfinite(x).all() for x in batch.values() if torch.is_tensor(x))
        return batch
    capture={};original=head.predict_action
    def predict(*args,**kwargs):
        x=original(*args,**kwargs);capture['pred']=x.detach();return x
    head.predict_action=predict
    def forward(batch,phase):
        return g['ft'].run_forward_pass(vla=model,action_head=SimpleNamespace(module=head),proprio_projector=prop,batch=batch,action_tokenizer=g['at'],device_id='cuda',use_l1_regression=True,use_proprio=True,use_film=False,num_patches=model.vision_backbone.get_num_patches()*2,use_pro_version=True,cfg=SimpleNamespace(phase=phase))
    def snapshot(step):
        obj=dict(parameters=g['trainable_state'](),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),global_step=step,next_physical_sample=step*8,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),initialization=str(a.reference/'resolved_prismatic_initialization.pt'),configuration=plan,scope='FORMAL_B_BASELINE')
        path=o/f'step{step:04}.pt';assert not path.exists();torch.save(obj,path)
        return path
    valfile=(o/'validation_metrics.csv').open('x'); valwriter=csv.DictWriter(valfile,fieldnames=['step','episode','timestep','appearance','action_L1','translation_L1_normalized','rotation_L1_normalized','gripper_L1','finite']);valwriter.writeheader()
    def validate(step):
        cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all();py=random.getstate();npstate=np.random.get_state()
        for m in mods.values():m.eval()
        examples=[(e,t,c) for e,t in dev for c in ('clean','background')]
        with torch.no_grad():
            for start in range(0,len(examples),8):
                part=examples[start:start+8];b=make_batch(part);loss,_=forward(b,'Inference');err=(capture['pred'].float()-b['actions'].to('cuda').float()).abs();assert torch.isfinite(err).all()
                for i,(e,t,c) in enumerate(part):valwriter.writerow(dict(step=step,episode=e,timestep=t,appearance=c,action_L1=float(err[i].mean()),translation_L1_normalized=float(err[i,:,:3].mean()),rotation_L1_normalized=float(err[i,:,3:6].mean()),gripper_L1=float(err[i,:,6].mean()),finite=True))
        valfile.flush();os.fsync(valfile.fileno())
        for m in mods.values():m.train()
        torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(cuda);random.setstate(py);np.random.set_state(npstate)
    # Loader equivalence and initial states are recorded before any formal update.
    g['save']('loader_acceptance.json',dict(status='PASS',physical_train=len(train),physical_dev=len(dev),official_normalizer=True,official_chunker=True,official_batch_transform=True,stop_repair=True,RLA_targets=False,future_images=False,tail_incomplete_excluded=True))
    snapshot(0);validate(0)
    logfile=(o/'training_metrics.csv').open('x');fields=['step','action_loss','lr','gradient_vla','gradient_head','gradient_proprio','allocated','reserved','cpu_maxrss_kib','elapsed_s','successful_updates','finite'];writer=csv.DictWriter(logfile,fieldnames=fields);writer.writeheader()
    started=time.time()
    for step in range(1,plan['updates']+1):
        optimizer.zero_grad(set_to_none=True);losses=[]
        chosen=order[(step-1)*8:step*8]; examples=[(e,t,c) for e,t in chosen for c in ('clean','background')]
        for offset in (0,8):
            b=make_batch(examples[offset:offset+8]);loss,_=forward(b,'Training');assert torch.isfinite(loss);losses.append(float(loss.detach()));(loss/2).backward()
        norms={}
        for k,m in mods.items():
            grads=[x.grad.detach().float() for x in m.parameters() if x.grad is not None]
            assert all(torch.isfinite(x).all() for x in grads)
            norms[k]=float(torch.stack([x.norm().square() for x in grads]).sum().sqrt())
        del grads
        lr=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step()
        assert all(torch.isfinite(x).all() for x in params)
        writer.writerow(dict(step=step,action_loss=np.mean(losses),lr=lr,gradient_vla=norms['vla'],gradient_head=norms['head'],gradient_proprio=norms['proprio'],allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved(),cpu_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,elapsed_s=time.time()-started,successful_updates=step,finite=True));logfile.flush()
        print('UPDATE',step,'/',plan['updates'],'loss',np.mean(losses),'elapsed',time.time()-started,flush=True)
        if step in plan['checkpoint_steps']:
            snapshot(step);validate(step)
            assert g['counts'](g['before'])==g['counts'](g['ecc']())
    logfile.close();valfile.close()
    final=torch.load(o/f"step{plan['updates']:04}.pt",map_location='cpu')
    assert final['global_step']==plan['updates']
    for k,m in mods.items():
        named={n:x for n,x in m.named_parameters() if x.requires_grad};assert named.keys()==final['parameters'][k].keys()
        for n,x in named.items():assert torch.equal(x.detach().cpu(),final['parameters'][k][n])
    g['save']('training_acceptance.json',dict(status='PASS_TRAINING_NOT_METHOD',successful_updates=plan['updates'],physical_exposures=needed,appearance_observation_exposures=needed*2,camera_image_exposures=needed*4,independent_train_episodes=4,independent_development_episodes=1,heldout_read=False,RLA=False,elapsed_s=time.time()-started,checkpoint_reload_parameter_equality=True,new_closed_loop=0))
