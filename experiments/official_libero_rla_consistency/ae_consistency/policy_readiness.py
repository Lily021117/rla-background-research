"""Official RLA overlay with common B20 step0 weights; optimizer updates=0."""
import json,sys,random,hashlib,dataclasses
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F

def execute(g):
    a,o=g['a'],g['o'];run=a.run.parent;project=Path(json.loads((a.run/'readiness_inputs.json').read_text())['project']);reference=Path(json.loads((a.run/'readiness_inputs.json').read_text())['B20_step0']);checkpoint=torch.load(reference,map_location='cpu');assert checkpoint['global_step']==0
    torch.cuda.set_per_process_memory_fraction(.25)
    for k,m in g['mods'].items():
        names={n for n,p in m.named_parameters() if p.requires_grad};assert names==set(checkpoint['parameters'][k]);state=m.state_dict();state.update(checkpoint['parameters'][k]);m.load_state_dict(state,strict=True)
    base_head=g['head'];model=g['model'];prop=g['prop'];overlay=project/'worktrees/rla-official-0fc6e97/rebuttal/exp2_vla_adapter/vla_adapter_overlay';sys.path.insert(0,str(overlay))
    from rla.action_head import RlaL1RegressionActionHead
    from rla.sidecar import RlaStore
    from rla.data import RlaCollator
    import rla.loss as official_loss
    from coverage_gripper.dataset import DerivedEpisodeDataset
    # New-module construction cannot consume the baseline training RNG.
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(20261024);torch.cuda.manual_seed_all(20261024);head=RlaL1RegressionActionHead(input_dim=model.llm_dim,hidden_dim=model.llm_dim,action_dim=7,use_pro_version=True).to('cuda',dtype=torch.bfloat16)
    src=base_head.state_dict();dst=head.state_dict();copied=[];expanded=[]
    for k,v in src.items():
        assert k in dst
        if dst[k].shape==v.shape:dst[k]=v;copied.append(k)
        elif k.endswith('gating_factor') and v.numel()==1:dst[k]=v.expand_as(dst[k]).clone();expanded.append(k)
        else:raise ValueError(('unexplained head mismatch',k,v.shape,dst[k].shape))
    head.load_state_dict(dst,strict=True);g['mods']['head']=head;g['head']=head;g['params']=[p for m in g['mods'].values() for p in m.parameters() if p.requires_grad];del g['optimizer'];del base_head,src,dst
    def restore_rng():
        torch.set_rng_state(checkpoint['torch_rng']);torch.cuda.set_rng_state_all(checkpoint['cuda_rng']);random.setstate(checkpoint['python_rng']);np.random.set_state(checkpoint['numpy_rng'])
    restore_rng();mods=g['mods']
    def digest():
        h=hashlib.sha256()
        for k,m in mods.items():
            for n,p in m.named_parameters():
                if p.requires_grad:h.update((k+n).encode());h.update(p.detach().cpu().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()
    initial_hash=digest();state=g['trainable_state']();torch.save(dict(parameters=state,torch_rng=checkpoint['torch_rng'],cuda_rng=checkpoint['cuda_rng'],python_rng=checkpoint['python_rng'],numpy_rng=checkpoint['numpy_rng'],global_step=0,base_step0=str(reference),scope='COMMON_OFFICIAL_RLA_C_D_ZERO_UPDATES'),o/'common_policy_step0.pt');del state
    statistics=run.parent/'run_20261023_04/normalization_statistics_expanded_train_only.json';ds=DerivedEpisodeDataset(run/'data_manifest.json',statistics,g['transform'],g['collate']);e=next(e for e in ds.manifest['episodes'] if e['split']=='train');t=e['valid_chunks']//2;raw=np.load(Path(e['root'])/'trajectory_labels.npz')['proprio'];collator=RlaCollator(2048,g['tokenizer'].pad_token_id);batches={};target_stats={}
    for arm in ('C','D'):
        restore_rng()
        items=[]
        for c in ('clean','background'):
            store=RlaStore(str(run/'sidecars'/arm/c),chunk=8,num_tokens=16,token_dim=64,norm='center');slot=store.slot(raw);assert slot>=0;item=ds.sample(e['episode'],t,c);item['rla']=store.frame(slot,t);assert item['rla'].shape==(8,1024);items.append(item)
        b=collator(items);b['labels']=b['labels'].cuda();b['proprio']=b['proprio'].cuda();assert int(model._process_action_masks(b['labels']).sum())==2*g['NUM_TOKENS'];batches[arm]=b
        z=b['rla'].float();target_stats[arm]=dict(norm=float(z.norm()),mean=float(z.mean()),variance=float(z.var()))
    for k in batches['C']:
        if k!='rla' and torch.is_tensor(batches['C'][k]):assert torch.equal(batches['C'][k],batches['D'][k]),k
    capture={};predict=head.predict_action;pop=head.pop_z_hat
    def predict_capture(*args,**kwargs):
        x=predict(*args,**kwargs);capture['action']=x.detach().clone();return x
    def pop_capture():
        z=pop();capture['z']=z;return z
    head.predict_action=predict_capture;head.pop_z_hat=pop_capture
    def call(b,wrapped=False):
        for m in mods.values():m.train();m.zero_grad(set_to_none=True)
        capture.clear();restore_rng();fn=official_loss.make_run_forward_pass(g['ft'].run_forward_pass) if wrapped else g['ft'].run_forward_pass
        result=fn(vla=model,action_head=SimpleNamespace(module=head),proprio_projector=prop,batch=b,action_tokenizer=g['at'],device_id='cuda',use_l1_regression=True,use_proprio=True,use_film=False,num_patches=model.vision_backbone.get_num_patches()*2,use_pro_version=True,cfg=SimpleNamespace(phase='Training'))
        if not wrapped:capture['z']=head.pop_z_hat()
        assert capture['action'].shape==(2,8,7) and capture['z'].shape==(2,8,1024);assert torch.isfinite(result[0]) and torch.isfinite(capture['z']).all();return result[0]
    gradients=[];outputs=[];action_losses=[]
    for arm in ('C','D'):
        loss=call(batches[arm]);action_losses.append(float(loss.detach()));outputs.append((capture['action'].cpu(),capture['z'].detach().cpu()));loss.backward();gradients.append({k+n:p.grad.detach().cpu().clone() for k,m in mods.items() for n,p in m.named_parameters() if p.grad is not None});capture.clear()
    assert torch.equal(outputs[0][0],outputs[1][0]) and torch.equal(outputs[0][1],outputs[1][1]);assert gradients[0].keys()==gradients[1].keys();sq=sum(float((gradients[0][k].float()-gradients[1][k].float()).square().sum()) for k in gradients[0]);base=sum(float(v.float().square().sum()) for v in gradients[0].values());relative=(sq/max(base,1e-30))**.5;assert relative<=1e-6
    del gradients,outputs
    act=call(batches['C']);aux=F.l1_loss(capture['z'],batches['C']['rla'].cuda().to(capture['z'].dtype));lam=.25*float(act.detach())/max(float(aux.detach()),1e-6);assert np.isfinite(lam) and lam>0
    official_loss.CFG=dataclasses.replace(official_loss.CFG,lambda_auto=False,lambda_rla=lam);capture.clear();results=[]
    for arm in ('C','D'):
        total=call(batches[arm],wrapped=True);z=capture['z'];aux=F.l1_loss(z,batches[arm]['rla'].cuda().to(z.dtype));(lam*aux).backward()
        norms={k:sum(float(p.grad.detach().float().norm().square()) for p in m.parameters() if p.grad is not None)**.5 for k,m in mods.items()};assert norms['vla']>0 and all(np.isfinite(x) for x in norms.values());results.append(dict(arm=arm,action_l1=official_loss.last_metrics()['action_l1'],auxiliary_l1=float(aux.detach()),lambda_aux=lam,weighted_auxiliary=lam*float(aux.detach()),auxiliary_gradient_norms=norms));capture.clear()
    # Future/action targets never condition the prediction.
    call(batches['C'],wrapped=True);prior=capture['action'].clone();changed=dict(batches['C'],rla=torch.zeros_like(batches['C']['rla']),actions=torch.zeros_like(batches['C']['actions']));call(changed,wrapped=True);assert torch.equal(prior,capture['action']);capture.clear()
    restored=torch.load(o/'common_policy_step0.pt',map_location='cpu')
    for k,m in mods.items():full=m.state_dict();full.update(restored['parameters'][k]);m.load_state_dict(full,strict=True)
    restore_rng();assert digest()==initial_hash
    g['save']('policy_initialization_acceptance.json',dict(status='PASS_ZERO_UPDATES',base_step0=str(reference),common_parameters_exact=True,new_parameters_common_seed=20261024,new_head_creation_preserves_baseline_RNG=True,shared_head_keys_copied=len(copied),official_gate_shape_expansions=expanded,C_D_outputs_equal=True,action_only_gradient_relative_difference=relative,auxiliary_gradient_reaches_shared_backbone=True,target_isolation_action_difference=0.,new_RLA_structure_not_claimed_equal_to_B_baseline_outputs=True,parameter_hash_before_after=initial_hash,strict_save_reload=True,lambda_aux=lam,lambda_auto_formula='0.25*detached(actionL1)/clamp(detached(auxL1),1e-6), normally every update',choice='once calibrated on C fixed training batch; exact fixed lambda and common C-based target normalization for C/D',target_statistics=target_stats,results=results,policy_optimizer_updates=0,policy_rollouts=0,peak_allocated=torch.cuda.max_memory_allocated()))
    for arm in ('C','D'):g['save'](f'policy_config_{arm}.json',dict(arm=arm,updates_next_turn_only=600,init=str(o/'common_policy_step0.pt'),target_roots={c:str(run/'sidecars'/arm/c) for c in ('clean','background')},lambda_aux=lam,action_weight=1.,same_B20_data=True,seed=20261023,lr=2e-4,optimizer='AdamW',weight_decay=.01,batch=8,accumulation=2,backend='Math SDPA',STOP='inference-aligned',norm='common20train action/proprio; common AE-C training-only RLA center',checkpoint_steps=[0,100,300,600],RLA_shape=[8,16,64],stage='joint action+auxiliary from commonstep0; no RLA-only stage1, explicit deviation to match B20 budget',optimizer_updates_this_run=0))
