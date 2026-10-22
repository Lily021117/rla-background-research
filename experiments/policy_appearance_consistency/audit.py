"""CPU-only fixed real-pair audit. No optimizer, checkpoint writes or rollouts."""
import os, json, csv, hashlib, subprocess, random, time, resource
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from method import paired_forward, predict_normalized, objective, rng_state, restore_rng
from policies.policy.vla_bc_policy_rla_unified import VLABCPolicyRLAUnified
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.normalize_util import get_image_range_normalizer

R=Path(os.environ['RLA_PROJECT_ROOT']); O=Path(os.environ['RUN_DIR'])
W=R/'worktrees/rla-official-0fc6e97'
DATA=R/'experiments/rla_background_invariance/step4_4_1_dataset_expansion/run_20261021_02'
COMMIT='0fc6e97212442ebccffa06f1d42f5481e5f22950'
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for x in iter(lambda:f.read(8388608),b''): h.update(x)
    return h.hexdigest()
def dump(name,x):
    with (O/name).open('x') as f: json.dump(x,f,indent=2,allow_nan=False)
def rows(p): return list(csv.DictReader(open(p)))
def git(p):
    return {k:subprocess.check_output(['git','-C',str(p),*v],text=True) for k,v in
            {'head':['rev-parse','HEAD'],'branch':['branch','--show-current'],'status':['status','--short'],'remotes':['remote','-v']}.items()}
def parameter_hash(m):
    h=hashlib.sha256()
    for n,p in m.state_dict().items(): h.update(n.encode()); h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()

start=time.time(); torch.set_num_threads(4); torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
assert not torch.cuda.is_available()
torch.manual_seed(42); np.random.seed(42); random.seed(42)
before_git={str(p):git(p) for p in [R,W]}
assert before_git[str(W)]['head'].strip()==COMMIT
assert not before_git[str(W)]['status']
protected={}
for prefix in ['policies','src','utils','docs']:
    for p in (W/prefix).rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts: protected[str(p)]=sha(p)
for n in ['transition_manifest.csv','sample_manifest.csv','episode_manifest.csv','render_config.json','delivery_verification.json','split_integrity.json']:
    p=DATA/n; protected[str(p)]=sha(p)
dump('protection_before.json',protected);dump('git_before.json',before_git)
trans=[x for x in rows(DATA/'transition_manifest.csv') if x['valid']=='True']
assert len(trans)==220 and sum(x['split']=='train' for x in trans)==190
assert all(x['split'] in ['train','validation'] for x in trans)
train_ids={x['episode'] for x in trans if x['split']=='train'}
val_ids={x['episode'] for x in trans if x['split']=='validation'}
assert len(train_ids)==19 and len(val_ids)==3 and not train_ids&val_ids
samples=rows(DATA/'sample_manifest.csv')
index={(x['episode'],int(x['source_state_id']),x['camera']):x for x in samples}
fixed=[]; seen=set()
for split,limit in [('train',2),('validation',1)]:
    for t in trans:
        if t['split']!=split or t['episode'] in seen: continue
        s=index[t['episode'],int(t['source_start']),'agentview']
        fixed.append({'transition':t,'sample':s}); seen.add(t['episode'])
        if sum(x['transition']['split']==split for x in fixed)==limit: break
dump('fixed_sample_manifest.json',{'selection':'first 2 distinct training episodes and first validation episode in frozen order, one current-state agentview pair each; no performance-based selection','samples':fixed})

# Only current-state arrays; no target/feature sidecar, future RGB or held-out data.
states=[];actions=[];coverage=[]
for t in trans:
    s=index[t['episode'],int(t['source_start']),'agentview']
    assert all(s[k]=='True' for k in ['geometry_check','segmentation_equal','depth_equal','finite'])
    assert int(t['source_future'])-int(t['source_start'])==8
    assert int(t['action_start_inclusive'])==int(t['source_start'])
    assert int(t['action_stop_exclusive'])==int(t['source_future'])
    a=np.asarray(json.loads(t['actions']),dtype=np.float32); assert a.shape==(8,7)
    assert sha(s['array'])==s['array_SHA256']
    with np.load(s['array'],allow_pickle=False) as z:
        state=z['fk_proprio'].astype(np.float32)
        assert state.shape==(8,) and np.isfinite(state).all()
        assert np.allclose(a[0],z['action'],rtol=0,atol=1e-7)
    if t['split']=='train': states.append(state);actions.append(a)
    coverage.append({'episode':t['episode'],'split':t['split'],'transition_id':t['transition_id'],'time_pair_pass':True,'same_label_by_construction':True})
normalizer=LinearNormalizer();normalizer.fit({'action':np.concatenate(actions),'state':np.stack(states)},mode='limits',last_n_dims=1)
normalizer['image']=get_image_range_normalizer()
dump('normalization_train_only.json',{n:v.tolist() for n,v in normalizer.state_dict().items()})
dump('pairing_coverage.json',{'transitions':coverage,'train_episodes':19,'validation_episodes':3,'heldout_read':0,'fit_transitions':190,'fit_appearance_count':1})
config=dict(action_dim=7,state_dim=8,num_cameras=1,horizon=12,n_obs_steps=1,n_action_steps=8,
            per_camera_dim=512,use_group_norm=True,share_backbone=False,img_size=128,
            resize_shape=(128,128),crop_shape=None,random_crop=True,imagenet_norm=False,
            state_hidden_dim=256,state_output_dim=256,spatial_hidden_dim=256,
            spatial_num_blocks=6,spatial_num_heads=8,spatial_mlp_ratio=4.,
            num_output_queries=16,shared_mlp_dim=2048,mlp_dropout=.1,loss_type='mse',use_latent_head=False)
model=VLABCPolicyRLAUnified(**config);model.set_normalizer(normalizer);model.train()
assert model.latent_encoder is model.latent_head is model.dino_extractor is None
assert not any('latent_head' in n for n,_ in model.named_parameters())
initial_hash=parameter_hash(model)
base_cfg={'reference_commit':COMMIT,'model':config,'init':'random: torchvision resnet18(weights=None), all other policy layers default initialization','seed':42,
    'optimizer':{'name':'AdamW','lr':3e-4,'betas':[.9,.999],'eps':1e-8,'weight_decay':1e-4},
    'scheduler':{'name':'cosine','warmup_steps':200},'paired_rng':'matched Python/NumPy/torch RNG per branch; no EMA teacher or detach',
    'batch_physical_pairs':4,'successful_updates_proposed':100,'run_training':False,'use_ema_proposed':True,
    'normalization':'official limits [-1,1] fitted on 190 train windows only; no Spatial-Pro stats',
    'active_action_mask':[True]*8+[False]*4,'camera':'agentview','language_input':False,
    'multi_task_training_allowed':False,'scope':'per-task only; architecture lacks language conditioning',
    'reference_differences':['LIBERO OSC 7D instead of ManiSkill joint-position target','FK8 instead of joint-qpos robot-specific state','agentview instead of front_lower_camera','128px existing render instead of 512px source resized128','background retained rather than foreground masked','8 verified actions supervised in unchanged12-output horizon; last4 excluded','small fixed budget and paired batch4 vs reference batch64','train-only normalizer fit avoids using validation episodes']}
for arm in ['P0','P1','P2']:
    cfg=dict(base_cfg,arm=arm,lambda_cons=.1 if arm=='P2' else 0.,
             training_appearances=['clean'] if arm=='P0' else ['clean','background'],
             image_exposures_per_update=4 if arm=='P0' else 8)
    dump('config_'+arm+'.json',cfg)
params=[p for p in model.parameters() if p.requires_grad]
results=[];backwards=0
for j,item in enumerate(fixed):
    t,s=item['transition'],item['sample']
    for condition in ['clean','background']: assert sha(s[condition])==s[condition+'_SHA256']
    with np.load(s['array'],allow_pickle=False) as z: st=torch.tensor(z['fk_proprio'].astype(np.float32))[None,None]
    obs=[]
    for condition in ['clean','background']:
        rgb=np.asarray(Image.open(s[condition]).convert('RGB')).copy()
        image=torch.from_numpy(rgb).permute(2,0,1).float()[None,None,None]/255.
        obs.append(normalizer.normalize({'image':image,'state':st}))
    target=torch.zeros(1,12,7)
    target[:,:8]=normalizer['action'].normalize(torch.tensor(json.loads(t['actions']),dtype=torch.float32)[None])
    mask=torch.tensor([[True]*8+[False]*4]); beginning=rng_state()
    def grad(arm,weight,cons_only=False):
        nonlocal_placeholder=None
        restore_rng(beginning);model.zero_grad(set_to_none=True)
        a,b=paired_forward(model,*obs)
        l=objective(a,b,target,mask,arm,weight)
        g=torch.autograd.grad(l['consistency'] if cons_only else l['loss'],params,allow_unused=False)
        return a.detach(),b.detach(),{k:float(v.detach()) for k,v in l.items()},[x.detach() for x in g]
    a,b,l1,g1=grad('P1',0.);backwards+=1
    a2,b2,l2,g2=grad('P2',0.);backwards+=1
    eq=max(float((x-y).abs().max()) for x,y in zip(g1,g2))
    assert eq==0 and torch.equal(a,a2) and torch.equal(b,b2) and l1==l2
    bc_grad=float(torch.sqrt(sum(x.double().square().sum() for x in g1)))
    del g1,g2
    restore_rng(beginning)
    ac,bc=paired_forward(model,obs[0],obs[0]); same=float((ac-bc).abs().max()); assert same==0
    del ac,bc
    _,_,lc,gc=grad('P2',.1,cons_only=True);backwards+=1
    cons_grad=float(torch.sqrt(sum(x.double().square().sum() for x in gc)))
    assert cons_grad>0 and all(torch.isfinite(x).all() for x in gc)
    del gc
    restore_rng(beginning)
    aa,bb=paired_forward(model,*obs);loss=objective(aa,bb,target,mask,'P2',.1)
    ga,gb=torch.autograd.grad(loss['consistency'],(aa,bb));backwards+=1
    assert ga.norm()>0 and gb.norm()>0 and torch.equal(ga,-gb)
    assert torch.count_nonzero(ga[:,8:])==0
    for invalid in ['latent_encoder','dino_extractor','latent_head']: assert getattr(model,invalid) is None
    results.append({'episode':t['episode'],'split':t['split'],'transition_id':t['transition_id'],
        'normalized_BC_MSE':l1['bc'],'normalized_consistency_MSE':l1['consistency'],
        'lambda0_gradient_max_abs':eq,'step0_output_max_abs':float((a-a2).abs().max()),
        'same_image_consistency_max_abs':same,'bc_gradient_norm':bc_grad,'consistency_gradient_norm':cons_grad,
        'lambda_0_1_gradient_ratio':.1*cons_grad/bc_grad,'both_output_gradients_nonzero':True,
        'finite':True,'masked_positions_zero_gradient':True})
    print('PAIR_PASS',j,t['episode'],results[-1],flush=True)
assert parameter_hash(model)==initial_hash
assert all(sha(p)==h for p,h in protected.items())
assert {str(p):git(p) for p in [R,W]}==before_git
dump('pairing_and_gradient_audit.json',{'status':'PASS_LIMITED_INTERFACE','results':results,
    'model_parameters':sum(p.numel() for p in params),'parameter_hash_before_after':initial_hash,
    'optimizer_updates':0,'optimizer_created':False,'backward_calls':backwards,'rollouts':0,
    'RLA_model_instantiated':False,'future_images_read':0,'heldout_read':0,
    'cpu_only':True,'torch':torch.__version__,'seed':42,'dtype':'float32','training_mode':True,
    'buffer_note':'ResNet GroupNorm replaces BatchNorm; no running-stat updates; full state_dict hash unchanged',
    'backend':'CPU math attention; no CUDA initialized','elapsed_seconds':time.time()-start,
    'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    'protection_hashes_verified':len(protected),'historical_git_unchanged':True,
    'scope_limit':'3 pairs only; no valid performance evaluation or formal training authorization'})
print('AUDIT_PASS_NO_UPDATES',flush=True)
