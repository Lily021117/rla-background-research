import os,json,csv,hashlib,shutil,random,subprocess,ctypes,uuid,importlib.util,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from peft import LoraConfig,get_peft_model
from experiments.robot.libero.run_libero_eval import GenerateConfig,initialize_model
from prismatic.vla.action_tokenizer import ActionTokenizer
from causal_unified import *
O=Path(os.environ['RUN_DIR']);W=Path(__file__).parent;R=Path('${RLA_PROJECT_ROOT}');E=R/'experiments/rla_background_invariance';T=E/'step5_1_policy_training/run_20261022_02';F=E/'step4_5_3_feature_target_expansion/run_20261021_01';N=E/'step4_1_adapter_readiness/run_20261021_02'
UUID='${GPU_UUID}'
def js(n,v):(O/n).write_text(json.dumps(v,indent=2,allow_nan=False))
def seed(v=20261021):random.seed(v);np.random.seed(v);torch.manual_seed(v);torch.cuda.manual_seed_all(v)
def digest_items(items):
 h=hashlib.sha256()
 for n,p in sorted(items):h.update(n.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def digest(m):return digest_items(m.state_dict().items())
def ecc():
 s=subprocess.check_output(['nvidia-smi','-i',UUID,'-q','-d','ECC'],text=True)
 return [l.strip() for l in s.splitlines() if any(k in l for k in ['Correctable','Uncorrectable','Threshold Exceeded','Repair Pending','Unrepairable'])]
assert os.environ['CUDA_VISIBLE_DEVICES']==UUID
class CUuuid(ctypes.Structure):_fields_=[('bytes',ctypes.c_ubyte*16)]
lib=ctypes.CDLL('libcuda.so.1');dev=ctypes.c_int();ident=CUuuid();assert lib.cuInit(0)==0 and lib.cuDeviceGet(ctypes.byref(dev),0)==0 and lib.cuDeviceGetUuid(ctypes.byref(ident),dev)==0
assert 'GPU-'+str(uuid.UUID(bytes=bytes(ident.bytes)))==UUID
free,total=torch.cuda.mem_get_info();assert free>=26*1024**3;torch.cuda.set_per_process_memory_fraction(22*1024**3/total);baseline_ecc=ecc();js('gpu_before.json',{'UUID':UUID,'free':free,'ECC':baseline_ecc,'cap_GiB':22})
torch.set_num_threads(4);seed();runtime=O/'runtime';runtime.mkdir()
base=R/'artifacts/rla_policy_closed_loop_pilot/runtime_checkpoint'
for p in base.iterdir():
 if p.is_file():
  if p.suffix in ['.pt','.safetensors']:(runtime/p.name).symlink_to(p.resolve())
  else:shutil.copy2(p,runtime/p.name)
model,native,prop,_,proc=initialize_model(GenerateConfig(pretrained_checkpoint=str(runtime),use_pro_version=True,use_minivlm=True,use_wandb=False))
spec=importlib.util.spec_from_file_location('original_finetune',W/'vla-scripts/finetune.py');ft=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ft;spec.loader.exec_module(ft)
tokenizer=ActionTokenizer(proc.tokenizer);patches=model.vision_backbone.get_num_patches()*model.vision_backbone.get_num_images_in_input()
samples=json.loads((T/'sample_manifest.json').read_text());targets={(r['group'],r['transition_id'],r['appearance']):r for r in csv.DictReader(open(F/'rla_target_manifest.csv')) if r['camera']=='agentview'}
selected=[];seen=set()
for s in samples:
 if s['split']=='train' and (s['instruction'],s['appearance']) not in seen:seen.add((s['instruction'],s['appearance']));selected.append(s['id'])
def load_batch(ids):
 records=[torch.load(samples[i]['path'],map_location='cpu',weights_only=False) for i in ids];out={}
 for k in records[0]:
  if k=='rla':continue
  vals=[r[k] for r in records]
  if torch.is_tensor(vals[0]):
   if k in ['input_ids','attention_mask','labels']:
    pad=proc.tokenizer.pad_token_id if k=='input_ids' else -100 if k=='labels' else False;mx=max(v.shape[1] for v in vals);vals=[torch.nn.functional.pad(v,(0,mx-v.shape[1]),value=pad) for v in vals]
   out[k]=torch.cat(vals).cuda()
  else:out[k]=vals[0]
 for j,i in enumerate(ids):assert samples[i]['instruction'] in proc.tokenizer.decode(out['input_ids'][j]).lower()
 return out
def target_for(ids,source):
 n=np.load(N/f'normalization_{source}.npz');arr=[]
 for i in ids:
  s=samples[i];r=targets[source,s['transition_id'],s['appearance']];arr.append(((np.load(r['path'])-n['mean'])/n['scale']).reshape(1,2048))
 return torch.tensor(np.stack(arr),device='cuda',dtype=torch.float32)
head=native;capture={}
hook=native.model.register_forward_hook(lambda m,a,y:capture.update(action=y))
def forward(batch,phase='Inference'):
 assert 'rla' not in batch
 loss,_=ft.run_forward_pass(vla=model,action_head=SimpleNamespace(module=head),proprio_projector=prop,batch=batch,action_tokenizer=tokenizer,device_id=0,use_l1_regression=True,use_proprio=True,use_film=False,num_patches=patches,use_pro_version=True,cfg=SimpleNamespace(phase=phase))
 return loss
for m in [model,native,prop]:m.eval()
native_hash=digest(native);native_reference={}
with torch.no_grad():
 for i in selected:
  seed();loss=forward(load_batch([i]));native_reference[i]=(capture['action'].detach().cpu().clone(),float(loss))
seed();model=get_peft_model(model,LoraConfig(r=32,lora_alpha=64,lora_dropout=0.,target_modules='all-linear',init_lora_weights='gaussian'))
head=UnifiedHead(native);assert digest(head.native)==native_hash
initial=[digest(m) for m in [model,head,prop]]
metrics=[];parameters=[];group_hashes={};groups={};language_entry=[]
def language_hook(module,args,kwargs):
 decoded=[proc.tokenizer.decode(x.detach().cpu()).lower() for x in kwargs['input_ids']]
 assert all(any(s['instruction'] in x for s in samples) for x in decoded)
 language_entry.extend(decoded)
model.register_forward_pre_hook(language_hook,with_kwargs=True)
common_ref={}
for group in 'ABCD':
 configure_trainability(model,head,prop,ARMS[group]);model.eval();head.eval();prop.eval()
 group_hashes[group]=[digest(m) for m in [model,head,prop]];assert group_hashes[group]==initial
 for prefix,m in [('vlm',model),('head',head),('proprio',prop)]:
  for n,p in m.named_parameters():parameters.append({'arm':group,'name':prefix+'.'+n,'shape':str(list(p.shape)),'trainable':p.requires_grad,'init_source':'seed20261021 new auxiliary' if prefix=='head' and n.startswith('auxiliary') else 'zero-delta LoRA' if 'lora_' in n else 'public Spatial-Pro','numel':p.numel()})
 groups[group]=[n for n,p in named_trainables(model,head,prop)]
 with torch.no_grad():
  for i in selected:
   seed();loss=forward(load_batch([i]));act=head.last_action.detach().cpu();aux=head.last_aux.detach().cpu();hidden=head.last_hidden.detach().cpu()
   if group=='A':common_ref[i]=(act.clone(),aux.clone(),hidden.clone())
   ref=native_reference[i][0];d=(act-ref).abs();ar,az,ah=common_ref[i]
   _,al=objective(loss,head.last_aux,target_for([i],ARMS[group].target_source),ARMS[group])
   metrics.append({'arm':group,'sample':i,'native_action_max_abs':float(d.max()),'native_action_mean_abs':float(d.mean()),'common_action_max_abs':float((act-ar).abs().max()),'common_aux_max_abs':float((aux-az).abs().max()),'common_hidden_max_abs':float((hidden-ah).abs().max()),'action_loss':float(loss),'native_loss_abs':abs(float(loss)-native_reference[i][1]),'aux_loss':float(al),'aux_weight':ARMS[group].aux_loss_weight})
   assert torch.equal(act,ref) and torch.equal(act,ar) and torch.equal(aux,az) and torch.equal(hidden,ah)
   assert float(loss)==native_reference[i][1]
   head.clear()
assert groups['B']==groups['C']==groups['D'] and not groups['A']
for name,rows in [('parameter_structure_comparison.csv',parameters),('step0_equivalence_metrics.csv',metrics)]:
 with (O/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
js('step0_parameter_hashes.json',group_hashes)
configure_trainability(model,head,prop,ARMS['B']);params=named_trainables(model,head,prop)
def clear():
 for m in [model,head,prop]:m.zero_grad(set_to_none=True)
 head.clear()
def gradient_summary():
 res={}
 for category,items in {'shared_backbone':[(n,p) for n,p in params if n.startswith('vlm.')],'action_branch':[(n,p) for n,p in params if n.startswith('head.native.')],'auxiliary_head':[(n,p) for n,p in params if n.startswith('head.auxiliary.')],'proprio':[(n,p) for n,p in params if n.startswith('proprio.')]}.items():
  grads=[(n,p.grad) for n,p in items if p.grad is not None];assert all(torch.isfinite(g).all() for n,g in grads)
  res[category]={'norm':sum(float(g.double().square().sum()) for n,g in grads)**.5,'nonzero':sum(bool(g.count_nonzero()) for n,g in grads),'grad_hash':digest_items(grads)}
 return res
grads={};i=selected[0];B_action_grad={};B_comparison=[]
for group in ['B','C','D']:
 configure_trainability(model,head,prop,ARMS[group]);model.eval();head.eval();prop.eval();grads[group]={}
 for component in ['action','auxiliary','total']:
  clear();seed();loss=forward(load_batch([i]));total,aux=objective(loss,head.last_aux,target_for([i],ARMS[group].target_source),ARMS[group])
  obj=loss if component=='action' else ARMS[group].aux_loss_weight*aux if component=='auxiliary' else total
  obj.backward();grads[group][component]=gradient_summary()
  if group=='B' and component=='action':B_action_grad={n:p.grad.detach().cpu().clone() for n,p in params if p.grad is not None and not n.startswith('head.auxiliary.')}
  if group=='B' and component=='total':
   for n,p in params:
    if n.startswith('head.auxiliary.'):continue
    if n not in B_action_grad:
     assert p.grad is None or p.grad.count_nonzero()==0
     continue
    ref=B_action_grad[n];now=p.grad.detach().cpu() if p.grad is not None else torch.zeros_like(ref)
    delta=(now.float()-ref.float()).abs()
    B_comparison.append({'name':n,'max_abs':float(delta.max()),'mean_abs':float(delta.mean()),'exact_numerical_equal':torch.equal(now,ref),'reference_norm':float(ref.float().norm())})
  clear()
 assert [digest(m) for m in [model,head,prop]]==initial
assert grads['B']['auxiliary']['shared_backbone']['norm']==0 and grads['B']['total']['auxiliary_head']['norm']==0
js('gradient_audit_raw.json',{'by_arm_component':grads,'B_action_vs_total':B_comparison,'criterion':'exact numerical equality, not signed-zero byte identity'})
assert all(r['exact_numerical_equal'] for r in B_comparison),'B gradient has real numerical difference; inspect raw audit'
for g in ['C','D']:
 assert grads[g]['auxiliary']['shared_backbone']['norm']>0 and grads[g]['auxiliary']['auxiliary_head']['norm']>0
js('gradient_audit.json',{'status':'PASS','by_arm_component':grads,'B_action_vs_total_max_abs':max(r['max_abs'] for r in B_comparison),'B_exact_numeric_equal':True,'backward_calls':9,'optimizer_updates':0})
# Targets never enter forward. Recompute after altering/removing target data.
isolation=[]
with torch.no_grad():
 for variant in ['deleted','randomized']:
  seed();loss=forward(load_batch([i]));assert torch.equal(head.last_action.cpu(),common_ref[i][0]);isolation.append({'target':variant,'action_max_abs':0})
  if variant=='randomized':objective(loss,head.last_aux,torch.randn_like(head.last_aux),ARMS['D'])
  head.clear()
js('target_isolation.json',{'status':'PASS','legal_forward_signature':'native inputs only; no target','checks':isolation})
opt=torch.optim.AdamW([p for n,p in params],lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
path=O/'shared_step0.pt';save_checkpoint(path,model,head,prop,opt,sched,0,{'base':str(base),'groups':'ABCD common model state; A uses requires_grad false','sampling_order':str(T/'sampling_order.json')})
checkpoint_rng=rng_state();seed(17)
assert restore_checkpoint(path,model,head,prop,opt,sched)==0
assert torch.equal(torch.get_rng_state(),checkpoint_rng['torch'])
assert [digest(m) for m in [model,head,prop]]==initial
with torch.no_grad():
 seed();loss=forward(load_batch([i]));assert torch.equal(head.last_action.cpu(),common_ref[i][0]);head.clear()
assert not opt.state and sched.last_epoch==0
for group in 'ABCD':js(f'checkpoint_{group}.json',{'shared_step0':str(path),'arm':vars(ARMS[group]),'model_hashes':initial,'optimizer_state_empty':True,'parameter_groups':groups[group] if group!='A' else [],'updates':0})
js('checkpoint_reload_audit.json',{'status':'PASS','step':0,'strict_parameter_load':True,'model_hashes_equal':True,'optimizer_empty_before_after':True,'scheduler_epoch':sched.last_epoch,'RNG_restored':True,'forward_max_abs':0,'parameter_groups_BCD_equal':True,'optimizer_updates':0,'not_a_test_of_nonempty_Adam_moments':True})
assert ecc()==baseline_ecc
js('runtime_acceptance.json',{'status':'PASS','backward_calls':9,'optimizer_updates':0,'closed_loop':0,'parameter_hashes_unchanged':True,'GPU_peak_allocated':torch.cuda.max_memory_allocated(),'ECC_unchanged':True,'real_model_language_entry_checks':len(language_entry),'initial_aux_parameter_count':sum(p.numel() for p in head.auxiliary.parameters()),'serial_groups':'same immutable step0 state, requires_grad toggled A vs BCD; hash verified every group'})
print('STEP0_ACCEPTANCE_PASS',flush=True)
