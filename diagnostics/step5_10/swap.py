"""Eight frozen in-memory component combinations; no optimizer/backward/rollout."""
from common import *
import sys,random,ctypes,uuid,shutil,time,resource,gc,importlib.util,collections
import torch
sys.path.insert(0,str(W))
from peft import LoraConfig,get_peft_model
from experiments.robot.libero.run_libero_eval import GenerateConfig,initialize_model,process_action
from prismatic.vla.action_tokenizer import ActionTokenizer
from causal_unified import UnifiedHead
from types import SimpleNamespace
assert read(O/'preparation_acceptance.json')['status']=='PASS'
UUID=os.environ['DIAGNOSTIC_GPU_UUID'];assert os.environ['CUDA_VISIBLE_DEVICES']==UUID
assert UUID=='${GPU_UUID}'
class CUuuid(ctypes.Structure):_fields_=[('bytes',ctypes.c_ubyte*16)]
lib=ctypes.CDLL('libcuda.so.1');dev=ctypes.c_int();ident=CUuuid();assert lib.cuInit(0)==0 and lib.cuDeviceGet(ctypes.byref(dev),0)==0 and lib.cuDeviceGetUuid(ctypes.byref(ident),dev)==0
assert 'GPU-'+str(uuid.UUID(bytes=bytes(ident.bytes)))==UUID
def ecc():return [l.strip() for l in subprocess.check_output(['nvidia-smi','-i',UUID,'-q','-d','ECC'],text=True).splitlines() if any(k in l for k in ['Correctable','Uncorrectable','Threshold Exceeded','Repair Pending','Unrepairable'])]
free,total=torch.cuda.mem_get_info();assert free>=26*1024**3;torch.cuda.set_per_process_memory_fraction(14*1024**3/total);baseecc=ecc()
js('gpu_before.json',{'UUID':UUID,'free_bytes':free,'ECC':baseecc,'allocation_cap_GiB':14,'other_processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv'],text=True)})
torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True);torch.set_num_threads(4)
def seed():random.seed(20261021);np.random.seed(20261021);torch.manual_seed(20261021);torch.cuda.manual_seed_all(20261021)
def digest(items):
 h=hashlib.sha256()
 for n,x in sorted(items):h.update(n.encode());h.update(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def backend():
 d=dict(flash=torch.backends.cuda.flash_sdp_enabled(),mem_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),math=torch.backends.cuda.math_sdp_enabled());assert d==dict(flash=False,mem_efficient=False,math=True);return d
runtime=O/'runtime';runtime.mkdir()
for p in BASE.iterdir():
 if p.is_file():
  if p.suffix in ['.pt','.safetensors']:(runtime/p.name).symlink_to(p.resolve())
  else:shutil.copy2(p,runtime/p.name)
seed();model,native,prop,_,proc=initialize_model(GenerateConfig(pretrained_checkpoint=str(runtime),use_pro_version=True,use_minivlm=True,use_wandb=False))
capture={};hook=native.model.register_forward_hook(lambda m,a,y:capture.update(normalized=y))
selected=read(O/'fixed_sample_manifest.json')['samples']
def batch(s):
 b=torch.load(s['path'],map_location='cpu',weights_only=False);b={k:v.cuda() if torch.is_tensor(v) else v for k,v in b.items() if k!='rla'}
 assert b['pixel_values'].shape==(1,12,224,224) and b['actions'].shape==(1,8,7) and b['proprio'].shape==(1,8)
 return b
def current_prompt(s,b):
 prompt=f'<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat action should the robot take to {s["instruction"]}?<|im_end|>\n<|im_start|>assistant\n'
 inp={k:v.cuda() for k,v in proc.tokenizer(prompt,return_tensors='pt').items()};assert torch.equal(inp['input_ids'],b['input_ids'][:,:-64]);return inp
def deploy(s,b,action_head):
 inp=current_prompt(s,b)
 with torch.inference_mode():
  decoded,_=model.predict_action(**inp,pixel_values=b['pixel_values'].to(torch.bfloat16),unnorm_key='libero_spatial_no_noops',do_sample=False,proprio=b['proprio'][0].float().cpu().numpy(),proprio_projector=prop,action_head=action_head,use_film=False)
 normalized=capture['normalized'].float().cpu().numpy().reshape(8,7).copy()
 return normalized,np.asarray(decoded)
for m in [model,native,prop]:m.eval();m.requires_grad_(False)
native_ref=[]
for s in selected:seed();native_ref.append(deploy(s,batch(s),native)[0])
seed();model=get_peft_model(model,LoraConfig(r=32,lora_alpha=64,lora_dropout=0.,target_modules='all-linear',init_lora_weights='gaussian'));head=UnifiedHead(native)
assert [digest(m.state_dict().items()) for m in [model,head,prop]]==read(T/'step0_parameter_hashes.json')['B']
spec=importlib.util.spec_from_file_location('diagnostic_finetune',W/'vla-scripts/finetune.py');ft=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ft;spec.loader.exec_module(ft)
actiontok=ActionTokenizer(proc.tokenizer);patches=model.vision_backbone.get_num_patches()*model.vision_backbone.get_num_images_in_input()
checkpoints=[torch.load(str(T/'B'/f'step{step:04d}.pt'),map_location='cpu',weights_only=False,mmap=True) for step in [0,1]]
trainable={r['name'] for r in csvread(T/'parameter_structure_comparison.csv') if r['arm']=='B' and r['trainable']=='True'}
modules={'VLM':model,'action_head':native,'proprio':prop,'auxiliary_fixed':head.auxiliary};groups={};owners=collections.defaultdict(set);storageowners=collections.defaultdict(set);buffers={};fullnorms={}
for component,m in modules.items():
 records=[];unique={};bufferrows=[]
 for n,p in m.named_parameters(remove_duplicate=False):
  name=('vlm.' if component=='VLM' else 'head.native.' if component=='action_head' else 'proprio.' if component=='proprio' else 'head.auxiliary.')+n
  owners[id(p)].add(component)
  if p.numel():storageowners[p.untyped_storage().data_ptr()].add(component)
  unique[id(p)]=p
  records.append({'name':name,'shape':list(p.shape),'numel':p.numel(),'historically_trainable':name in trainable,'object_id_group':str(id(p)),'dtype':str(p.dtype)})
 for n,b in m.named_buffers(remove_duplicate=False):bufferrows.append({'name':n,'shape':list(b.shape),'dtype':str(b.dtype),'persistent':n in m.state_dict(),'source':'same base configuration; no differing saved buffer omitted','SHA256':digest([(n,b)])})
 fullnorms[component]=sum(float(p.detach().float().square().sum()) for p in unique.values())**.5
 groups[component]={'parameter_count_unique':sum(p.numel() for p in unique.values()),'historically_trainable_parameter_count':sum(r['numel'] for r in records if r['historically_trainable']),'parameters':records,'buffers':bufferrows,'full_step0_parameter_norm':fullnorms[component],'saved_source':'vlm LoRA/action_queries only, other VLM params fixed public base' if component=='VLM' else 'head.native state' if component=='action_head' else 'proprio state' if component=='proprio' else 'head.auxiliary step0 fixed'}
assert all(len(v)==1 for v in owners.values()) and all(len(v)==1 for v in storageowners.values())
aliasgroups=[]
for c,m in modules.items():
 dd=collections.defaultdict(list)
 for n,p in m.named_parameters(remove_duplicate=False):dd[id(p)].append(n)
 aliasgroups.extend({'component':c,'names':v} for v in dd.values() if len(v)>1)
expected={n for n,p in model.named_parameters() if 'lora_' in n or 'action_queries' in n};assert expected==set(checkpoints[0]['vlm'])==set(checkpoints[1]['vlm'])
covered={r['name'] for g in groups.values() for r in g['parameters']};assert trainable<=covered
frozen0=digest((n,p) for n,p in model.named_parameters() if n not in expected)
js('component_inventory.json',{'status':'PASS_COMPLETE_DISJOINT','components':groups,'cross_component_parameter_aliases':[],'cross_component_storage_aliases':[],'within_component_tied_parameter_groups':aliasgroups,'uncovered_historical_trainables':sorted(trainable-covered),'all_action_affecting_modules_covered':True,'visual_backbone_projector_LoRA_action_queries_normalization_embedding':'all belong to VLM; frozen base stays same across combinations','auxiliary_policy':'fixed step0, forward output ablation checked separately','nonpersistent_buffers':'listed and owned by their component; deterministic same configuration; no optimizer updates to buffers'})
def load_combo(bits):
 with torch.no_grad():
  for n,p in model.named_parameters():
   if n in expected:p.copy_(checkpoints[int(bits[0])]['vlm'][n].to(device=p.device,dtype=p.dtype))
 h=checkpoints[int(bits[1])]['head'];native.load_state_dict({k[len('native.'):]:v for k,v in h.items() if k.startswith('native.')},strict=True)
 prop.load_state_dict(checkpoints[int(bits[2])]['proprio'],strict=True)
 head.auxiliary.load_state_dict({k[len('auxiliary.'):]:v for k,v in checkpoints[0]['head'].items() if k.startswith('auxiliary.')},strict=True)
 for m in [model,head,prop]:m.eval();m.requires_grad_(False)
 head.clear()
def changing_hashes():return [digest((n,p) for n,p in model.named_parameters() if n in expected),digest(native.state_dict().items()),digest(prop.state_dict().items()),digest(head.auxiliary.state_dict().items())]
rows=[];positionrows=[];outs={};baseline=None;checks=[];start=time.monotonic();targetall=[]
for bits in read(O/'fixed_sample_manifest.json')['group_order']:
 assert ecc()==baseecc;backend();load_combo(bits);before=changing_hashes();out=[]
 for index,s in enumerate(selected):
  seed();b=batch(s);target=b['actions'][0].float().cpu().numpy();aug=dict(b)
  for k in ['input_ids','labels']:aug[k]=torch.cat([b[k],torch.full_like(b[k][:,:1],2)],dim=1)
  aug['attention_mask']=torch.cat([b['attention_mask'],torch.ones_like(b['attention_mask'][:,:1])],dim=1)
  with torch.inference_mode():
   ft.run_forward_pass(vla=model,action_head=SimpleNamespace(module=head),proprio_projector=prop,batch=aug,action_tokenizer=actiontok,device_id=0,use_l1_regression=True,use_proprio=True,use_film=False,num_patches=patches,use_pro_version=True,cfg=SimpleNamespace(phase='Inference'))
   trained=capture['normalized'].float().cpu().numpy().reshape(8,7).copy();head.clear()
  seed();pred,decoded=deploy(s,b,head);head.clear();assert np.array_equal(pred,trained),'unified STOP forward/deploy mismatch'
  if index==0:
   seed();repeat,_=deploy(s,b,head);head.clear();assert np.array_equal(pred,repeat)
  if bits=='000':
   assert np.array_equal(pred,native_ref[index]),'full step0 differs frozen native A'
   targetall.append(target)
   if index==0:
    ablation=head.auxiliary.register_forward_hook(lambda m,a,y:torch.zeros_like(y));seed();ablated,_=deploy(s,b,head);head.clear();ablation.remove();assert np.array_equal(pred,ablated),'auxiliary output affects action'
  assert np.isfinite(pred).all() and np.isfinite(decoded).all();out.append(pred)
  reference=pred if baseline is None else baseline[index];decodedtarget=model._unnormalize_actions(target,unnorm_key='libero_spatial_no_noops');ep=np.stack([process_action(x.copy(),'openvla') for x in decoded]);et=np.stack([process_action(x.copy(),'openvla') for x in decodedtarget]);cp=np.clip(ep[:,:6],-1,1)*np.array([.05]*3+[.5]*3);ct=np.clip(et[:,:6],-1,1)*np.array([.05]*3+[.5]*3)
  base={'combination':bits,'VLM_step':int(bits[0]),'head_step':int(bits[1]),'proprio_step':int(bits[2]),'sample_id':s['id'],'episode':s['episode'],'task':s['instruction'],'split':s['split'],'appearance':s['appearance']}
  for position in range(8):
   err=abs(pred[position]-target[position]);drift=abs(pred[position]-reference[position]);e=abs(ep[position]-et[position]);c=abs(cp[position]-ct[position])
   positionrows.append({**base,'chunk_position':position,'normalized_L1':float(err.mean()),'normalized_drift_L1':float(drift.mean()),'normalized_translation_L1':float(err[:3].mean()),'normalized_rotation_L1':float(err[3:6].mean()),'normalized_gripper_L1':float(err[6]),'decoded_translation_command_L1':float(e[:3].mean()),'decoded_rotation_command_L1':float(e[3:6].mean()),'gripper_command_L1':float(e[6]),'gripper_command_matches':bool(ep[position,6]==et[position,6]),'controller_translation_delta_L1_m':float(c[:3].mean()),'controller_rotation_delta_L1_rad':float(c[3:6].mean()),'normalized_first6_outside_unit_ratio':float((abs(pred[position,:6])>1).mean()),'gripper_outside_0_1':bool(pred[position,6]<0 or pred[position,6]>1),'controller_first6_clipping_ratio':float((abs(ep[position,:6])>1).mean()),'finite':True})
 if bits=='000':baseline=np.stack(out)
 outs['combo_'+bits]=np.stack(out);assert changing_hashes()==before,'inference modified parameters'
 checks.append({'combination':bits,'strict_source_load':True,'inference_parameter_hashes_unchanged':True,'parameter_hashes':before,'STOP_train_deploy_max_abs':0,'repeat_inference_max_abs':0,'native_A_max_abs':0 if bits=='000' else None})
 print('COMBINATION_COMPLETE',bits,flush=True)
assert digest((n,p) for n,p in model.named_parameters() if n not in expected)==frozen0
assert ecc()==baseecc;outs['targets']=np.stack(targetall);np.savez(O/'fixed_outputs.npz',**outs);cs('per_position_metrics.csv',positionrows)
js('inference_acceptance.json',{'status':'PASS','combinations':8,'samples':14,'all_strict_source_loads':True,'all_finite':True,'full_step0_native_A_exact_equal':True,'STOP_train_deploy_max_abs':0,'repeat_max_abs':0,'auxiliary_ablation_action_max_abs':0,'frozen_base_parameters_unchanged':True,'all_parameters_unchanged_by_inference':True,'combination_checks':checks,'optimizer_updates':0,'backward_calls':0,'new_rollouts':0,'new_heldout_scores':0,'GPU_UUID':UUID,'ECC_unchanged':True,'GPU_peak_allocated':torch.cuda.max_memory_allocated(),'CPU_peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'seconds':time.monotonic()-start,'backend':backend(),'model_checkpoints_written':0})
print('ALL_SWAPS_PASS',flush=True)
