"""Frozen real-input training/deployment comparison; zero backward/updates."""
import os,json,csv,hashlib,shutil,random,time,subprocess,ctypes,uuid,sys,importlib.util,gc
from pathlib import Path
O=Path(__file__).parent;R=Path('${RLA_PROJECT_ROOT}');E=R/'experiments/rla_background_invariance';W=R/'worktrees/rla-background-causal-unified-trainer';T=E/'step5_7_causal_training/run_20261022_01';S=E/'step5_1_policy_training/run_20261022_02'
sys.path.insert(0,str(W))
import torch,numpy as np
from peft import LoraConfig,get_peft_model
from experiments.robot.libero.run_libero_eval import GenerateConfig,initialize_model
from prismatic.vla.action_tokenizer import ActionTokenizer
from causal_unified import UnifiedHead
from types import SimpleNamespace
UUID='${GPU_UUID}'
def js(n,x):
 with (O/n).open('x') as f:json.dump(x,f,indent=2)
def digest(m):
 h=hashlib.sha256()
 for n,p in sorted(m.state_dict().items()):h.update(n.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def ecc():return [l.strip() for l in subprocess.check_output(['nvidia-smi','-i',UUID,'-q','-d','ECC'],text=True).splitlines() if any(k in l for k in ['Correctable','Uncorrectable','Threshold Exceeded','Repair Pending','Unrepairable'])]
assert os.environ['CUDA_VISIBLE_DEVICES']==UUID
class CUuuid(ctypes.Structure):_fields_=[('bytes',ctypes.c_ubyte*16)]
lib=ctypes.CDLL('libcuda.so.1');dev=ctypes.c_int();ident=CUuuid();assert lib.cuInit(0)==0 and lib.cuDeviceGet(ctypes.byref(dev),0)==0 and lib.cuDeviceGetUuid(ctypes.byref(ident),dev)==0
assert 'GPU-'+str(uuid.UUID(bytes=bytes(ident.bytes)))==UUID
free,total=torch.cuda.mem_get_info();assert free>=26*1024**3;torch.cuda.set_per_process_memory_fraction(20*1024**3/total);baseecc=ecc();js('offline_gpu_before.json',{'UUID':UUID,'free':free,'ECC':baseecc})
torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True);torch.set_num_threads(4)
runtime=O/'offline_runtime';runtime.mkdir();base=R/'artifacts/rla_policy_closed_loop_pilot/runtime_checkpoint'
for p in base.iterdir():
 if p.is_file():
  if p.suffix in ['.pt','.safetensors']:(runtime/p.name).symlink_to(p.resolve())
  else:shutil.copy2(p,runtime/p.name)
model,native,prop,_,proc=initialize_model(GenerateConfig(pretrained_checkpoint=str(runtime),use_pro_version=True,use_minivlm=True,use_wandb=False))
random.seed(20261021);np.random.seed(20261021);torch.manual_seed(20261021);torch.cuda.manual_seed_all(20261021)
model=get_peft_model(model,LoraConfig(r=32,lora_alpha=64,lora_dropout=0.,target_modules='all-linear',init_lora_weights='gaussian'));head=UnifiedHead(native)
assert [digest(m) for m in [model,head,prop]]==json.loads((T/'step0_parameter_hashes.json').read_text())['B']
spec=importlib.util.spec_from_file_location('frozen_finetune',W/'vla-scripts/finetune.py');ft=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ft;spec.loader.exec_module(ft)
at=ActionTokenizer(proc.tokenizer);patches=model.vision_backbone.get_num_patches()*model.vision_backbone.get_num_images_in_input();samples=json.loads((S/'sample_manifest.json').read_text());seen=set();selected=[]
for s in samples:
 key=(s['split'],s['instruction'],s['appearance'])
 if key not in seen:seen.add(key);selected.append(s)
rows=[];outputs={};checkpoints=[('A',0),('B',1),('B',50),('B',100),('C',100),('D',100)]
for arm,step in checkpoints:
 ck=torch.load(T/'common_step0.pt' if step==0 else T/arm/f'step{step:04d}.pt',map_location='cpu',weights_only=False)
 expected={n for n,p in model.named_parameters() if 'lora_' in n or 'action_queries' in n};assert expected==set(ck['vlm'])
 with torch.no_grad():
  for n,p in model.named_parameters():
   if n in expected:p.copy_(ck['vlm'][n].to(p.device))
 head.load_state_dict(ck['head'],strict=True);prop.load_state_dict(ck['proprio'],strict=True);del ck;gc.collect()
 for m in [model,head,prop]:m.eval();m.requires_grad_(False)
 before=[digest(m) for m in [model,head,prop]]
 for s in selected:
  b=torch.load(s['path'],map_location='cpu',weights_only=False);b={k:v.cuda() if torch.is_tensor(v) else v for k,v in b.items() if k!='rla'}
  with torch.inference_mode():
   loss,_=ft.run_forward_pass(vla=model,action_head=SimpleNamespace(module=head),proprio_projector=prop,batch=b,action_tokenizer=at,device_id=0,use_l1_regression=True,use_proprio=True,use_film=False,num_patches=patches,use_pro_version=True,cfg=SimpleNamespace(phase='Inference'))
   teach=head.last_action.detach().float().cpu().numpy();head.clear()
   prompt=f'<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat action should the robot take to {s["instruction"]}?<|im_end|>\n<|im_start|>assistant\n'
   inp=proc.tokenizer(prompt,return_tensors='pt');inp={k:v.cuda() for k,v in inp.items()}
   act,_=model.predict_action(**inp,pixel_values=b['pixel_values'].to(torch.bfloat16),unnorm_key='libero_spatial_no_noops',do_sample=False,proprio=b['proprio'][0].float().cpu().numpy(),proprio_projector=prop,action_head=head,use_film=False)
   deploy=head.last_action.detach().float().cpu().numpy();head.clear()
  assert np.isfinite(deploy).all() and np.isfinite(teach).all();target=b['actions'].float().cpu().numpy();ref=outputs.get(('A',0,s['id']))
  outputs[arm,step,s['id']]=deploy
  rows.append({'arm':arm,'step':step,'sample_id':s['id'],'split':s['split'],'episode':s['episode'],'appearance':s['appearance'],'training_path_action_L1':float(np.mean(abs(teach-target))),'deployment_path_action_L1':float(np.mean(abs(deploy-target))),'train_deploy_normalized_action_max_abs':float(abs(deploy-teach).max()),'train_deploy_normalized_action_L1':float(abs(deploy-teach).mean()),'deployment_shift_from_A_L1':0 if ref is None else float(abs(deploy-ref).mean()),'first_training_action':json.dumps(teach[0,0].tolist()),'first_deploy_action':json.dumps(deploy[0,0].tolist()),'train_sequence_length':int(b['input_ids'].shape[1]),'deploy_prompt_length':int(inp['input_ids'].shape[1])})
 assert [digest(m) for m in [model,head,prop]]==before;assert ecc()==baseecc
 print('FROZEN_CHECKPOINT',arm,step,flush=True)
with (O/'fixed_input_action_probe.csv').open('x') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
js('offline_probe_verification.json',{'status':'PASS','samples_per_checkpoint':len(selected),'checkpoint_versions':len(checkpoints),'Math_SDPA':True,'optimizer_updates':0,'backward_calls':0,'new_rollouts':0,'parameters_unchanged':True,'ECC_unchanged':ecc()==baseecc,'peak_GPU_allocated':torch.cuda.max_memory_allocated(),'UUID':UUID,'scope':'matched current pixel/proprio/language, actual training forward vs deployment predict_action; no future target in either forward'})
print('OFFLINE_PROBE_COMPLETE',flush=True)
