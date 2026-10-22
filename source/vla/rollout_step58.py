import os,json,csv,hashlib,time,shutil,subprocess,ctypes,uuid,traceback,random,resource,sys,gc
from pathlib import Path
from collections import deque
import numpy as np,torch,imageio
from PIL import Image
from peft import LoraConfig,get_peft_model
from experiments.robot.libero.run_libero_eval import GenerateConfig,initialize_model,process_action
from experiments.robot.openvla_utils import normalize_proprio
from causal_unified import UnifiedHead
E=Path('${RLA_PROJECT_ROOT}/experiments/rla_background_invariance');OLD=E/'step5_2_closed_loop/run_20261022_02';T=E/'step5_7_causal_training/run_20261022_01';O=Path(os.environ['RUN_DIR'])
sys.path.insert(0,str(OLD));from env_helpers import make_env,fingerprint,renderer,observation
UUID='${GPU_UUID}'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def js(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False)
def digest(m):
 h=hashlib.sha256()
 for n,p in sorted(m.state_dict().items()):h.update(n.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def ecc():
 raw=subprocess.check_output(['nvidia-smi','-i',UUID,'-q','-d','ECC'],text=True)
 return [l.strip() for l in raw.splitlines() if any(k in l for k in ['Correctable','Uncorrectable','Threshold Exceeded','Repair Pending','Unrepairable'])]
assert os.environ['CUDA_VISIBLE_DEVICES']==UUID
class CUuuid(ctypes.Structure):_fields_=[('bytes',ctypes.c_ubyte*16)]
lib=ctypes.CDLL('libcuda.so.1');dev=ctypes.c_int();ident=CUuuid();assert lib.cuInit(0)==0 and lib.cuDeviceGet(ctypes.byref(dev),0)==0 and lib.cuDeviceGetUuid(ctypes.byref(ident),dev)==0
assert 'GPU-'+str(uuid.UUID(bytes=bytes(ident.bytes)))==UUID
free,total=torch.cuda.mem_get_info();assert free>=26*1024**3;torch.cuda.set_per_process_memory_fraction(24*1024**3/total)
torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
base_ecc=ecc();js(O/'gpu_before.json',{'UUID':UUID,'free_bytes':free,'ECC':base_ecc,'allocation_cap_GiB':24})
frozen=json.loads((O/'manifest_frozen.json').read_text());assert sha(O/'evaluation_manifest.csv')==frozen['manifest_SHA256'] and sha(O/'evaluation_protocol.json')==frozen['protocol_SHA256']
torch.set_num_threads(4);models={};inv=list(csv.DictReader(open(O/'checkpoint_inventory.csv')))
initial=json.loads((T/'step0_parameter_hashes.json').read_text())['B'];final=json.loads((T/'checkpoint_reload_audit.json').read_text())
for arm in 'ABCD':
 row=next(x for x in inv if x['arm']==arm);assert sha(row['path'])==row['SHA256'];runtime=O/f'{arm}_runtime';runtime.mkdir()
 for p in Path(row['base_directory']).iterdir():
  if p.is_file():
   if p.suffix in ['.pt','.safetensors']:(runtime/p.name).symlink_to(p.resolve())
   else:shutil.copy2(p,runtime/p.name)
 random.seed(20261021);np.random.seed(20261021);torch.manual_seed(20261021);torch.cuda.manual_seed_all(20261021)
 model,native,prop,_,proc=initialize_model(GenerateConfig(pretrained_checkpoint=str(runtime),use_pro_version=True,use_minivlm=True,use_wandb=False))
 random.seed(20261021);np.random.seed(20261021);torch.manual_seed(20261021);torch.cuda.manual_seed_all(20261021)
 model=get_peft_model(model,LoraConfig(r=32,lora_alpha=64,lora_dropout=0.,target_modules='all-linear',init_lora_weights='gaussian'));head=UnifiedHead(native)
 assert [digest(m) for m in [model,head,prop]]==initial
 if arm!='A':
  ck=torch.load(row['path'],map_location='cpu',weights_only=False);assert ck['step']==100 and ck['provenance']['backend']=={'flash':False,'mem_efficient':False,'math':True}
  expected={n for n,p in model.named_parameters() if 'lora_' in n or 'action_queries' in n};assert set(ck['vlm'])==expected
  head.load_state_dict(ck['head'],strict=True);prop.load_state_dict(ck['proprio'],strict=True)
  with torch.no_grad():
   for n,p in model.named_parameters():
    if n in expected:p.copy_(ck['vlm'][n].to(p.device))
  del ck;gc.collect()
  assert [digest(m) for m in [model,head,prop]]==final[arm]['parameter_hashes']
 for m in [model,head,prop]:m.eval();m.requires_grad_(False)
 hashes=[digest(m) for m in [model,head,prop]];models[arm]=(model,head,prop,proc,hashes)
 js(O/f'{arm}_loaded.json',{'strict_load':True,'checkpoint_SHA256':row['SHA256'],'parameter_hashes':hashes,'all_frozen':True,'no_optimizer':True,'global_step':int(row['step'])})
 print('MODEL_LOADED',arm,flush=True)
def infer(arm,images,proprio,language):
 model,head,prop,proc,_=models[arm]
 prompt=f'<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat action should the robot take to {language.lower()}?<|im_end|>\n<|im_start|>assistant\n'
 inp=proc(prompt,Image.fromarray(images[0]));inp['pixel_values']=torch.cat([inp['pixel_values'],proc.image_processor(Image.fromarray(images[1]),return_tensors='pt')['pixel_values']],dim=1)
 assert language in proc.tokenizer.decode(inp['input_ids'][0],skip_special_tokens=False).lower()
 for k,v in inp.items():inp[k]=v.cuda().to(torch.bfloat16) if k=='pixel_values' else v.cuda()
 assert inp['pixel_values'].shape==(1,12,224,224)
 state=normalize_proprio(proprio.copy(),model.norm_stats['libero_spatial_no_noops']['proprio'])
 with torch.inference_mode():actions,_=model.predict_action(**inp,unnorm_key='libero_spatial_no_noops',do_sample=False,proprio=state,proprio_projector=prop,action_head=head,use_film=False)
 head.clear();actions=np.asarray(actions);assert actions.shape==(8,7) and np.isfinite(actions).all();return actions
(O/'trials').mkdir();rows=list(csv.DictReader(open(O/'evaluation_manifest.csv')));results=[];start=time.time()
for ix,row in enumerate(rows):
 assert ecc()==base_ecc,'ECC increment';assert not torch.backends.cuda.flash_sdp_enabled() and not torch.backends.cuda.mem_efficient_sdp_enabled() and torch.backends.cuda.math_sdp_enabled()
 out=O/'trials'/row['trial_id'];out.mkdir();env=r=writer=None;executed=[];chunks=[];lat=[];states=[];proprios=[];result={**row,'status':'INVALID','success':None,'policy_steps':0,'timeout':False,'invalid_action':False,'simulator_exception':False,'success_source':'env.check_success()','optimizer_updates':0,'video':str(out/'video.mp4'),'trajectory':str(out/'trajectory.npz')};t=time.time();stage='reset'
 try:
  random.seed(int(row['seed']));np.random.seed(int(row['seed']));torch.manual_seed(int(row['seed']));torch.cuda.manual_seed_all(int(row['seed']))
  env=make_env(row,row['appearance']);r=renderer(env)
  for _ in range(10):env.step([0,0,0,0,0,0,-1])
  env.sim.forward();fp=fingerprint(env)
  with np.load(OLD/'geometry'/f'{row["state_id"]}_settled.npz') as ref:
   for k,v in fp.items():np.testing.assert_allclose(v,ref[k],rtol=0,atol=1e-8,err_msg=k)
  assert not env.check_success();q=deque();writer=imageio.get_writer(out/'video.mp4',fps=2.5)
  for step in range(220):
   if not q:
    assert ecc()==base_ecc,'ECC increment during rollout';stage='observation'
    imgs,prop,rgb=observation(env,r);proprios.append(prop.copy());writer.append_data(np.concatenate(rgb,axis=1));ts=time.time();stage='inference';a=infer(row['model'],imgs,prop,row['task']);torch.cuda.synchronize();lat.append(time.time()-ts)
    if ix<4 and step==0:
     repeat=infer(row['model'],imgs,prop,row['task']);assert np.array_equal(a,repeat),'repeat inference mismatch';js(out/'smoke.json',{'shape':[8,7],'finite_actions':True,'repeat_max':0,'initial_geometry_passed':True,'counts_in_160':True,'current_observation_FK_language_only':True})
    chunks.append(a.copy());q.extend(a)
   stage='action';action=process_action(q.popleft().copy(),'openvla');assert np.isfinite(action).all()
   stage='simulation';_,_,done,_=env.step(action.tolist());executed.append(action);states.append(env.get_sim_state().copy());result['policy_steps']=step+1;success=bool(env.check_success())
   if success or done:break
  result.update(status='COMPLETE',success=success,timeout=not success and not done and result['policy_steps']==220,failure_reason=None if success else ('environment_done_without_success' if done else 'timeout_220_steps'),initial_geometry_passed=True,mean_chunk_latency=float(np.mean(lat)),peak_GPU_allocated=torch.cuda.max_memory_allocated(),action_min=np.min(executed,axis=0).tolist(),action_max=np.max(executed,axis=0).tolist())
 except Exception:result.update(status='INVALID',failure_reason=traceback.format_exc(),failure_stage=stage,invalid_action=stage=='action',simulator_exception=stage in ['simulation','reset'])
 finally:
  if writer:writer.close()
  if r:r.close()
  if env:env.close()
  np.savez(out/'trajectory.npz',executed_actions=np.asarray(executed),predicted_chunks=np.asarray(chunks),inference_seconds=np.asarray(lat),sim_states=np.asarray(states),current_FK_proprio=np.asarray(proprios));result['seconds']=time.time()-t;js(out/'result.json',result)
 results.append(result)
 keys=list(dict.fromkeys(k for x in results for k in x))
 with (O/'per_trial_results.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(results)
 print('TRIAL',ix+1,160,row['trial_id'],result['status'],result['success'],result['policy_steps'],round(result['seconds'],2),flush=True)
 if result['status']!='COMPLETE':raise RuntimeError(result['failure_reason'])
assert ecc()==base_ecc
for arm,(m,h,p,proc,before) in models.items():assert [digest(x) for x in [m,h,p]]==before
js(O/'rollout_acceptance.json',{'status':'PASS','completed_trials':160,'optimizer_updates':0,'backward_calls':0,'parameters_unchanged':True,'ECC_unchanged':True,'GPU_UUID':UUID,'seconds':time.time()-start,'peak_GPU_allocated':torch.cuda.max_memory_allocated(),'CPU_peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'manifest_SHA256':sha(O/'evaluation_manifest.csv')})
print('ROLLOUT_COMPLETE',flush=True)
