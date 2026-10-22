"""One actual Pro VLA batch, action loss and auxiliary gradients; ZERO updates."""
import os,json,hashlib,time,gc,shutil,subprocess,importlib.util,sys,resource,ctypes,uuid
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([],'GPU')
import torch
from experiments.robot.libero.run_libero_eval import GenerateConfig,initialize_model
from sparse_rla.adapter import SparseHead,make_loader
R=Path('${RLA_PROJECT_ROOT}');OUT=Path(os.environ['RUN_DIR']);UUID='${GPU_UUID}'
DATA=Path(os.environ.get('SPARSE_DATA_DIR',str(OUT)))
def read(p):return json.loads(Path(p).read_text())
def write(n,v):
 p=OUT/n;assert not p.exists(),p;p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def digest(m):
 h=hashlib.sha256()
 for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def ecc():
 raw=subprocess.check_output(['nvidia-smi','-i',UUID,'-q','-d','ECC'],text=True)
 return [s.strip() for s in raw.splitlines() if any(k in s for k in ['Correctable','Uncorrectable','Threshold Exceeded','Repair Pending','Unrepairable'])]
def main():
 assert read(DATA/'preparation.json')['status']=='PASS'
 if DATA!=OUT:
  for name in ['data_manifest.csv','samples.json','episodes.json','preparation.json','protected_checkpoints_before.json','protected_sources_before.json','git_before.json']:
   assert not (OUT/name).exists();shutil.copy2(DATA/name,OUT/name)
  write('accepted_data_source.json',dict(path=str(DATA),read_only=True,reused_after_singleton_batch_fix=True))
 assert os.environ['CUDA_VISIBLE_DEVICES']==UUID and torch.cuda.device_count()==1
 p=torch.cuda.get_device_properties(0)
 class CUuuid(ctypes.Structure):_fields_=[('bytes',ctypes.c_ubyte*16)]
 lib=ctypes.CDLL('libcuda.so.1');dev=ctypes.c_int();identity=CUuuid()
 assert lib.cuInit(0)==0 and lib.cuDeviceGet(ctypes.byref(dev),0)==0
 assert lib.cuDeviceGetUuid(ctypes.byref(identity),dev)==0
 actual_uuid='GPU-'+str(uuid.UUID(bytes=bytes(identity.bytes)));assert actual_uuid==UUID
 free,total=torch.cuda.mem_get_info();assert free>=26*1024**3,(free,'requires26GiB startup')
 torch.cuda.set_per_process_memory_fraction(min(22*1024**3,free-4*1024**3)/total)
 before_ecc=ecc();write('gpu_before.json',dict(uuid=actual_uuid,free_bytes=free,total_bytes=total,ecc=before_ecc))
 torch.set_num_threads(4);torch.manual_seed(20261021)
 source=R/'artifacts/rla_policy_closed_loop_pilot/runtime_checkpoint';runtime=OUT/'runtime_metadata';runtime.mkdir()
 for f in source.iterdir():
  if not f.is_file():continue
  if f.suffix in ['.safetensors','.pt']:(runtime/f.name).symlink_to(f.resolve())
  else:shutil.copy2(f,runtime/f.name)
 cfg=GenerateConfig(pretrained_checkpoint=str(runtime),use_pro_version=True,use_minivlm=True,use_wandb=False)
 model,vanilla,proprio,_,processor=initialize_model(cfg)
 for x in [model,vanilla,proprio]:x.eval().requires_grad_(False)
 model_before=digest(model);vanilla_before=digest(vanilla);proprio_before=digest(proprio)
 shared=[(n,p) for n,p in model.named_parameters() if 'action_queries' in n];assert len(shared)==1
 shared[0][1].requires_grad_(True)
 torch.manual_seed(20261021);head=SparseHead(model.llm_dim).to(torch.bfloat16).cuda()
 state=head.state_dict();old=vanilla.state_dict();broadcast=[]
 for k,v in old.items():
  assert k in state
  if state[k].shape!=v.shape:
   assert k.endswith('gating_factor') and v.numel()==1 and state[k].shape==(40,1)
   state[k]=v.expand_as(state[k]).clone();broadcast.append(k)
  else:state[k]=v
 assert len(broadcast)==24;head.load_state_dict(state,strict=True);del state,old
 before_head=digest(head);captured={}
 def capture(module,args,output):captured['actions']=output.detach().float().cpu().clone()
 hook=head.model.register_forward_hook(capture)
 stats=read(source/'dataset_statistics.json')['libero_spatial_no_noops']
 spec=importlib.util.spec_from_file_location('actual_finetune',Path.cwd()/'vla-scripts/finetune.py');ft=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ft;spec.loader.exec_module(ft)
 npatches=model.vision_backbone.get_num_patches()*model.vision_backbone.get_num_images_in_input()
 def forward(batch,h):
  return ft.run_forward_pass(vla=model,action_head=SimpleNamespace(module=h),proprio_projector=proprio,
   batch=batch,action_tokenizer=dataset.action_tokenizer,device_id=0,use_l1_regression=True,use_proprio=True,
   use_film=False,num_patches=npatches,use_pro_version=True,cfg=SimpleNamespace(phase='Training'))
 def rss_kib():return int(next(x for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:')).split()[1])
 rss=[dict(phase='model_and_heads_loaded',rss_kib=rss_kib())]
 results=[];torch.cuda.reset_peak_memory_stats();start=time.monotonic()
 for group in ['B','C']:
  loader,dataset=make_loader(DATA,group,processor,stats);batch=next(iter(loader))
  schema={k:dict(shape=list(v.shape),dtype=str(v.dtype),finite=bool(torch.isfinite(v).all())) for k,v in batch.items() if torch.is_tensor(v)}
  assert all(x['finite'] for x in schema.values());assert batch['rla'].shape==(1,1,2048) and batch['actions'].shape==(1,8,7) and batch['proprio'].shape==(1,8)
  if group=='B':
   write('interface_schema.json',dict(actual_batch=schema,teacher_shape=[32,64],target_horizons=[8],
    action_chunk=8,policy_queries=40,structural_projection_parameters=0,alternative_2048_to1024_linear_parameters=2098176,
    actual_module_sources=dict(dataset=__import__('rla.data',fromlist=['x']).__file__,finetune=ft.__file__),
    raw_camera_resolution=[128,128],policy_camera_order=['agentview','wrist'],proprio='world site xyz+official body-quaternion axisangle+two finger qpos',
    control_frequency_hz=20,num_workers=0,RLDS_regenerated_observations_used=False,
    extra_head_parameters=sum(p.numel() for p in head.parameters())-sum(p.numel() for p in vanilla.parameters()),
    output_projection_shape=list(head.model.fc2_rla.weight.shape),legacy_auto_patch_disabled=True,
    initialization='seed20261021; strict copy pretrained shared tensors, broadcast24 scalar gates to40; default Linear initialization for auxiliary output, LayerNorm ones/zeros, bridge zeros'))
   np.savez(OUT/'real_batch.npz',**{k:v.numpy() for k,v in batch.items() if torch.is_tensor(v)})
  cpu_actions=batch['actions'].clone();batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
  if group=='B':
   with torch.no_grad():vanilla_loss,vanilla_metrics=forward(batch,vanilla)
   assert torch.isfinite(vanilla_loss)
   write('vanilla_forward.json',dict(action_loss=float(vanilla_loss),metrics=vanilla_metrics,optimizer_updates=0))
  head.train();torch.manual_seed(20261021)
  for p in list(head.parameters())+[shared[0][1]]:p.grad=None
  action_loss,metrics=forward(batch,head);zhat=head.pop_z_hat();assert zhat.shape==(1,1,2048)
  aux=torch.nn.functional.l1_loss(zhat.float(),batch['rla'].float());total=action_loss+.05*aux
  assert torch.isfinite(total)
  probes=[head.model.fc2_rla.weight,head.model.mlp_resnet_blocks[0].o_proj.weight,shared[0][1]]
  grads=torch.autograd.grad(aux,probes,retain_graph=True,allow_unused=False)
  norms=[float(g.float().norm()) for g in grads];assert all(np.isfinite(norms)) and all(x>0 for x in norms)
  total.backward()
  assert all(torch.isfinite(p.grad).all() for p in list(head.parameters())+[shared[0][1]] if p.grad is not None)
  assert head.model.fc2.weight.grad is not None and head.model.fc2_rla.weight.grad is not None
  assert digest(head)==before_head and ecc()==before_ecc
  results.append(dict(group=group,action_loss=float(action_loss),auxiliary_L1=float(aux),total_loss=float(total),
   auxiliary_gradient_norms=dict(aux_output_head=norms[0],shared_head_trunk=norms[1],original_VLM_action_queries=norms[2]),
   head_unchanged=True,target_normalization='matched3 train-only slot mean / centered global RMS',
   new_projection_parameters=0,optimizer_constructed=False,optimizer_updates=0))
  # Drop the target: upstream policy forward ignores sidecar fields.
  head.eval();base={k:v for k,v in batch.items() if k!='rla'}
  with torch.no_grad():
   torch.manual_seed(111);l0,_=forward(base,head);q0=head.pop_z_hat().clone();a0=captured['actions'].clone()
   poisoned=base|{'rla':torch.full_like(batch['rla'],999)}
   torch.manual_seed(111);l1,_=forward(poisoned,head);q1=head.pop_z_hat();a1=captured['actions'].clone()
  assert torch.equal(l0,l1) and torch.equal(q0,q1) and torch.equal(a0,a1)
  results[-1].update(target_deleted_or_poisoned_policy_max_difference=float((a0-a1).abs().max()),
   target_deleted_or_poisoned_latent_max_difference=float((q0-q1).abs().max()))
  for p in list(head.parameters())+[shared[0][1]]:p.grad=None
  rss.append(dict(phase=group+'_after_checks',rss_kib=rss_kib()))
 write('smoke_results.json',results)
 write('resource_snapshots.json',dict(CPU_RSS_snapshots=rss,B_to_C_RSS_growth_kib=rss[-1]['rss_kib']-rss[-2]['rss_kib'],
  no_oom=True,limited_to_two_gradient_smoke_batches=True))
 assert digest(model)==model_before and digest(vanilla)==vanilla_before and digest(proprio)==proprio_before and digest(head)==before_head
 write('parameter_protection.json',dict(original_VLM_before=model_before,original_VLM_after=digest(model),
  original_action_head_before=vanilla_before,original_action_head_after=digest(vanilla),
  original_proprio_before=proprio_before,original_proprio_after=digest(proprio),
  sparse_head_before=before_head,sparse_head_after=digest(head),all_equal=True,
  original_VLM_trainable_for_gradient_only=shared[0][0],optimizer_constructed=False,optimizer_steps=0))
 write('smoke_acceptance.json',dict(status='PASS',matched_episodes=3,aligned_transitions=30,batch_size=1,
  backward_calls=2,autograd_probe_calls=2,successful_optimizer_updates=0,closed_loop_episodes=0,
  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),CPU_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  seconds=time.monotonic()-start,ecc_unchanged=ecc()==before_ecc,parameters_unchanged=True))
 print('REAL_VLA_SPARSE_H8_32TOKEN_SMOKE_PASS_ZERO_UPDATES',flush=True)
if __name__=='__main__':main()
