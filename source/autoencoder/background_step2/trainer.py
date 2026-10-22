"""Cache-boundary adapter. Original AE inference/loss and BasicTrainer updates/save/load are reused."""
import torch
from src.trainers.rla_autoencoder_trainer import RlaAutoencoderTrainer
from src.trainers.basic import BasicTrainer
from .common import *

class IndexDataset(torch.utils.data.Dataset):
 def __len__(self):return 380
 def __getitem__(self,index):return dict(index=torch.tensor(index))

class PairedTrainer(RlaAutoencoderTrainer):
 def __init__(self,nets,group,lam,load_dir=None,step=0,skip_misc=False):
  self.group=group;self.cons_weight=lam;self.lambda_l1=1.;self.lambda_mse=1.;self.lambda_vq=1.
  self.inverse_input_mode='sub';self.use_vq=False;self.decoder_only_mode=False
  self.rgb_key='features';self.last_output=None
  BasicTrainer.__init__(self,nets,IndexDataset(),output_dir=str(OUT/group),load_dir=load_dir,step=step,
    max_steps=200,batch_size=4,batch_split=4,optimizer=dict(name='AdamW',args=dict(lr=1e-4,weight_decay=0.)),
    ema_rate=[],fp16_mode='inflat_all',log_scale=8.,fp16_scale_growth=.001,num_workers=0,
    prefetch_data=False,save_latest_only=False,skip_load_misc=skip_misc,load_strict=True,skip_initial_snapshot=True,
    i_sample=1000000,i_save=50,freeze_models=[],wandb_run=None)
  assert self.dataloader.num_workers==0 and not self.dataloader.persistent_workers
  self.optimizer_calls=0
  original=self.optimizer.step
  def counted(*args,**kwargs):
   ret=original(*args,**kwargs);self.optimizer_calls+=1;return ret
  self.optimizer.step=counted
 def _get_masked_rgb_sequence(self,data):
  # Upstream frozen DINO cache includes the declared preprocessing/support.
  return data['features']
 def _extract_dino_tokens_sequence(self,features):
  assert features.shape[1:]==(2,1,1024,1024)
  return features,(32,32)
 def inference_batch(self,models,data,training=False):
  out=RlaAutoencoderTrainer.inference_batch(self,models,data,training=training)
  self.last_output=out
  return out
 def training_losses(self,**args):
  parts=[];latent=[]
  for v in range(2):
   terms,_=RlaAutoencoderTrainer.training_losses(self,features=args['features'][:,v])
   parts.append(terms);latent.append(self.last_output['enc_tokens'])
  cons=(latent[0]-latent[1]).square().mean()
  rec=parts[0]['loss']+parts[1]['loss']
  terms=dict(loss=rec+self.cons_weight*cons,reconstruction=rec,consistency=cons,
   clean_l1=parts[0]['l1'],clean_mse=parts[0]['mse'],background_l1=parts[1]['l1'],background_mse=parts[1]['mse'])
  assert all(torch.isfinite(x).all() for x in terms.values()),'Nonfinite loss: stop'
  return terms,{}

def make(group,lam,resume_step=None):
 torch.manual_seed(20261020)
 return PairedTrainer(models(),group,lam,load_dir=str(OUT/group) if resume_step is not None else None,
                       step=resume_step or 0,skip_misc=False)

def digest_optimizer(opt):
 import hashlib
 h=hashlib.sha256()
 for i,s in sorted(opt.state_dict()['state'].items()):
  for key,val in sorted(s.items()):
   h.update(f'{i}:{key}'.encode())
   if torch.is_tensor(val):
    for chunk in val.detach().reshape(-1).split(4*1024**2):h.update(chunk.cpu().contiguous().numpy().tobytes())
   else:h.update(str(val).encode())
 return h.hexdigest()

def update(trainer,indices):
 start=time.monotonic();calls=trainer.optimizer_calls
 data=[batch([i]) for i in indices]
 for m in trainer.models.values():m.train()
 metrics=trainer.run_step(data)
 assert trainer.optimizer_calls==calls+1,'Optimizer skipped (overflow/nonfinite): preflight/formal run must stop'
 for name,m in trainer.models.items():
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()),('Invalid gradients',name)
  assert all(torch.isfinite(p).all() for p in m.parameters()),('Nonfinite parameters',name)
 trainer.step+=1
 log=dict(step=trainer.step,successful_updates=trainer.step,attempts=trainer.step,overflow_skips=0,OOM=0,
  view_pair_batch=len(indices),branch_samples=2*len(indices),sampling_ids=indices,loss=metrics['loss'],
  learning_rate=trainer.optimizer.param_groups[0]['lr'],elapsed_seconds=time.monotonic()-start,
  peak_gpu_bytes=torch.cuda.max_memory_allocated(),**resources())
 append(OUT/trainer.group/'train.jsonl',log)
 print('UPDATE',trainer.group,trainer.step,log['loss'],flush=True)
 if trainer.step%5==0:health()
 return log

def save(trainer):
 trainer.save()
 write(OUT/trainer.group/f'progress_step{trainer.step:07d}.json',dict(step=trainer.step,
  schedule_sha256=sha(OUT/'actual_sampling_sequence.json'),sampler='explicit immutable source-ID list; cursor=successful step',
  attempts=trainer.step,overflow_skips=0,discarded_or_replayed_updates=0,
  parameter_hashes={k:model_hash(m) for k,m in trainer.models.items()}))

@torch.inference_mode()
def evaluate(trainer,step):
 rs=read(OUT/'validation_records.json');rows=[];lat=[]
 for m in trainer.models.values():m.eval()
 for index,r in enumerate(rs):
  outputs=[]
  for v in ('clean','background'):
   a=np.load(r['cache'][v],mmap_mode='r');x=torch.from_numpy(a[[r['current_cache_index'],r['future_cache_index']]].copy()).float().cuda()
   out=trainer.inference_batch(trainer.models,dict(features=x.unsqueeze(0).unsqueeze(2)))
   pred=out['x1_pred'].float();target=out['x1_gt'].float();z=out['enc_tokens'].float()
   assert torch.isfinite(pred).all() and torch.isfinite(z).all()
   outputs.append(dict(z=z.flatten(),l1=float((pred-target).abs().mean()),mse=float((pred-target).square().mean())))
  a,b=outputs;na=float(a['z'].norm());nb=float(b['z'].norm());d=a['z']-b['z']
  rows.append(dict(episode_id=r['episode_id'],task=r['task'],camera=r['camera'],window=r['window'],
   output_t=r['output_t'],output_future=r['output_future'],clean_l1=a['l1'],clean_mse=a['mse'],background_l1=b['l1'],background_mse=b['mse'],
   latent_mse=float(d.square().mean()),relative_l2=float(d.norm()/(a['z'].norm()+1e-12)),
   cosine=float(torch.nn.functional.cosine_similarity(a['z'],b['z'],dim=0)),clean_norm=na,background_norm=nb))
  lat.append(torch.stack([a['z'],b['z']]).cpu().numpy())
 keys=[k for k in rows[0] if isinstance(rows[0][k],float)]
 # First aggregate each episode/camera, then equal-weight episodes/cameras.
 per_demo=[]
 for eid in sorted({r['episode_id'] for r in rows}):
  group=[r for r in rows if r['episode_id']==eid]
  per_demo.append(dict(episode_id=eid,**{k:float(np.mean([r[k] for r in group])) for k in keys}))
 summary={k:float(np.mean([r[k] for r in per_demo])) for k in keys}
 summary['latent_per_dim_variance']=float(np.stack(lat).var(axis=0).mean())
 summary['reconstruction_sum']=summary['clean_l1']+summary['clean_mse']+summary['background_l1']+summary['background_mse']
 write(OUT/trainer.group/f'validation_step{step:07d}.json',dict(step=step,rows=rows,per_demo=per_demo,summary=summary,independent_episodes=4,view_pairs=80,
  reconstruction_scope='true-future AE reconstruction, not forward prediction',reserved_test_scored=False))
 print('VALIDATION',trainer.group,step,summary,flush=True)
 return summary
