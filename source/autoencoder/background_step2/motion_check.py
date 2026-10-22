"""Final frozen validation-only motion discrimination context, no fitted neural/physical head."""
import itertools
import torch
from scipy.stats import spearmanr
from .common import *

def main():
 assert read(OUT/'training_complete.json')['status']=='PASS'
 gpu_gate()
 splits=read(OUT/'split_manifest.json')['splits'];labels={};train=[]
 for split in ('train','validation'):
  for e in splits[split]:
   g=np.load(e['geometry_path']);names=list(g['body_names']);p=g['body_positions']
   idx=[names.index(n) for n in ('gripper0_eef','gripper0_finger_joint1_tip','gripper0_finger_joint2_tip','akita_black_bowl_1_main','akita_black_bowl_2_main')]
   for w in e['windows']:
    t,u=w['output_t'],w['output_future'];width=np.linalg.norm(p[[t,u],idx[1]]-p[[t,u],idx[2]],axis=1)
    y=np.concatenate([p[u,idx[0]]-p[t,idx[0]],[width[1]-width[0]],p[u,idx[3]]-p[t,idx[3]],p[u,idx[4]]-p[t,idx[4]]])
    labels[(e['episode_id'],w['window'])]=y
    if split=='train':train.append(y)
 train=np.stack(train);scale=train.std(0);scale[scale<1e-6]=1.
 thresholds={}
 for name,amp in (('gripper',np.abs(train[:,3])),('named_bowl1',np.linalg.norm(train[:,4:7],axis=1)),('named_bowl2',np.linalg.norm(train[:,7:10],axis=1))):
  nz=amp[amp>1e-6];thresholds[name]=dict(near_zero_m=1e-6,large_motion_m=float(np.quantile(nz,.75)) if len(nz) else None,train_nonzero_count=len(nz),train_windows=len(train))
 write(OUT/'motion_thresholds_train_only.json',dict(thresholds=thresholds,label_scale=scale.tolist(),definition='World body displacement and actual fingertip-separation change; both explicitly named bowls, no guessed task-target identity. No new head or physical supervision.'))
 records=read(OUT/'validation_records.json');results={}
 with torch.inference_mode():
  for group in ('A','B','C'):
   nets=models()
   if group!='A':
    for name,m in nets.items():m.load_state_dict(torch.load(OUT/group/'ckpts'/f'{name}_step0000200.pt',map_location='cpu',weights_only=True),strict=True)
   for m in nets.values():m.eval().requires_grad_(False)
   before={k:model_hash(m) for k,m in nets.items()};zs=[]
   for r in records:
    views=[]
    for v in ('clean','background'):
     a=np.load(r['cache'][v],mmap_mode='r');res=torch.from_numpy((a[r['future_cache_index']].astype(np.float32)-a[r['current_cache_index']].astype(np.float32)).copy()).unsqueeze(0).cuda()
     z,_=nets['encoder'](res);assert torch.isfinite(z).all();views.append(z[0].cpu().numpy().ravel())
    zs.append(np.stack(views))
   zs=np.stack(zs);np.save(OUT/f'validation_latents_{group}.npy',zs)
   rows=[]
   for eid in sorted({r['episode_id'] for r in records}):
    for camera in ('agentview','wrist'):
     ids=[i for i,r in enumerate(records) if r['episode_id']==eid and r['camera']==camera]
     pd=[];zd=[];gr=[];b1=[];b2=[]
     for i,j in itertools.combinations(ids,2):
      yi=labels[(eid,records[i]['window'])];yj=labels[(eid,records[j]['window'])]
      pd.append(float(np.linalg.norm((yi-yj)/scale)));zd.append(float(np.linalg.norm(zs[i,0]-zs[j,0])))
      gr.append(float(abs(yi[3]-yj[3])));b1.append(float(np.linalg.norm(yi[4:7]-yj[4:7])));b2.append(float(np.linalg.norm(yi[7:]-yj[7:])))
     def corr(x):
      return float(spearmanr(x,zd).statistic) if np.std(x)>1e-12 and np.std(zd)>1e-12 else None
     rows.append(dict(episode_id=eid,camera=camera,within_demo_pairs=len(pd),
       motion_distance_latent_distance_spearman=corr(pd),gripper_change_difference_latent_distance_spearman=corr(gr),
       named_bowl1_change_difference_latent_distance_spearman=corr(b1),named_bowl2_change_difference_latent_distance_spearman=corr(b2),
       clean_latent_per_dim_variance=float(zs[ids,0].var(0).mean())))
   results[group]=dict(per_episode_camera=rows,overall_clean_latent_per_dim_variance=float(zs[:,0].var(0).mean()),
    interpretation='Descriptive physical-motion discrimination association only; not physical MAE, task semantics or control accuracy.',
    frozen_hashes_before=before,frozen_hashes_after={k:model_hash(m) for k,m in nets.items()})
   assert results[group]['frozen_hashes_after']==before
   del nets;torch.cuda.empty_cache();health()
 write(OUT/'motion_fidelity_context.json',dict(groups=results,reserved_test_scored=False,optimization_steps=0,validation_only=True))
if __name__=='__main__':main()
