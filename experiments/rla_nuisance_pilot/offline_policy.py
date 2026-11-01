"""Deployment prediction on identical saved observations, with no target in inputs."""
import random,hashlib
import numpy as np,torch
from PIL import Image
from rla_nuisance_pilot.common import *
from rla_low_lr_continue.evaluation import initialize
def execute(g):
 from coverage_gripper.dataset import DerivedEpisodeDataset
 from rla.data import RlaCollator
 plan,_,_=initialize(g);mods=g['mods'];model,head,prop=mods['vla'],mods['head'],mods['proprio']
 ds=DerivedEpisodeDataset(plan['inputs']['data_manifest'],plan['inputs']['action_proprio_statistics'],g['transform'],RlaCollator(2048,g['tokenizer'].pad_token_id));stats=read(plan['inputs']['action_proprio_statistics']);model.base_model.model.norm_stats={'derived':stats}
 registry=read(ROOT/'evaluation/policy_registry.json');inventory=read(ROOT/'checkpoint_inventory.json');anchors=read(ROOT/'offline_anchors.json');conditions=['clean']+[c['id'] for c in read(ROOT/'conditions.json')];records=[];capture={}
 original=head.predict_action
 def predict(*args,**kw):
  v=original(*args,**kw);capture['normalized_actions']=v.detach().float().cpu().numpy();return v
 head.predict_action=predict
 for key,path in registry.items():
  assert sha(path)==inventory[key]['sha256'];ck=torch.load(path,map_location='cpu');assert ck['global_step']==1800
  for k,m in mods.items():
   state=m.state_dict();assert set(ck['parameters'][k]).issubset(state);state.update(ck['parameters'][k]);m.load_state_dict(state,strict=True);m.eval().requires_grad_(False)
  del ck;arm='C' if key.startswith('C-') else 'D';n=read(BASE/f'targets/{arm}/statistics.json');mu=np.array(n['mean'],np.float32);scale=n['global_scale']
  for a in anchors:
   episode,t=a['episode'],a['t'];target_action=ds.sample(episode,t,'clean')['actions'];target_action=np.asarray(target_action).reshape(8,7).astype(np.float32);clean=None
   for condition in conditions:
    inp=np.load(ROOT/'offline_images'/episode/condition/f'{t:04}.npz');images=inp['rgb'][0].transpose(0,2,3,1);p=inp['proprio'];prompt=g['QwenPromptBuilder']('openvla');prompt.add_turn('human',f'What action should the robot take to {a["task"]}?');prompt.add_turn('gpt','');tokens=g['tokenizer'](prompt.get_prompt(),add_special_tokens=True).input_ids;del tokens[-3:]
    batch=dict(input_ids=torch.tensor([tokens],device='cuda'),attention_mask=torch.ones((1,len(tokens)),device='cuda',dtype=torch.bool),pixel_values=torch.cat([g['imageproc'](Image.fromarray(im),return_tensors='pt')['pixel_values'] for im in images],dim=1).to('cuda',dtype=torch.bfloat16))
    ps=stats['proprio'];lo=np.array(ps['q01']);hi=np.array(ps['q99']);p=np.clip(2*(p-lo)/(hi-lo+1e-8)-1,-1,1);p=np.where(np.array(ps['min'])==np.array(ps['max']),0,p)
    seed=int(rank(episode+':'+str(t))[:8],16);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():actions,_=model.predict_action(**batch,unnorm_key='derived',proprio=p,proprio_projector=prop,action_head=head,use_film=False);z=head.pop_z_hat().float().cpu().numpy().reshape(8,16,64)
    action=capture.pop('normalized_actions').reshape(8,7);actions=np.asarray(actions);assert np.isfinite(z).all() and np.isfinite(action).all() and actions.shape==(8,7)
    label=np.load(ROOT/'offline_targets'/arm/episode/condition/f'{t:04}.npy');yn=(label-mu)/scale;rawz=z*scale+mu
    if condition=='clean':clean=(action.copy(),actions.copy(),z.copy(),rawz.copy(),label.copy(),yn.copy())
    for h in range(8):
     row=dict(model=key,episode=episode,t=t,appearance=condition,horizon=h+1,current_action=h==0,executed_chunk_member=True,AE_raw_norm=float(np.linalg.norm(label[h])),AE_normalized_norm=float(np.linalg.norm(yn[h])),AE_raw_MSE_vs_clean=float(np.mean((label[h]-clean[4][h])**2)),AE_standardized_MSE_vs_clean=float(np.mean((yn[h]-clean[5][h])**2)),AE_standardized_cosine_vs_clean=float(np.sum(yn[h]*clean[5][h])/(np.linalg.norm(yn[h])*np.linalg.norm(clean[5][h]))) if min(np.linalg.norm(yn[h]),np.linalg.norm(clean[5][h]))>1e-12 else None,policy_raw_latent_norm=float(np.linalg.norm(rawz[h])),policy_standardized_latent_MSE_vs_clean=float(np.mean((z[h]-clean[2][h])**2)),policy_raw_latent_MSE_vs_clean=float(np.mean((rawz[h]-clean[3][h])**2)),policy_latent_L1_own_target=float(np.mean(abs(z[h]-yn[h]))),policy_latent_raw_L1_own_target=float(np.mean(abs(rawz[h]-label[h]))))
     for name,sl in [('translation',slice(0,3)),('rotation',slice(3,6)),('gripper',slice(6,7))]:
      row[name+'_normalized_L1_demonstration']=float(np.mean(abs(action[h,sl]-target_action[h,sl])));row[name+'_normalized_L1_vs_clean']=float(np.mean(abs(action[h,sl]-clean[0][h,sl])));row[name+'_decoded_L1_vs_clean']=float(np.mean(abs(actions[h,sl]-clean[1][h,sl])))
     records.append(row)
    dest=ROOT/'offline_predictions'/key/episode/condition;dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/f'{t:04}.npz',action_normalized=action,action_decoded=actions,latent_normalized=z,action_labels=target_action)
   assert g['counts'](g['before'])==g['counts'](g['ecc']())
  print('OFFLINE_POLICY',key,'PASS',flush=True)
 table(ROOT/'offline_metrics.csv',records);save(ROOT/'offline_policy_acceptance.json',dict(status='PASS',policy_forward_calls=6*100*7,models=6,anchors=100,optimizer_updates=0,AE_updates=0,DINO_updates=0,backward=0,target_in_policy_inputs=False,random_conditions='same explicit Python/NumPy/torch/CUDA seed per anchor; eval inference; all appearances/models reset',action_units='normalized common training coordinates; decoded controller command units, not measured pose displacement or angle; gripper continuous before deployment threshold',execution='chunk8 executed open loop until success/done; all8 eligible, first entry additionally reported; no alternative horizon deployment'))
