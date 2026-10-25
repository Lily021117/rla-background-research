"""Same fixed demonstration observations, seven dimensions and eight horizons."""
import json,csv,random,os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from stove_aux_weight.training import initialize
def execute(g):
 plan,_,_=initialize(g);root=g['a'].run.parent
 project=Path(plan['inputs']['project'])/'experiments/official_libero_rla_consistency'
 post=project/'stove_postlift/run_20261025_01'
 fixed=json.loads((post/'fixed_demo_samples.json').read_text())
 registry=json.loads((root/os.environ.get('RLA_POLICY_REGISTRY','policy_registry.json')).read_text())
 registry.update({'C':str(project/'run_20261023_10/stove/policy_run/C/isolated_smoke/step0600.pt'),'D-center':str(project/'stove_centering/run_20261024_01/D-center/isolated_smoke/step0600.pt')})
 stats=json.loads(Path(plan['inputs']['action_proprio_statistics']).read_text());g['model'].base_model.model.norm_stats={'derived':stats}
 with (root/'fixed_inference_manifest.json').open('x') as f:json.dump(fixed,f,indent=2)
 with (root/'fixed_action_dimension_metrics.csv').open('x') as f:
  fields=['model','source','split','episode','appearance','t','horizon','dimension','target','prediction','absolute_error']
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for arm,path in registry.items():
   ck=torch.load(path,map_location='cpu');assert ck['global_step']==600
   for k,m in g['mods'].items():
    sd=m.state_dict();assert set(ck['parameters'][k]).issubset(sd);sd.update(ck['parameters'][k]);m.load_state_dict(sd,strict=True);m.eval().requires_grad_(False)
   del ck
   for x in fixed:
    assert x['source']=='demo'
    obs=np.load(Path(x['root'])/'frames'/f"{x['t']:04}.npz")
    actions=np.load(Path(x['root'])/'trajectory_labels.npz')['actions'][x['t']:x['t']+8].astype(float).copy()
    actions[:,:6]=np.clip(actions[:,:6],np.asarray(stats['action']['q01'])[:6],np.asarray(stats['action']['q99'])[:6]);actions[:,6]=1-np.clip(actions[:,6],0,1)
    prompt=g['QwenPromptBuilder']('openvla');prompt.add_turn('human','What action should the robot take to pick up the black bowl on the stove and place it on the plate?');prompt.add_turn('gpt','')
    tokens=g['tokenizer'](prompt.get_prompt(),add_special_tokens=True).input_ids;del tokens[-3:]
    inp=dict(input_ids=torch.tensor([tokens],device='cuda'),attention_mask=torch.ones((1,len(tokens)),dtype=torch.bool,device='cuda'),pixel_values=torch.cat([g['imageproc'](Image.fromarray(im),return_tensors='pt')['pixel_values'] for im in obs[x['appearance']].transpose(0,2,3,1)],dim=1).to('cuda',dtype=torch.bfloat16))
    st=stats['proprio'];lo=np.asarray(st['q01']);hi=np.asarray(st['q99']);p=np.clip(2*(obs['proprio']-lo)/(hi-lo+1e-8)-1,-1,1);p=np.where(np.asarray(st['min'])==np.asarray(st['max']),0,p)
    random.seed(20261023);np.random.seed(20261023);torch.manual_seed(20261023);torch.cuda.manual_seed_all(20261023)
    with torch.inference_mode():pred,_=g['model'].predict_action(**inp,unnorm_key='derived',proprio=p,proprio_projector=g['prop'],action_head=g['head'],use_film=False)
    pred=np.asarray(pred);assert pred.shape==(8,7) and np.isfinite(pred).all()
    for h in range(8):
     for d in range(7):w.writerow(dict(model=arm,source='demo',split=x['split'],episode=x['episode'],appearance=x['appearance'],t=x['t'],horizon=h+1,dimension=d,target=actions[h,d],prediction=pred[h,d],absolute_error=abs(pred[h,d]-actions[h,d])))
   f.flush();print('FIXED_INFERENCE',arm,len(fixed),flush=True)
 g['save']('fixed_inference_acceptance.json',dict(samples_per_model=len(fixed),models=list(registry),optimizer_updates=0,backward=0,closed_loop=0,coordinates='de-normalized delta controller commands; not Cartesian metres; gripper0close1open; target Q99 clipping retained'))
