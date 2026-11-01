"""One DINO extraction per saved image batch; frozen AE-C/D inference, no backward."""
import sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_nuisance_pilot.common import *
from official_libero_rla_consistency.ae_consistency.common import load,ecc,counters
def main():
 before=ecc();ext,_=load(P);ext.dino_extractor.eval().requires_grad_(False);records=read(ROOT/'offline_render_inventory.json');inventory=read(ROOT/'encoder_inventory.json');counts=dict(DINO_calls=0,AE_encoder_calls=0,backward=0)
 with torch.inference_mode():
  for row in records:
   data=np.load(row['path']);tokens,hw=ext.episode_tokens(data['rgb'],batch_frames=4);rays=ext.episode_plucker(data['K'],data['c2w'],hw,(224,224));p=ROOT/'offline_features'/row['episode']/row['appearance']/f'{row["t"]:04}.npz';p.parent.mkdir(parents=True,exist_ok=True);np.savez(p,tokens=tokens.float().cpu().numpy(),rays=rays.float().cpu().numpy());counts['DINO_calls']+=3
  for arm in ['C','D']:
   ck=torch.load(inventory[arm]['path'],map_location='cpu',weights_only=False);assert ck['global_step']==2000
   for name in ['encoder','decoder']:ext.models[name].load_state_dict(ck['models'][name],strict=True);ext.models[name].eval().requires_grad_(False)
   from rla_ae_from_scratch.run import digest
   assert digest({k:ext.models[k] for k in ['encoder','decoder']})==read(P/f'experiments/rla_ae_from_scratch/run_20261029_01/{arm}/completion.json')['final_parameter_sha256'];del ck
   for row in records:
    data=np.load(ROOT/'offline_features'/row['episode']/row['appearance']/f'{row["t"]:04}.npz');x=data['tokens'];r=data['rays'];z=ext.models['encoder'](torch.from_numpy(x[1:9]-x[0]).cuda().reshape(8,392,1024),num_views=2,plucker=torch.from_numpy(np.repeat(r[:1],8,axis=0)).cuda())[0].float().cpu().numpy();assert z.shape==(8,16,64) and np.isfinite(z).all()
    path=ROOT/'offline_targets'/arm/row['episode']/row['appearance']/f'{row["t"]:04}.npy';path.parent.mkdir(parents=True,exist_ok=True);np.save(path,z);counts['AE_encoder_calls']+=1
   assert counters(before)==counters(ecc());print('OFFLINE_AE_COMPLETE',arm,flush=True)
 save(ROOT/'offline_ae_acceptance.json',dict(status='PASS',counts=counts,AE_updates=0,DINO_updates=0,horizon='t to t+h h1..8',normalization='existing frozen own AE training statistics; no refitting',temporary_official_constructor_weights='strictly replaced before any AE encoder forward; only DINO extraction precedes replacement'))
if __name__=='__main__':main()
