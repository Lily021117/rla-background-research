"""Frozen official DINO features; bounded per-episode extraction."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from common import *
p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args();a.run.mkdir(exist_ok=False)
manifest=json.loads((a.dataset/'frozen_dataset_manifest.json').read_text());assert manifest['status']=='PASS' and sum(e['split']=='train' for e in manifest['episodes'])==20
save(a.run/'data_manifest.json',manifest);save(a.run/'source_provenance.json',dict(data_manifest_sha256=sha(a.dataset/'frozen_dataset_manifest.json'),official_commit='0fc6e97212442ebccffa06f1d42f5481e5f22950',DINO='frozen official DINOv3 ViT-L/16 local verified checkpoint',resolution=224,camera_order=['agentview','robot0_eye_in_hand'],batch_frames=4,feature_storage='float32, no half cast',policy_updates=0,policy_rollouts=0))
before=ecc();(a.run/'cache_ecc_before.txt').write_text(before);ext,init=load(a.project);inventory=[];start=time.time()
save(a.run/'official_initialization.json',dict(config=str(init/'config.yaml'),config_sha256=sha(init/'config.yaml'),encoder=str(ext.encoder_ckpt),encoder_sha256=sha(ext.encoder_ckpt),decoder=str(init/'ckpts/decoder_step0090000.pt'),decoder_sha256=sha(init/'ckpts/decoder_step0090000.pt'),step=90000,not_paper_step40000=True))
with torch.inference_mode():
    for e in manifest['episodes']:
        root=Path(e['root']);lab=np.load(root/'trajectory_labels.npz');out=a.run/'features'/e['episode'];out.mkdir(parents=True)
        for c in ('clean','background'):
            rgb=np.stack([np.load(root/'frames'/f'{t:04}.npz')[c] for t in range(e['frames'])]);tokens,hw=ext.episode_tokens(rgb,batch_frames=4);assert tokens.dtype==torch.float32 and torch.isfinite(tokens).all();path=out/(c+'.npy');np.save(path,tokens.cpu().numpy())
            if c=='clean':
                r=ext.episode_plucker(lab['K'].astype(np.float32),lab['c2w'].astype(np.float32),hw,(224,224));np.save(out/'plucker.npy',r.cpu().numpy())
            if not inventory:
                repeat,_=ext.episode_tokens(rgb[:4],batch_frames=4);assert torch.equal(tokens[:4],repeat);save(a.run/'DINO_repeat_check.json',dict(max_abs_difference=0.,shape=list(tokens.shape),dtype=str(tokens.dtype)))
            inventory.append(dict(episode=e['episode'],split=e['split'],appearance=c,frames=e['frames'],shape=list(tokens.shape),dtype=str(tokens.dtype),file=str(path),sha256=sha(path),source_root=str(root)))
            del tokens,rgb
        assert counters(before)==counters(ecc());print('CACHE',e['episode'],len(inventory),time.time()-start,flush=True)
save(a.run/'feature_manifest.json',inventory);save(a.run/'feature_acceptance.json',dict(status='PASS',episodes=25,appearance_sequences=50,RGB_images=sum(e['frames']*4 for e in manifest['episodes']),ECC_increment=False,peak_allocated=torch.cuda.max_memory_allocated(),seconds=time.time()-start))
