"""Actual official CPU loader smoke + frozen per-task originals/placeholders.
No guessed segmentation is generated; blocked panels are explicitly labelled.
"""
import argparse, json, sys, hashlib
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from omegaconf import OmegaConf
from paired_hue import PairedHueReader, VerifiedMaskProvider

def main():
    p=argparse.ArgumentParser();p.add_argument('--official',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.official))
    from src.datasets.trajectory_dataset import TrajectoryDataset
    cfg=OmegaConf.to_container(OmegaConf.load(a.official/'rebuttal/exp2_vla_adapter/configs/rla/16x64_libero.yaml'),resolve=True)
    selected=json.loads((a.output/'audit/fixed_visualization_manifest.json').read_text())
    result=[]; ds_by_split={}; loader_counts={}
    for split,key in [('train','dataset'),('val','val_dataset')]:
        args=cfg[key]['args']; args.update(root=str(a.data),use_mp_for_indexing=False,
            trajectory_info_cache_file=str(a.output/f'official_{split}_index.json'))
        ds=TrajectoryDataset(**args);ds_by_split[split]=ds
        loader_counts[split]={'trajectories':len(ds.trajectory_info),'valid_starts':ds.total_valid_starts,
            'per_suite':{s:sum(t['group_name']==s for t in ds.trajectory_info) for s in args['configs']}}
    # Fixed validation per suite to prove actual split/reading, not train-only smoke.
    ds=ds_by_split['val']
    for s in cfg['val_dataset']['args']['configs']:
        t=next(t for t in ds.trajectory_info if t['group_name']==s)
        selected.append(dict(suite=s,trajectory_id=t['traj_id'],frames=[0,9],horizon=10,cameras=cfg['dataset']['args']['cameras'],split='val'))
    panels=a.output/'visualizations';panels.mkdir(exist_ok=True)
    for i,r in enumerate(selected):
        ds=ds_by_split[r['split']]; t=next(t for t in ds.trajectory_info if t['group_name']==r['suite'] and t['traj_id']==r['trajectory_id'])
        h=dict(version=1,dataset_idx=t['dataset_idx'],group_name=r['suite'],traj_id=r['trajectory_id'],frame_id=r['frames'][0],
            sampled_horizon=r['horizon'],feasible_horizon=r['horizon'],camera_keys=r['cameras'],rgb_variant='base',interaction_frame_indices=None,
            dataset_root=ds.datasets[t['dataset_idx']].root)
        try:
            x=ds.getitem_from_handle(h,apply_augmentation=False)
            disabled=PairedHueReader(ds,VerifiedMaskProvider({}),enabled=False).read(h,0,r['split'])
            assert np.array_equal(x['rgbs'],disabled['x_prime']['rgbs'])
            assert np.array_equal(np.asarray(x['frame_id']),r['frames'])
            try: PairedHueReader(ds,VerifiedMaskProvider({})).read(h,0,r['split'])
            except RuntimeError as e: blocked=str(e)
            else: raise AssertionError('Unverified mask unexpectedly accepted')
            rgb=np.moveaxis(x['rgbs'],2,-1);fg=x['foreground_masks']
            panel=Image.new('RGB',(4*224,4*250),(32,32,32));draw=ImageDraw.Draw(panel)
            for ti in range(2):
                for ci in range(2):
                    y=(ti*2+ci)*250
                    panel.paste(Image.fromarray(rgb[ti,ci]),(0,y+25))
                    panel.paste(Image.fromarray(fg[ti,ci]).convert('RGB'),(224,y+25))
                    draw.text((2,y),f'{r["suite"]}/{r["trajectory_id"]} t={r["frames"][ti]} cam={ci}',fill='white')
                    draw.text((230,y),'PLACEHOLDER, NOT BG',fill='red')
                    draw.text((452,y+80),'AUGMENTATION BLOCKED',fill='red')
                    draw.text((676,y+80),'DIFFERENCE UNAVAILABLE',fill='red')
            panel.save(panels/f'{i:03d}_{r["suite"]}_{r["trajectory_id"]}.png')
            result.append(dict(record=r,loaded=True,rgb_shape=list(rgb.shape),placeholder_unique=np.unique(fg).tolist(),
                disabled_identity=True,real_aug_status='BLOCKED',reason=blocked))
        except Exception as e:
            result.append(dict(record=r,loaded=False,error=f'{type(e).__name__}:{e}'))
        print(i,result[-1]['loaded'],flush=True)
    (a.output/'loader_validation.json').write_text(json.dumps({'loader_counts':loader_counts,'records':result,
        'model_forwards':0,'updates':0,'backward':0},indent=2))

if __name__=='__main__':main()
