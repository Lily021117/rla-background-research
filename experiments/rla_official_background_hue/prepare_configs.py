"""Produce explicit, NOT YET TRAINING-ADMITTED experiment specifications.

This entry only prepares configs; it never invokes train.py or loads a model.
"""
import argparse,json
from pathlib import Path
from omegaconf import OmegaConf

def main():
    p=argparse.ArgumentParser();p.add_argument('--official',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    recipe=OmegaConf.to_container(OmegaConf.load(a.official/'rebuttal/exp2_vla_adapter/configs/rla/16x64_libero.yaml'),resolve=True)
    for method in ['AE-Base','AE-Aug','AE-Aug-Cons']:
        spec=dict(method=method,admission='BLOCKED_VERIFIED_BACKGROUND_MASKS',
            official_commit='0fc6e97212442ebccffa06f1d42f5481e5f22950',official_recipe=recipe,
            initialization=dict(description='按官方实现创建的共同未训练初始化',seed=20261102,
                modules=['encoder','decoder'],pretrained_ae_forbidden=True,auto_resume=False,
                shared_step0_path='REQUIRED_NEW_SHARED_STEP0',shared_step0_sha256=None,
                dino='frozen_pretrained_dinov2_vitl14',image_decoder='visualization_only_not_required'),
            paired=dict(branches=['x','x' if method=='AE-Base' else 'x_prime'],
                augmentation='background_hsv_hue',delta_h=[-.25,.25],augmentation_seed=20261102,
                val_augmentation_seed=20261103,mask_registry=None,mask_whitelist=['table','wall'],
                dedicated_occurrence_key=True,original_rgb_preserved=True,feature_cache_edit_forbidden=True),
            loss=dict(reconstruction='0.5 * sum_over_branches(mean_L1_future_DINO + mean_MSE_future_DINO)',
                consistency='mean_MSE(z_x,z_x_prime)' if method=='AE-Aug-Cons' else None,
                lambda_cons=None if method=='AE-Aug-Cons' else 0,
                reduction='each mean over physical batch, both cameras, DINO patches and channels; latent mean over batch,16,64'),
            fairness=dict(physical_pairs_per_update=128,reconstruction_branches_per_pair=2,
                reconstruction_targets_per_update=256,rgb_views_per_update=1024,
                identical_sampler_and_rng=True,common_update_budget=True),
            budget=dict(proposed_max_updates=20000,validation_every=1000,checkpoint_every=1000,
                status='planning_upper_bound_not_author_actual_training_length',
                physical_transition_exposures=2560000,reconstruction_target_exposures=5120000,
                gpu_hours='UNMEASURED: no model throughput benchmark authorized/executed this round'),
            calibration=dict(train_only=True,disposable_updates_max=100,checkpoints=[0,10,25,50,100],
                rule='On fixed train batches, median of 0.1*||g_rec_encoder||/||g_cons_encoder|| over checkpoints25,50,100; exclude near-zero/nonfinite terms; if fewer than2 usable nodes STOP',
                formal_start='reload common untrained step0 for every group; consistency from first formal update',
                no_sweep=True),
            launch_safety=dict(formal_training_enabled=False,reason='Masks/alignment not admitted; paired trainer end-to-end and shared step0 creation remain gated',
                forbid=['step90000','historical AE weights','automatic latest checkpoint','whole_image_color_jitter']))
        (a.output/f'{method}.json').write_text(json.dumps(spec,indent=2,ensure_ascii=False)+'\n')
    print('Three specifications saved; no training launched.')

if __name__=='__main__':main()
