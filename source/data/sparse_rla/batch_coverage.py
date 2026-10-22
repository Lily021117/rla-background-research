"""CPU-only coverage of all30 source-aligned windows through actual VLA transforms."""
import os,json,csv,hashlib
from pathlib import Path
import tensorflow as tf
tf.config.set_visible_devices([],'GPU')
import torch
from transformers import AutoConfig,AutoImageProcessor,AutoProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor,PrismaticProcessor
from sparse_rla.adapter import make_loader
R=Path('${RLA_PROJECT_ROOT}');OUT=Path(os.environ['RUN_DIR']);DATA=Path(os.environ['SPARSE_DATA_DIR'])
AutoConfig.register('openvla',OpenVLAConfig);AutoImageProcessor.register(OpenVLAConfig,PrismaticImageProcessor);AutoProcessor.register(OpenVLAConfig,PrismaticProcessor)
processor=AutoProcessor.from_pretrained(OUT/'runtime_metadata',trust_remote_code=False,local_files_only=True)
stats=json.loads((OUT/'runtime_metadata/dataset_statistics.json').read_text())['libero_spatial_no_noops']
rows=json.loads((DATA/'samples.json').read_text());result=[]
for group in ['B','C']:
 loader,ds=make_loader(DATA,group,processor,stats)
 for i,batch in enumerate(loader):
  r=rows[i];assert batch['actions'].shape==(1,8,7) and batch['rla'].shape==(1,1,2048) and batch['proprio'].shape==(1,8)
  assert all(torch.isfinite(v).all() for v in batch.values() if torch.is_tensor(v))
  assert torch.equal(batch['rla'][0],torch.from_numpy(ds.store.frame(r['slot'],r['current_video_index'])))
  result.append(dict(model=group,episode_id=r['episode_id'],window=r['window'],source_state=r['current_source_state'],
   future_state=r['future_source_state'],action_lo=r['action_source_lo'],action_hi=r['action_source_hi_exclusive'],
   actions_shape='1x8x7',proprio_shape='1x8',latent_target_shape='1x1x2048',finite=True,real_transform_and_collator=True))
 assert i+1==30
with (OUT/'batch_coverage.csv').open('x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
p=OUT/'batch_coverage_acceptance.json';assert not p.exists();p.write_text(json.dumps(dict(status='PASS',
 models=['B','C'],unique_physical_windows=30,actual_batch_constructions=60,episodes=3,gradient_batches_not_included_here=0,
 optimizer_updates=0,GPU_inference=0,legacy_singleton_proprio_fixed_by_isolated_collator=True),indent=2))
prep=json.loads((OUT/'runtime_metadata/preprocessor_config.json').read_text())
p=OUT/'preprocessing_schema.json';assert not p.exists();p.write_text(json.dumps(dict(policy_image_processor=prep,
 camera_order=['agentview','wrist'],native_replay_RGB_shape=[128,128,3],orientation='rotate180 once before official image transform',
 policy_input_channels='2 cameras x 2 fused vision normalization streams x 3RGB =12',
 proprio='FK world eef grip-site position3, official robot0_right_hand quaternion to axisangle3, actual finger-joint qpos2',
 teacher='DINOv3 ViT-L16 at512px, original Step2 cached preprocessing and full32x64 token ordering',
 action_normalization='actual libero_dataset_transform then published Pro bounds_q99 normalizer and action mask',
 normalization_statistics_source=str(OUT/'runtime_metadata/dataset_statistics.json'),
 legacy_RLA_STEPS_auto_patch=0,explicit_sparse_target_horizons=[8],explicit_sparse_target_count=1,action_chunk_count=8),indent=2))
print('REAL_BATCH_COVERAGE_PASS',len(result),flush=True)
