"""CPU semantic check on fixed TRAIN inputs; no model/teacher forward or update."""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
plan=json.loads((a.out/'run_plan.json').read_text());project=Path(plan['inputs']['project']);ref=project/'experiments/official_libero_rla_consistency/run_20261023_02/vla-official'
sys.path[:0]=[str(ref),str(project/'worktrees/rla-official-0fc6e97/rebuttal/exp2_vla_adapter/vla_adapter_overlay'),str(Path(__file__).parent.parent/'official_libero_rla_consistency')]
import tensorflow as tf
tf.config.set_visible_devices([],'GPU')
from transformers import AutoTokenizer
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor
from prismatic.models.backbones.llm.prompting import QwenPromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform
from prismatic.vla.constants import NUM_TOKENS,STOP_INDEX,IGNORE_INDEX,ACTION_TOKEN_BEGIN_IDX
from prismatic.training.train_utils import get_current_action_mask,get_next_actions_mask
from rla.data import RlaCollator
from coverage_gripper.dataset import DerivedEpisodeDataset
cfg=ref/'pretrained_models/configs';tokenizer=AutoTokenizer.from_pretrained(str(cfg));processor=PrismaticImageProcessor.from_pretrained(str(cfg));collator=RlaCollator(2048,tokenizer.pad_token_id)
transform=RLDSBatchTransform(ActionTokenizer(tokenizer),tokenizer,lambda img:processor(img,return_tensors='pt')['pixel_values'][0],QwenPromptBuilder,use_wrist_image=True,use_proprio=True,use_minivlm=True)
ds=DerivedEpisodeDataset(plan['train_only_manifest'],plan['inputs']['action_proprio_statistics'],transform,collator)
fixed=[]
for e,t in json.loads(Path(plan['inputs']['B20_sampling_order']).read_text()):
 if (e,t) not in fixed:fixed.append((e,t))
 if len(fixed)==2:break
stats=json.loads(Path(plan['inputs']['action_proprio_statistics']).read_text());rows=[]
for e,t in fixed:
 root,_,meta=ds.episodes[e];assert meta['split']=='train';b=collator([ds.sample(e,t,'clean')]);ids=b['input_ids'];labels=b['labels'];length=ids.shape[1]-NUM_TOKENS-1
 prompt=QwenPromptBuilder('openvla');prompt.add_turn('human',f"What action should the robot take to {meta['task']}?");prompt.add_turn('gpt','');prefix=tokenizer(prompt.get_prompt(),add_special_tokens=True).input_ids;del prefix[-3:]
 prefix=torch.tensor([prefix]);assert torch.equal(ids[:,:length],prefix) and int(ids[0,-1])==STOP_INDEX
 direct_labels=torch.cat([torch.full_like(prefix,IGNORE_INDEX),torch.full((1,NUM_TOKENS),ACTION_TOKEN_BEGIN_IDX+1),torch.tensor([[STOP_INDEX]])],dim=1)
 mask=lambda x:get_current_action_mask(x)|get_next_actions_mask(x)
 assert torch.equal(mask(labels),mask(direct_labels)) and int(mask(labels).sum())==NUM_TOKENS
 # Both paths select the same preceding-token hidden positions, accounting for label shift.
 train_indices=torch.where(mask(labels[:,1:])[0])[0]
 direct_indices=torch.arange(len(prefix[0])-1,len(prefix[0])-1+NUM_TOKENS);assert torch.equal(train_indices,direct_indices)
 with np.load(root/'frames'/f'{t:04}.npz') as frame:
  rgb=frame['clean'].transpose(0,2,3,1);raw=frame['proprio'].copy()
 pixels=torch.cat([processor(Image.fromarray(x),return_tensors='pt')['pixel_values'] for x in rgb],dim=1);assert torch.equal(pixels,b['pixel_values'])
 st=stats['proprio'];lo=np.asarray(st['q01']);hi=np.asarray(st['q99']);direct=np.clip(2*(raw-lo)/(hi-lo+1e-8)-1,-1,1);direct=np.where(np.asarray(st['min'])==np.asarray(st['max']),0,direct)
 trained=b['proprio'].numpy().reshape(8);delta=float(np.max(abs(trained-direct)))
 with np.load(root/'trajectory_labels.npz') as lab:raw_delta=float(np.max(abs(raw-lab['proprio'][t])))
 rows.append(dict(episode=e,timestep=t,split='train',appearance='clean',prompt_prefix_exact=True,prompt_tokens=length,STOP_exact=True,full_action_mask_exact=True,VLM_query_tokens=NUM_TOKENS,continuous_action_elements=8*7,hidden_readout_indices_exact=True,dual_camera_pixels_exact=True,frame_vs_training_label_FK_max_absolute_difference=raw_delta,normalized_proprio_max_absolute_difference=delta,normalization_arithmetic='same equation; float32 TF versus NumPy float64 intermediates'))
assert all(r['frame_vs_training_label_FK_max_absolute_difference']==0 for r in rows)
with (a.out/'teacher_input_equivalence_audit.json').open('x') as f:json.dump(dict(status='PASS_INPUT_AND_INDEX_SEMANTICS',samples=rows,teacher_forward_or_supervision_added=0,GPU_allocations=0,optimizer_updates=0,backward=0,closed_loop=0,sealed_reads=0,actual_deployed_action_bitwise_forward_comparison=False,scope='Input/prompt/mask/readout semantic identity; no claim of bitwise batch8-vs-batch1 forward equality'),f,indent=2)
print('TEACHER_INPUT_SEMANTICS_PASS',rows,flush=True)
