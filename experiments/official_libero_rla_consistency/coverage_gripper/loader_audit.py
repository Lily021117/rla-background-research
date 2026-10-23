"""CPU-only real official transform/collator acceptance of the expanded pool."""
import argparse,sys,json,os,hashlib
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np,torch
p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.reference/'vla-official'))
import tensorflow as tf
tf.config.set_visible_devices([],'GPU')
from transformers import AutoTokenizer
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor
from prismatic.vla.datasets import RLDSBatchTransform
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.backbones.llm.prompting import QwenPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.constants import NUM_TOKENS,STOP_INDEX,IGNORE_INDEX
from dataset import DerivedEpisodeDataset
config=a.reference/'vla-official/pretrained_models/configs';tok=AutoTokenizer.from_pretrained(str(config));proc=PrismaticImageProcessor.from_pretrained(str(config))
transform=RLDSBatchTransform(ActionTokenizer(tok),tok,lambda x:proc(x,return_tensors='pt')['pixel_values'][0],QwenPromptBuilder,use_wrist_image=True,use_proprio=True,use_minivlm=True);collator=PaddedCollatorForActionPrediction(2048,tok.pad_token_id)
dataset=DerivedEpisodeDataset(a.out/'frozen_dataset_manifest.json',a.out/'normalization_statistics_expanded_train_only.json',transform,collator);checks=[]
for name,(root,ch,e) in dataset.episodes.items():
    lab=np.load(root/'trajectory_labels.npz');g=lab['actions'][:,6];event=np.flatnonzero((g[:-1]<0)&(g[1:]>0))+1
    for t in sorted(set([0,int(event[0]) if len(event) else 0])):
        keys=[(name,t,c) for c in ('clean','background')];b=dataset.batch(keys)
        assert b['pixel_values'].shape==(2,12,224,224) and b['actions'].shape==(2,8,7) and b['proprio'].shape==(2,8)
        assert all(torch.isfinite(v).all() for v in b.values() if torch.is_tensor(v))
        np.testing.assert_array_equal(b['actions'][0].numpy(),b['actions'][1].numpy());np.testing.assert_array_equal(b['proprio'][0].numpy(),b['proprio'][1].numpy())
        assert (b['input_ids'][:,-1]==STOP_INDEX).all()
        assert (b['labels']!=IGNORE_INDEX).sum().item()==2*(NUM_TOKENS+1)
        assert e['task'] in tok.decode(b['input_ids'][0],skip_special_tokens=False).lower()
        checks.append(dict(episode=name,split=e['split'],timestep=t,action_start=t,action_end_exclusive=t+8,shapes={k:list(v.shape) for k,v in b.items() if torch.is_tensor(v)},paired_labels_equal=True,paired_proprio_equal=True,language_correct=True,STOP_correct=True,RLA_target=False,policy_forward=False))
with (a.out/'expanded_loader_audit.json').open('x') as f:json.dump(dict(status='PASS_OFFICIAL_TRANSFORM_COLLATOR',episodes=len(dataset.episodes),paired_batches=len(checks),checks=checks,optimizer_updates=0,backward=0,policy_inference=0,new_closed_loop=0),f,indent=2)
print('PASS',len(dataset.episodes),'episodes,',len(checks),'paired batches')
