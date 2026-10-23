"""Frozen checkpoints on every original admitted window; no optimizer/backward."""
import argparse,os,sys,json,csv,hashlib,ctypes,uuid,subprocess,random,time,importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--previous',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out/'checkpoint_diagnosis';out.mkdir()
    sys.path.insert(0,str(a.reference/'vla-official'))
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU');torch.set_num_threads(4)
    from transformers import AutoTokenizer
    from peft import LoraConfig,get_peft_model
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor
    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.projectors import ProprioProjector
    from prismatic.models.backbones.llm.prompting import QwenPromptBuilder
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets import RLDSBatchTransform
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from prismatic.vla.datasets.rlds.traj_transforms import chunk_act_obs
    from prismatic.vla.datasets.rlds.utils.data_utils import normalize_action_and_proprio
    from prismatic.vla.datasets.rlds.oxe.transforms import libero_dataset_transform
    from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE,STOP_INDEX,NUM_TOKENS,IGNORE_INDEX
    def save(name,x):
        with (out/name).open('x') as f:json.dump(x,f,indent=2)
    def sha(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
        return h.hexdigest()
    gpu=os.environ['RLA_GPU_UUID'];before=subprocess.check_output(['nvidia-smi','-i',gpu,'-q','-d','ECC'],text=True);(out/'ecc_before.txt').write_text(before)
    cu=ctypes.CDLL('libcuda.so.1');cu.cuInit(0);dev=ctypes.c_int();assert cu.cuDeviceGet(ctypes.byref(dev),0)==0;buf=(ctypes.c_ubyte*16)();assert cu.cuDeviceGetUuid_v2(ctypes.byref(buf),dev)==0
    assert str(uuid.UUID(bytes=bytes(buf)))==gpu.removeprefix('GPU-') and torch.cuda.mem_get_info()[0]>=26*1024**3
    torch.cuda.set_per_process_memory_fraction(.49);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    seed=20261023;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    configdir=a.reference/'vla-official/pretrained_models/configs';model=OpenVLAForActionPrediction(OpenVLAConfig.from_pretrained(str(configdir)))
    source=torch.load(a.reference/'resolved_prismatic_initialization.pt',map_location='cpu');mapped={}
    renames=[('vision_backbone.dino_featurizer','vision_backbone.featurizer'),('vision_backbone.siglip_featurizer','vision_backbone.fused_featurizer'),('llm_backbone.llm','language_model'),('projector.projector.0','projector.fc1'),('projector.projector.2','projector.fc2'),('projector.projector.4','projector.fc3'),('gamma','scale_factor')]
    for k,x in source.items():
        for old,new in renames:k=k.replace(old,new)
        mapped[k]=x
    expected=model.state_dict();assert set(expected)-set(mapped)=={'action_queries.weight'} and not set(mapped)-set(expected)
    model.load_state_dict({k:mapped.get(k,x) for k,x in expected.items()},strict=True);del source,mapped,expected
    model.vision_backbone.set_num_images_in_input(2);model=get_peft_model(model,LoraConfig(r=64,lora_alpha=128,lora_dropout=0.,target_modules='all-linear',init_lora_weights='gaussian'))
    for n,x in model.named_parameters():
        if 'action_queries' in n:x.requires_grad_(True)
    model=model.to('cuda',dtype=torch.bfloat16)
    prop=ProprioProjector(llm_dim=model.llm_dim,proprio_dim=8).to('cuda',dtype=torch.bfloat16);head=L1RegressionActionHead(input_dim=model.llm_dim,hidden_dim=model.llm_dim,action_dim=7,use_pro_version=True).to('cuda',dtype=torch.bfloat16)
    mods={'vla':model,'proprio':prop,'head':head};trainable_keys={k:{n for n,x in m.named_parameters() if x.requires_grad} for k,m in mods.items()}
    for m in mods.values():m.eval().requires_grad_(False)
    tokenizer=AutoTokenizer.from_pretrained(str(configdir));imageproc=PrismaticImageProcessor.from_pretrained(str(configdir));at=ActionTokenizer(tokenizer)
    transform=RLDSBatchTransform(at,tokenizer,lambda img:imageproc(img,return_tensors='pt')['pixel_values'][0],QwenPromptBuilder,use_wrist_image=True,use_proprio=True,use_minivlm=True);collate=PaddedCollatorForActionPrediction(2048,tokenizer.pad_token_id)
    normpath=a.previous/'normalization_statistics_train_only.json';raw_stats=json.loads(normpath.read_text());stats={k:{kk:tf.convert_to_tensor(vv) for kk,vv in val.items()} for k,val in raw_stats.items()};model.base_model.model.norm_stats={'derived':raw_stats}
    entries=json.loads((a.previous/'episode_manifest.json').read_text());datasets={};fixed=[];pipeline=[]
    for e in entries:
        root=a.previous/'episodes'/e['episode'];lab=np.load(root/'trajectory_labels.npz');actions=lab['actions'].astype(np.float32).copy();actions[:,6]=1-np.clip(actions[:,6],0,1);N=len(actions)
        # Numerically compare the real local action conversion to the official transform.
        check=libero_dataset_transform({'action':tf.constant(lab['actions'].astype(np.float32)),'observation':{'state':tf.constant(lab['proprio'])}})
        np.testing.assert_array_equal(check['action'].numpy(),actions)
        traj={'action':tf.constant(actions),'observation':{'proprio':tf.constant(lab['proprio']),'timestep':tf.range(N)},'task':{'language_instruction':tf.constant([e['task']]*N)},'dataset_name':tf.constant(['single_task_dual224_fk_v3']*N),'absolute_action_mask':tf.constant(np.tile([False]*6+[True],(N,1)))}
        chunk=chunk_act_obs(normalize_action_and_proprio(traj,stats,ACTION_PROPRIO_NORMALIZATION_TYPE),window_size=1,future_action_window_size=7);chunk=tf.nest.map_structure(lambda x:x.numpy(),chunk)
        datasets[e['episode']]=(root,chunk,e)
        for appearance in ('clean','background'):
            for t in range(N-8):fixed.append(dict(episode=e['episode'],split=e['split'],timestep=t,appearance=appearance))
        np.testing.assert_array_equal(chunk['action'][:N-8,:,6],np.stack([actions[t:t+8,6] for t in range(N-8)]))
        pipeline.append(dict(episode=e['episode'],official_transform_equal=True,normalization_preserved_gripper=True,valid_action_chunks=N-8,padded_slots_used=0,gripper_source='HDF5 actions[:,6]',proprio_gripper_separate=True))
    save('fixed_sample_manifest.json',dict(selection='all512 physical windows x2 appearances, fixed before checkpoint scoring',samples=fixed,checkpoint_steps=[0,45,135,270],normalization_SHA256=sha(normpath)));save('batch_pipeline_audit.json',pipeline)
    spec=importlib.util.spec_from_file_location('official_finetune_diagnosis',a.reference/'vla-official/vla-scripts/finetune.py');ft=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ft;spec.loader.exec_module(ft)
    captured={};predict_original=head.predict_action
    def capture(*args,**kwargs):
        x=predict_original(*args,**kwargs);captured['prediction']=x.detach().clone();return x
    head.predict_action=capture
    def batch_for(part):
        items=[]
        for e in part:
            root,ch,entry=datasets[e['episode']];t=e['timestep'];raw=tf.nest.map_structure(lambda x:x[t].copy() if isinstance(x[t],np.ndarray) else x[t],ch)
            with np.load(root/'frames'/f'{t:04}.npz') as d:rgb=d[e['appearance']].transpose(0,2,3,1)
            raw['observation']['image_primary']=rgb[0:1];raw['observation']['image_wrist']=rgb[1:2];it=transform(raw);pl=len(it['input_ids'])-NUM_TOKENS;it['input_ids']=torch.cat([it['input_ids'],torch.tensor([STOP_INDEX])]);it['labels']=it['input_ids'].clone();it['labels'][:pl]=IGNORE_INDEX;items.append(it)
        b=collate(items);b['labels']=b['labels'].to('cuda');b['proprio']=b['proprio'].to('cuda');assert int(model._process_action_masks(b['labels']).sum())==NUM_TOKENS*len(part)
        return b
    def forward(b):
        loss,metrics=ft.run_forward_pass(vla=model,action_head=SimpleNamespace(module=head),proprio_projector=prop,batch=b,action_tokenizer=at,device_id='cuda',use_l1_regression=True,use_proprio=True,use_film=False,num_patches=model.vision_backbone.get_num_patches()*2,use_pro_version=True,cfg=SimpleNamespace(phase='Inference'))
        pred=captured['prediction'].float().cpu().numpy();assert np.isfinite(pred).all() and torch.isfinite(loss);return pred
    inventory={int(r['step']):r for r in csv.DictReader((a.previous/'checkpoint_inventory.csv').open())};metrics=[];deploy=[];started=time.time()
    for step in (0,45,135,270):
        path=a.previous/inventory[step]['path'];assert sha(path)==inventory[step]['sha256'];ck=torch.load(path,map_location='cpu');assert ck['global_step']==step
        for k,m in mods.items():
            assert set(ck['parameters'][k])==trainable_keys[k];state=m.state_dict();state.update(ck['parameters'][k]);m.load_state_dict(state,strict=True)
        del ck,state
        predictions=[];targets=[]
        with torch.inference_mode():
            for start in range(0,len(fixed),8):
                b=batch_for(fixed[start:start+8]);pred=forward(b);predictions.append(pred);targets.append(b['actions'].numpy())
        predictions=np.concatenate(predictions);targets=np.concatenate(targets);np.savez_compressed(out/f'predictions_step{step:04}.npz',prediction=predictions,target=targets)
        for e in entries:
            for appearance in ('clean','background'):
                ix=[i for i,x in enumerate(fixed) if x['episode']==e['episode'] and x['appearance']==appearance];pr=predictions[ix];gt=targets[ix]
                for pos in ('all',*range(8)):
                    x=pr if pos=='all' else pr[:,pos:pos+1];y=gt if pos=='all' else gt[:,pos:pos+1];err=abs(x-y);pg=x[:,:,6];tg=y[:,:,6];closed=tg<.5;opened=tg>.5
                    metrics.append(dict(checkpoint=step,episode=e['episode'],split=e['split'],appearance=appearance,chunk_position=pos,windows=len(ix),translation_L1=float(err[:,:,:3].mean()),rotation_L1=float(err[:,:,3:6].mean()),gripper_L1=float(err[:,:,6].mean()),gripper_prediction_mean=float(pg.mean()),gripper_prediction_std=float(pg.std()),gripper_prediction_min=float(pg.min()),gripper_prediction_max=float(pg.max()),gripper_target_std=float(tg.std()),close_recall=float(np.mean(pg[closed]<.5)) if closed.any() else None,open_recall=float(np.mean(pg[opened]>.5)) if opened.any() else None,neutral_fraction=float(np.mean(pg==.5)),threshold_accuracy=float(np.mean(((pg<.5)&closed)|((pg>.5)&opened)))))
        # Audit actual deployment entry at first frame and first close event,
        # for the first train and development demos, without any environment step.
        if step in (0,270):
            chosen=[next(e for e in entries if e['split']==r) for r in ('train','validation')]
            for e in chosen:
                root,ch,_=datasets[e['episode']];lab=np.load(root/'trajectory_labels.npz');g=lab['actions'][:,6];events=np.flatnonzero((g[:-1]<0)&(g[1:]>0))+1
                for t in sorted(set([0,int(events[0]) if len(events) else 0])):
                    for appearance in ('clean','background'):
                        part=[dict(episode=e['episode'],split=e['split'],timestep=t,appearance=appearance)];b=batch_for(part)
                        with torch.inference_mode():
                            offline=forward(b)[0];repeat=forward(b)[0];np.testing.assert_array_equal(offline,repeat)
                            prefix=b['input_ids'][:,:-(NUM_TOKENS+1)].to('cuda');mask=torch.ones_like(prefix,dtype=torch.bool)
                            decoded,_=model.predict_action(input_ids=prefix,attention_mask=mask,pixel_values=b['pixel_values'].to('cuda',dtype=torch.bfloat16),unnorm_key='derived',proprio=b['proprio'],proprio_projector=prop,action_head=head,use_film=False)
                        expected=model._unnormalize_actions(offline,'derived');diff=abs(decoded-expected)
                        deploy.append(dict(step=step,episode=e['episode'],timestep=t,appearance=appearance,max_control_action_difference=float(diff.max()),max_gripper_difference=float(diff[:,6].max()),gripper_threshold_disagreements=int(np.sum(np.sign(2*decoded[:,6]-1)!=np.sign(2*expected[:,6]-1))),repeated_offline_max=0.,prediction_only_inputs='current images, language, normalized robot FK proprio; no demo actions/future images'))
        print('CHECKPOINT',step,'done',time.time()-started,flush=True)
    with (a.out/'checkpoint_gripper_metrics.csv').open('x') as f:w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    save('deployment_path_comparison.json',deploy)
    after=subprocess.check_output(['nvidia-smi','-i',gpu,'-q','-d','ECC'],text=True);(out/'ecc_after.txt').write_text(after)
    def counts(x):return [s.strip() for s in x.splitlines() if any(k in s for k in ('Correctable','Uncorrectable','SRAM SM','Threshold Exceeded'))]
    assert counts(before)==counts(after)
    assert all(p.grad is None for m in mods.values() for p in m.parameters())
    save('acceptance.json',dict(status='PASS_FROZEN_INFERENCE',optimizer_updates=0,backward=0,new_closed_loop=0,checkpoint_steps=[0,45,135,270],fixed_appearance_windows=len(fixed),GPU_UUID=gpu,Math_SDPA=True,ECC_increment=False,all_frozen=True,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),elapsed_s=time.time()-started))
if __name__=='__main__':main()
