"""Frozen stop-token intervention; source snapshot stays unchanged."""
from pathlib import Path
p=Path(__file__).parent/'offline_probe.py';s=p.read_text()
s=s.replace("O/'offline_runtime'","O/'stop_runtime'").replace("'offline_gpu_before.json'","'stop_gpu_before.json'")
s=s.replace("checkpoints=[('A',0),('B',1),('B',50),('B',100),('C',100),('D',100)]","checkpoints=[('A',0),('B',100)]")
needle="teach=head.last_action.detach().float().cpu().numpy();head.clear()"
extra="""teach=head.last_action.detach().float().cpu().numpy();head.clear()
   augmented=dict(b)
   for k in ['input_ids','labels']:
    augmented[k]=torch.cat([b[k],torch.full_like(b[k][:,:1],2)],dim=1)
   augmented['attention_mask']=torch.cat([b['attention_mask'],torch.ones_like(b['attention_mask'][:,:1])],dim=1)
   ft.run_forward_pass(vla=model,action_head=SimpleNamespace(module=head),proprio_projector=prop,batch=augmented,action_tokenizer=at,device_id=0,use_l1_regression=True,use_proprio=True,use_film=False,num_patches=patches,use_pro_version=True,cfg=SimpleNamespace(phase='Inference'))
   stopout=head.last_action.detach().float().cpu().numpy();head.clear()"""
assert needle in s;s=s.replace(needle,extra)
s=s.replace("'train_sequence_length':int(b['input_ids'].shape[1])","'stop_augmented_vs_deploy_max_abs':float(abs(stopout-deploy).max()),'stop_augmented_vs_deploy_L1':float(abs(stopout-deploy).mean()),'prefix_token_IDs_equal':bool(torch.equal(b['input_ids'][:,:-64],inp['input_ids'])),'train_sequence_length':int(b['input_ids'].shape[1])")
s=s.replace("'fixed_input_action_probe.csv'","'stop_token_intervention.csv'").replace("'offline_probe_verification.json'","'stop_probe_verification.json'")
exec(compile(s,str(p),'exec'))
