import os,json,csv,hashlib,shutil,subprocess
from pathlib import Path
O=Path(__file__).resolve().parent;P=O.parent/'run_20261021_01';Q=O.parent/'run_20261021_02'
import torch,yaml
from transformers import AutoTokenizer
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def js(n,v):
 with (O/n).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False)
protected=json.loads((P/'protection_before.json').read_text())['sha256']
for p,h in protected.items():assert sha(p)==h,p
for root in [P,Q]:
 for p in root.rglob('*'):
  if p.is_file() and not p.is_symlink() and 'code' not in p.relative_to(root).parts and 'source_runtime' not in p.parts:protected[str(p)]=sha(p)
js('protection_before.json',{'sha256':protected})
for folder in ['rla','sparse_rla']:shutil.copytree(P/'code'/folder,O/'code'/folder)
samples=json.loads((Q/'corrected_sample_manifest.json').read_text());old=json.loads((P/'sample_manifest.json').read_text())
cfg=json.loads((Q/'C_config.json').read_text());source=Path(cfg['instruction_source'])
tasks={r['episode']:r for r in csv.DictReader(source.open())}
tok=AutoTokenizer.from_pretrained(P/'B/source_runtime',local_files_only=True,trust_remote_code=False)
rows=[]
for s in samples:
 assert sha(s['path'])==s['SHA256']
 b=torch.load(s['path'],map_location='cpu',weights_only=False);a=torch.load(old[s['id']]['path'],map_location='cpu',weights_only=False)
 text=tok.decode(b['input_ids'][0].tolist(),skip_special_tokens=False).lower();expected=tasks[s['episode']]['task']
 assert expected in text and sum(t in text for t in {r['task'] for r in tasks.values()})==1
 for k in ['pixel_values','proprio','actions','rla']:assert torch.equal(a[k],b[k]),(s['id'],k)
 assert all(s[k]==old[s['id']][k] for k in ['episode','split','transition_id','appearance'])
 rows.append({'sample_id':s['id'],'episode':s['episode'],'task_id':expected.replace(' ','_'),'source_task':expected,'actual_decoded_input':text,'source_metadata':str(source),'split':s['split'],'appearance':s['appearance'],'transition_id':s['transition_id'],'pass':True,'nonlanguage_fields_identical':True})
assert len(rows)==440
assert len({s['episode'] for s in samples if s['split']=='train'})==19
assert len({s['episode'] for s in samples if s['split']=='validation'})==3
assert not ({s['episode'] for s in samples if s['split']=='train'} & {s['episode'] for s in samples if s['split']=='validation'})
with (O/'language_condition_audit.csv').open('x') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
js('sample_manifest.json',samples)
shutil.copy2(P/'sampling_order.json',O/'sampling_order.json');shutil.copy2(P/'common_config.json',O/'common_config.json')
for arm in ['B','C']:
 c=json.loads((P/f'{arm}_config.json').read_text());c['language_manifest']=str(O/'sample_manifest.json');c['fresh_initialization_required']=True
 js(f'{arm}_config.json',c)
 with (O/f'training_config_{arm}.yaml').open('x') as f:yaml.safe_dump(c,f,allow_unicode=True)
s=(P/'code/train_pilot.py').read_text()
s=s.replace("samples=json.loads((O/'sample_manifest.json').read_text());order=json.loads((O/'sampling_order.json').read_text())","samples=json.loads((O/'sample_manifest.json').read_text());order=json.loads((O/'sampling_order.json').read_text())\nlanguage_log=(OUT/'actual_model_language.jsonl').open('x',buffering=1)\nexpected_instruction=[]\ndef language_hook(module,args,kwargs):\n ids=kwargs.get('input_ids');assert ids is not None\n for tokens,expected in zip(ids,expected_instruction):\n  decoded=processor.tokenizer.decode(tokens.detach().cpu().tolist(),skip_special_tokens=False).lower();assert expected in decoded\n  language_log.write(json.dumps({'expected':expected,'actual_model_input':decoded,'passed':True})+'\\n')\nmodel.register_forward_pre_hook(language_hook,with_kwargs=True)")
s=s.replace("def loadbatch(ids):","def loadbatch(ids):\n global expected_instruction\n expected_instruction=[samples[i]['instruction'] for i in ids]")
s=s.replace("write('initialization.json',", "prior=json.loads((O.parent/'run_20261021_01'/ARM/'initialization.json').read_text());assert prior['common_VLM_hashes']==initial and prior['head']==digest_state(head.state_dict()) and prior['proprio']==digest_state(proprio.state_dict())\nwrite('initialization.json',")
with (O/'code/train_pilot.py').open('x') as f:f.write(s)
s=(P/'finalize.py').read_text().replace('# Step 5.1 自定义RLA辅助策略训练','# Step 5.1.1 修正任务语言后的策略重训练')
with (O/'finalize.py').open('x') as f:f.write(s)
js('pretraining_acceptance.json',{'language_batches':440,'passed':True,'nonlanguage_identical':True,'fresh_base_checkpoint':cfg['base_checkpoint'],'source_git':subprocess.check_output(['git','rev-parse','HEAD'],cwd=O/'code',text=True).strip(),'sampling_order_SHA256':sha(O/'sampling_order.json'),'previous_invalid_updates':200,'new_updates_before_training':0})
print('AUDIT_PASS_440')
