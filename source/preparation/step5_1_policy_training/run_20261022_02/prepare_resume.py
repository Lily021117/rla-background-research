import json,csv,hashlib,shutil,subprocess
from pathlib import Path
import torch,yaml
from transformers import AutoTokenizer
O=Path(__file__).resolve().parent;P=O.parent/'run_20261022_01'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def js(n,v):
 with (O/n).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False)
protected=json.loads((P/'protection_before.json').read_text())['sha256']
for p,h in protected.items():assert sha(p)==h,p
for p in P.rglob('*'):
 if p.is_file() and not p.is_symlink() and 'source_runtime' not in p.parts and 'code' not in p.relative_to(P).parts:protected[str(p)]=sha(p)
js('protection_before.json',{'sha256':protected})
for folder in ['rla','sparse_rla']:shutil.copytree(P/'code'/folder,O/'code'/folder)
for name in ['sample_manifest.json','sampling_order.json','language_condition_audit.csv']:shutil.copy2(P/name,O/name)
samples=json.loads((O/'sample_manifest.json').read_text())
tokenizer=AutoTokenizer.from_pretrained(P/'B/source_runtime',local_files_only=True,trust_remote_code=False)
for s in samples:
 assert sha(s['path'])==s['SHA256']
 batch=torch.load(s['path'],map_location='cpu',weights_only=False)
 assert s['instruction'] in tokenizer.decode(batch['input_ids'][0].tolist(),skip_special_tokens=False).lower()
assert len(samples)==440
UUID='${GPU_UUID}'
for name in ['common_config','B_config','C_config']:
 c=json.loads((P/(name+'.json')).read_text());c['GPU_UUID']=UUID;c['GPU_switch_user_authorized']=True
 if name=='B_config':c['resume_valid_checkpoint']=str(P/'B/training_step0050.pt');c['resume_SHA256']=sha(c['resume_valid_checkpoint'])
 js(name+'.json',c)
 if name!='common_config':
  with (O/f'training_config_{name[0]}.yaml').open('x') as f:yaml.safe_dump(c,f,allow_unicode=True)
s=(P/'code/train_pilot.py').read_text()
s=s.replace("checkpoint(0);evaluate(0);start=time.time();torch.cuda.reset_peak_memory_stats()", "checkpoint(0);evaluate(0);start_step=0\nif ARM=='B':\n path=Path(cfg['resume_valid_checkpoint']);assert sha(path)==cfg['resume_SHA256'];start_step=restore(path);assert start_step==50\n model.eval();head.eval();proprio.eval();saved_rng=rng()\n with torch.no_grad():forward(loadbatch([order[0][0]]),'Inference');reference=captured['action'].detach().clone()\n assert restore(path)==50\n with torch.no_grad():forward(loadbatch([order[0][0]]),'Inference');assert torch.equal(reference,captured['action'])\n setrng(saved_rng);model.train();head.train();proprio.train()\n shutil.copy2(O.parent/'run_20261022_01/B/preflight_recovery.json',OUT/'preflight_recovery.json')\n write('resume_verification.json',{'strict_reload':True,'step':50,'optimizer_and_rng_restored':True,'output_max_difference':0,'unpersisted_old_update_replayed_once':51,'source_SHA256':sha(path)})\nstart=time.time();torch.cuda.reset_peak_memory_stats()")
s=s.replace("for step in range(1,101):", "if ARM=='B':\n prior_rows=[json.loads(l) for l in (O.parent/'run_20261022_01/B/training.jsonl').read_text().splitlines()]\n for row in prior_rows[:50]:row['execution_origin']='previous_GPU0_persisted';log.write(json.dumps(row)+'\\n')\nfor step in range(start_step+1,101):")
s=s.replace("'replayed_updates':0,'preflight_updates_included':1", "'replayed_updates':1 if ARM=='B' else 0,'updates_executed_this_process':100-start_step,'total_successful_computations_including_unpersisted':101 if ARM=='B' else 100,'preflight_updates_included':1")
with (O/'code/train_pilot.py').open('x') as f:f.write(s)
shell=(P/'run.sh').read_text().replace('run_20261022_01','run_20261022_02').replace('${GPU_UUID}',UUID)
shell=shell[:shell.index("export CUDA_VISIBLE_DEVICES=''",shell.index('train_pilot.py C'))]
with (O/'run.sh').open('x') as f:f.write(shell)
js('pretraining_acceptance.json',{'status':'PASS','rechecked_language_batches':440,'unchanged_nonlanguage_fields_previous_acceptance':str(P/'language_condition_audit.csv'),'GPU_UUID':UUID,'B_resume_valid_correct_language_step':50,'C_initialization':'fresh public Spatial-Pro; not invalid run','retained_budget_each':100,'planned_new_updates_B':50,'planned_new_updates_C':100,'replayed_unpersisted_update_B':1,'historical_invalid_updates_separate':200})
print('PREPARED_GPU1_RESUME')
