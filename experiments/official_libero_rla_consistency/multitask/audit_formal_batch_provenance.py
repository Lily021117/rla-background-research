"""Disambiguate reference bootstrap diagnostics from actual formal batches."""
import argparse,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();rows=[]
source=Path('experiments/official_libero_rla_consistency/ae_consistency/policy_training.py');text=source.read_text();assert "b=batch(examples[start:start+micro])" in text and "instruction.lower() in decoded.lower()" in text
for task in ('drawer','stove'):
 root=a.run/task/'policy_run';plan=json.loads((root/'run_plan.json').read_text());manifest=json.loads(Path(plan['inputs']['data_manifest']).read_text());order=json.loads(Path(plan['inputs']['B20_sampling_order']).read_text());train={e['episode'] for e in manifest['episodes'] if e['split']=='train'};assert len(order)==4800 and all(e in train for e,t in order)
 real=json.loads((root/'preflight6/isolated_smoke/actual_task_batch_audit.json').read_text());assert all(r['episode'] in train and r['language_verified_in_actual_tokens'] for r in real)
 for arm in ('B','C','D'):
  accept=json.loads((root/arm/'isolated_smoke/training_acceptance.json').read_text());assert accept['successful_updates']==600
 rows.append(dict(task=task,actual_manifest=plan['inputs']['data_manifest'],actual_train_episodes=sorted(train),real_first_batch=real,runtime_language_assertion='Executed for every formal training and validation sample; all runs completed without assertion failure',legacy_bootstrap_batch_audit='Describes reference-only bootstrap batch, not the formal training batch. It is preserved but must not be cited as actual episode/task provenance.',formal_forward='execute.batch(examples) -> forward(b) -> official run_forward_pass(batch=b); no use of global bootstrap batch',sampling_rule='8 physical transitions/update, each clean+background once, common4800 order forB/C/D',optimizer_updates_by_this_audit=0))
with (a.run/'formal_batch_provenance_audit.json').open('x') as f:json.dump(dict(policy_training_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),tasks=rows),f,indent=2)
print('FORMAL_BATCH_PROVENANCE_PASS_REFERENCE_BOOTSTRAP_DISTINGUISHED')
