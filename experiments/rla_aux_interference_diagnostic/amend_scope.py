"""Record a scope correction without expanding the user's gradient budget."""
import json,sys
from pathlib import Path
out=Path(sys.argv[1]);ss=json.loads((out/'snapshot_manifest.json').read_text());bs=json.loads((out/'batch_manifest.json').read_text())
old=out/'C/diagnostic_retry1';units=sorted(old.glob('unit_*.json'));assert len(units)==73
last=json.loads(units[-1].read_text());assert last['counters']['objective_full_batches']==225
chosen=[]
for s in ss:
 if s['step']==0:s=dict(s,batch_ids=[bs[0]['id'],bs[4]['id']])
 elif s['step']==600:s=dict(s,batch_ids=[b['id'] for b in bs[:7]])
 else:continue
 chosen.append(s)
assert sum(len(s['batch_ids']) for s in chosen)==30
with (out/'corrected_snapshot_manifest.json').open('x') as f:json.dump(chosen,f,indent=2)
with (out/'scope_correction.json').open('x') as f:json.dump(dict(status='PARTIAL_ORIGINAL_COVERAGE',reason='Action-only head layer_norm1/fc1 were incorrectly included in the original aggregate; one-way RLA mask structurally blocks auxiliary influence. VLM/proprio metrics remain valid. Original mixed head/global metrics excluded from primary inference.',stopped_attempt='diagnostic_retry1',complete_units=73,complete_fullbatch_extractions=225,complete_microbatch_forwards=450,complete_microbatch_backwards=450,inflight_unit='C_20261102_600 / 20261101_200',inflight_fullbatch_extraction_upper_bound=3,inflight_microbatch_forward_backward_upper_bound=6,inflight_exact_count='not persisted; controlled SIGINT during gradient CPU accumulation; conservatively charged whole3-objective unit',corrected_units=30,corrected_fullbatch_extractions=90,total_fullbatch_extraction_upper_bound=318,total_microbatch_forward_backward_upper_bound=636,budget_not_expanded=True,selection='Final600 all4 method/order snapshots on first7 pre-frozen batches; common step0 on first batch of each order; deterministic prefix truncation for remaining budget, no gradient/outcome selection',omitted_corrected_coverage='100/300 full-head shared gradients; eighth batch for final600; other6 step0 batches',no_new_acceptance_units=True,primary_scope='VLM + proprio + shared mlp_resnet_blocks; exclude layer_norm1/fc1 and final readouts',optimizer_updates=0,AE_updates=0,new_rollouts=0),f,indent=2)
