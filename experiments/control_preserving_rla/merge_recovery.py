"""Retain failed attempt; select one complete optimizer path and quantify replay."""
import argparse
import csv
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();o=a.out
old=o/'E-D-keep/formal';new=o/'E-D-keep/resume_gpu1';dest=o/'combined_metrics/E-D-keep';dest.mkdir(parents=True,exist_ok=False)
def rows(p):return list(csv.DictReader(p.open()))
def write(p,rs):
 with p.open('x') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
before=rows(old/'training_metrics.csv');after=rows(new/'training_metrics.csv')
assert len(before)==430 and int(before[-1]['step'])==430
assert len(after)==300 and [int(r['step']) for r in after]==list(range(301,601))
merged=[r for r in before if int(r['step'])<=300]+after;assert [int(r['step']) for r in merged]==list(range(1,601))
write(dest/'training_metrics.csv',merged)
vb=rows(old/'validation_metrics.csv');va=rows(new/'validation_metrics.csv')
write(dest/'validation_metrics.csv',[r for r in vb if int(r['step'])<300]+va)
ib={int(r['step']):r for r in before};ia={int(r['step']):r for r in after};overlap=sorted(set(ib)&set(ia));assert len(overlap)==130
fields=['action_loss','auxiliary_loss','weighted_auxiliary','keep_loss','weighted_keep','total_loss','lr','gradient_vla','gradient_head','gradient_proprio','gradient_aux_head']
training_comparison={k:dict(max_absolute_difference=max(abs(float(ib[s][k])-float(ia[s][k])) for s in overlap),different_rows=sum(float(ib[s][k])!=float(ia[s][k]) for s in overlap)) for k in fields}
def index(rs):return {(r['episode'],r['timestep'],r['appearance']):r for r in rs if int(r['step'])==300}
v0,v1=index(vb),index(va);assert set(v0)==set(v1) and len(v0)==1406
validation_comparison={k:dict(max_absolute_difference=max(abs(float(v0[s][k])-float(v1[s][k])) for s in v0),different_rows=sum(float(v0[s][k])!=float(v1[s][k]) for s in v0)) for k in ('action_L1','translation_L1','rotation_L1','gripper_L1','auxiliary_L1')}
result=dict(original_successful_computed_updates=430,resumed_successful_computed_updates=300,total_actual_successful_computations=730,effective_checkpoint_path_updates=600,replayed_updates=130,failed_update_attempts=1,failed_step=431,failed_before_optimizer_step=True,failed_partial_forward_exposures_not_counted_as_complete_updates=True,old_attempt_and_validation_preserved=True,derived_training_rows=600,discarded_old_rows_in_derived_path='301..430 only; original files retained',training_replay_comparison=training_comparison,validation300_rows_compared=1406,validation300_comparison=validation_comparison)
with (o/'recovery_comparison.json').open('x') as f:json.dump(result,f,indent=2)
print('RECOVERY_MERGE_COMPLETE',result['replayed_updates'])
