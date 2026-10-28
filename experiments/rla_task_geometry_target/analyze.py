"""Complete-results analysis; no model or simulation execution."""
import argparse,json,csv
from pathlib import Path
import numpy as np
from labels import read,save,sha
def table(p,rows):
 with p.open('x') as f:w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def analyze(out):
 rows=read(out/'per_trial_results.json');assert len(rows)==400 and all(r['status']=='COMPLETE' for r in rows)
 refs=read(out/'C_reference_results.json');assert len(refs)==200
 conditions=['clean']+[f'new_background_{i}' for i in range(1,5)];orders=[20261101,20261102];methods=['C-reference','Action-only-match','Geometry-target'];data={};counts=[];traces=[];training=[];validation=[];inventory=[]
 for order in orders:
  for method in methods:
   rr=[r for r in refs if r['order']==order] if method=='C-reference' else [r for r in rows if r['model']==f'{method}_order{order}_s600']
   assert len(rr)==100;d={(r['seed'],r['condition']):r for r in rr};states=sorted({r['seed'] for r in rr});assert len(states)==20
   data[order,method]=np.array([[d[s,c]['success'] for c in conditions] for s in states],int)
   for j,c in enumerate(conditions):counts.append(dict(order=order,method=method,appearance=c,successes=int(data[order,method][:,j].sum()),trials=20,rate=float(data[order,method][:,j].mean())))
   for r in rr:traces.append(dict(order=order,method=method,state=r['seed'],appearance=r['condition'],success=r['success'],stage=r['stage'],steps=r['steps'],source=r['trajectory']))
   if method!='C-reference':
    formal=out/f'seed_{order}'/method/'formal';accept=read(formal/'training_acceptance.json');assert accept['successful_updates']==600 and accept['strict_reload']
    plan=read(out/f'seed_{order}/run_plan.json');expected=[json.loads(x) for x in Path(plan['paired_C_sample_trace']).read_text().splitlines()];actual=[json.loads(x) for x in (formal/'actual_sample_ids.jsonl').read_text().splitlines()];assert actual==expected
    training.extend(dict(order=order,method=method,**r) for r in csv.DictReader((formal/'training_metrics.csv').open()));validation.extend(dict(order=order,method=method,**r) for r in csv.DictReader((formal/'validation_metrics.csv').open()));inventory.append(dict(order=order,method=method,global_step=600,sha256=sha(formal/'step0600.pt'),path=str(formal/'step0600.pt')))
   for r in rr:
    ref=next(q for q in refs if q['order']==order and q['seed']==r['seed'] and q['condition']==r['condition'])
    for k in ('state_sha256','xml_sha256','language'):assert ref[k]==r[k]
 rng=np.random.RandomState(20261028);ix=rng.randint(0,20,(20000,20));stats=[];flips=[]
 def stat(label,d):
  v=d[ix];return dict(comparison=label,difference=float(d.mean()),CI95=np.quantile(v.mean(tuple(range(1,v.ndim))),[.025,.975]).tolist())
 for ref in ('Action-only-match','C-reference'):
  both=[]
  for order in orders:
   g=data[order,'Geometry-target'];b=data[order,ref];d=g-b;both.append(d)
   for j,c in enumerate(conditions):
    stats.append(stat(f'{order} Geometry minus {ref} {c}',d[:,j]));flips.append(dict(order=order,reference=ref,appearance=c,geometry_wins=int(((g[:,j]==1)&(b[:,j]==0)).sum()),reference_wins=int(((g[:,j]==0)&(b[:,j]==1)).sum())))
   stats.append(stat(f'{order} Geometry minus {ref} four_background_mean',d[:,1:]));stats.append(stat(f'{order} Geometry minus {ref} background_minus_clean_interaction',d[:,1:]-d[:,0,None]))
  stats.append(stat(f'two orders Geometry minus {ref} four_background_mean',np.mean(both,axis=0)[:,1:]))
 for order in orders:
  for method in methods:
   a=data[order,method];stats.append(stat(f'{order} {method} four_background_minus_clean',a[:,1:]-a[:,0,None]))
 table(out/'success_summary.csv',counts);table(out/'paired_flips.csv',flips);table(out/'episode_metrics.csv',traces);table(out/'training_metrics.csv',training);table(out/'validation_metrics.csv',validation);table(out/'checkpoint_inventory.csv',inventory);save(out/'paired_statistics.json',dict(physical_state_clusters=20,draws=20000,seed=20261028,all_appearances_and_orders_resampled_together=True,results=stats,development_reused=True,training_seed_uncertainty_not_included=True))
 changed=[p for p,h in read(out/'protected_inputs.json').items() if sha(p)!=h];assert not changed
 save(out/'delivery_verification.json',dict(status='COMPLETE_EXECUTION_NOT_METHOD',successful_updates=2400,actual_update_computations=2400,replayed_updates=0,valid_rollouts=400,attempted_rollouts=400,invalid_rollouts=0,C_reused_trials=200,AE_updates=0,teacher_updates=0,sealed_reads=0,protected_inputs_unchanged=True))
 lines=['# Task-relative geometry auxiliary supervision','','Process completed; method validity requires reading all paired results.','', '| Order | Method | Clean | BG1 | BG2 | BG3 | BG4 |','|---|---|---|---|---|---|---|']
 for order in orders:
  for method in methods:lines.append(f'| {order} | {method} | '+' | '.join(f'{n}/20' for n in data[order,method].sum(0))+' |')
 lines+=['','## Scope','Geometry versus architecture-matched action-only is the primary controlled comparison. Geometry versus C also changes supervision meaning, effective output dimension and normalization. Extra simulation labels are used only for training, not deployment. This is not DBC or a demonstrated RLA algorithm improvement. No explicit invariance switch or future-only ablation was tested.','', 'All20 development states have been reused in research. Two sampling orders are not independent full training seeds. No sealed split was read. Final600 only; no checkpoint selection by outcomes.','', '2400 new successful updates and400 valid new rollouts; C200 results reused. AE and teacher updates0. Stop after this run, no coefficient search.']
 with (out/'report.md').open('x') as f:f.write('\n'.join(lines)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();analyze(a.out)
