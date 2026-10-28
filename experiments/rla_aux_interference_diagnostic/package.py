"""Allowlisted sanitized evidence; never copy checkpoints/data/gradient arrays."""
import hashlib,json,re,shutil,sys
from pathlib import Path
out=Path(sys.argv[1]);src=Path(__file__).resolve().parent;dest=src/'delivery'/out.name;dest.mkdir(parents=True,exist_ok=False)
allowed=['report.md','next_step_recommendation.md','diagnostic_protocol.json','checkpoint_inventory.json','snapshot_manifest.json','corrected_snapshot_manifest.json','batch_manifest.json','initial_rng_audit.json','preparation_acceptance.json','scope_correction.json','precompute_incident.json','delivery_verification.json','resource_release_verification.json','executed_source_inventory.json','supplemental_provenance.json','run_plan.json','historical_training_curves.csv','paired_validation_summary.csv','per_demo_validation.csv','time_third_validation.csv','earliest_observed_offline_difference.json','target_switch_per_batch.csv','target_switch_summary.csv','snapshot_summary.csv','earlier_valid_VLM_proprio_gradients.csv','superseded_mixed_head_statistics.csv','gradient_effect_size.csv','additional_provenance_verification.json']
def sanitize(t):
 t=t.replace(str(out),'${RUN_ROOT}').replace('/data/lyc/projects/rla-wm','${PROJECT_ROOT}').replace('/data/lyc','${DATA_ROOT}')
 return re.sub(r'GPU-[0-9a-f-]{36}','${GPU_UUID}',t)
rows=[]
for name in allowed:
 p=out/name;assert p.is_file(),name
 content=sanitize(p.read_text())
 with (dest/name).open('x') as f:f.write(content)
 rows.append(dict(file=name,sha256=hashlib.sha256(content.encode()).hexdigest()))
for attempt,names in [('diagnostic_retry1',['acceptance_20261101_1.json','acceptance_20261102_1.json']),('diagnostic_scope_corrected',['diagnostic_acceptance.json','gradient_scope.json','gradient_statistics.csv'])]:
 for name in names:
  content=sanitize((out/'C'/attempt/name).read_text())
  with (dest/(attempt+'_'+name)).open('x') as f:f.write(content)
(dest/'figures').mkdir()
for p in (out/'figures').glob('*.png'):shutil.copyfile(p,dest/'figures'/p.name)
with (dest/'publication_scope.json').open('x') as f:json.dump(dict(visibility='public, explicitly authorized',included='source, reproducible configs, sanitized reports, summary/per-batch scalar metrics, scientific plots',excluded=['weights','raw images/data','videos','sidecars','full gradient tensors','credentials'],text_files=rows),f,indent=2)
print(dest)
