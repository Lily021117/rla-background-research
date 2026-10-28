"""Post-run read-only verification and bounded accounting, including gaps."""
import hashlib,json,subprocess,sys
from pathlib import Path
out=Path(sys.argv[1]);w=out/'C/diagnostic_scope_corrected'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
protected=json.loads((out/'protected_inputs.json').read_text());mismatch=[]
for path,expected in protected.items():
 actual=sha(Path(path))
 if actual!=expected:mismatch.append(dict(path=path,expected=expected,actual=actual))
assert not mismatch
accept=json.loads((w/'diagnostic_acceptance.json').read_text());assert accept['status']=='PASS'
c=accept['counters'];assert c['completed_units']==30 and c['objective_full_batches']==90 and c['forward_microbatches']==c['backward_microbatches']==180
old=json.loads((out/'C/diagnostic_retry1/unit_073.json').read_text())['counters']
assert old['objective_full_batches']==225 and old['forward_microbatches']==old['backward_microbatches']==450
ps=subprocess.check_output(['ps','-eo','pid,args'],text=True)
remaining=[r for r in ps.splitlines() if '--entry rla_aux_interference_diagnostic.worker' in r and str(out) in r and 'bootstrap.py' in r]
assert not remaining
gpu='GPU-205a3f5a-d91b-6385-c588-4e4cf3505f7e';ecc=subprocess.check_output(['nvidia-smi','-i',gpu,'-q','-d','ECC'],text=True)
def counts(s):return [x.strip() for x in s.splitlines() if any(k in x for k in ('Correctable','Uncorrectable','SRAM SM','Threshold Exceeded'))]
assert counts(ecc)==counts((w/'ecc_before.txt').read_text())
free=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip()
v=dict(status='PARTIAL_SCOPE_COVERAGE',optimizer_updates=0,AE_updates=0,new_rollouts=0,sealed_reads=0,complete_gradient_effective_batches=315,complete_microbatch_forwards=630,complete_microbatch_backwards=630,inflight_microbatch_calls_min=1,inflight_microbatch_calls_max=6,exact_inflight_count='not persisted at controlled interruption',gradient_effective_batch_upper_bound=318,microbatch_forward_backward_upper_bound=636,acceptance_effective_batches_included=6,implementation_launch_retry=1,scope_correction_restart=1,infrastructure_retries=0,completed_original_units=73,completed_corrected_units=30,corrected_parameter_snapshots=5,original_plan_snapshots=13,protected_input_files=len(protected),historical_hash_mismatches=mismatch,remaining_task_GPU_workers=remaining,ECC_increment=False,final_GPU_free_MiB=int(free),full_gradient_arrays_saved=False,actual_corrected_counter=c)
with (out/'delivery_verification.json').open('x') as f:json.dump(v,f,indent=2)
with (out/'resource_release_verification.json').open('x') as f:json.dump(dict(remaining_task_GPU_workers=remaining,ECC_increment=False,free_MiB=int(free),gpu_uuid=gpu),f,indent=2)
print('PROTECTION_AND_RESOURCE_VERIFIED',len(protected))
