"""Publish small sanitized audit artifacts; never package data/checkpoints."""
import argparse,hashlib,json,re,subprocess,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();o=a.output;src=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
before=json.loads((o/'source_provenance.json').read_text())
assert all(sha(p)==h for p,h in before['hashes'].items())
for n in ('official_recipe_audit.md','historical_vs_official_config_diff.csv','experiment_definition.md','minimal_pipeline_report.md','next_training_plan.md','report.md'):
    with (o/n).open('x') as f:f.write((src/n).read_text())
for n in ('data_interface_audit.json','sidecar_alignment_audit.json','checkpoint_acceptance.json','source_provenance.json'):
    s=(o/n).read_text()
    s=s.replace(os.environ['RLA_PROJECT_ROOT'],'$PROJECT_ROOT').replace(os.environ['VLA_ROOT'],'$VLA_ROOT')
    s=re.sub(re.escape(os.environ['DATA_ROOT'])+r'/[^"\s]+','$DATA_PATH',s)
    with (src/n).open('x') as f:f.write(s)
verification=dict(status='PARTIAL',formal_training_ready=False,optimizer_updates=0,backward=0,new_rollouts=0,heldout_reads=0,GPU_compute=False,
    protected_source_hashes_unchanged=True,protection_scope='hashes recorded in source_provenance; no exhaustive historical rehash',
    released_archive_SHA256='24e177f9bcdda3ff0d73789c59987c9aed9bb618629130a4afa6fe931878b44e',
    strict_encoder_decoder_CPU_load=True,new_paired_AE_forward=False,new_official_policy_forward=False,
    new_sidecar_generation=False,checkpoint_RNG_resume_gate=False,
    excluded_uploads=['checkpoints','datasets','RGB','videos','trajectories','credentials'],
    branch=subprocess.check_output(['git','-C',str(src),'branch','--show-current'],text=True).strip())
for base in (src,o):
    with (base/'delivery_verification.json').open('x') as f:json.dump(verification,f,indent=2)
print(json.dumps(verification,indent=2))
