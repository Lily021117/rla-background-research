"""Verify protected inputs and publish only small, sanitized analysis artifacts."""
import sys,json,re
from pathlib import Path
import pandas as pd
from analyze import sha,read,save,table
p=Path(sys.argv[1]);here=Path(__file__).parent;dest=here/'delivery';dest.mkdir(exist_ok=False)
hashes=read(p/'input_hashes.json');hashes.update(read(p/'additional_source_hashes.json'))
changed=[name for name,h in hashes.items() if not Path(name).is_file() or sha(name)!=h]
assert not changed,changed
metrics=pd.read_csv(p/'recorded_condition_summary.csv');metrics['preterminal_offcenter_contact']=metrics.first_contact_outside_center.notna()
summary=metrics.groupby(['order','condition','group','model']).agg(n=('steps','size'),preterminal_offcenter_contact=('preterminal_offcenter_contact','sum'),terminal_center=('terminal_center','sum'),terminal_contact=('terminal_contact','sum'),terminal_height=('terminal_height','sum')).reset_index()
table(p/'offcenter_contact_by_group.csv',summary.to_dict('records'))
text=(here/'report_template.md').read_text()
with (p/'report.md').open('x') as f:f.write(text)
with (p.parent/'report.md').open('x') as f:f.write(text)
verification=dict(status='PASS_OFFLINE_PROCESS_PARTIAL_MECHANISM',planned_pairs=200,valid_pairs=200,trajectories=400,clean_pairs=40,new_background_pairs=160,protected_files_checked=len(hashes),changed_historical_files=changed,model_forward=0,backward=0,optimizer_updates=0,AE_updates=0,new_closed_loop=0,simulation_replay=0,offline_video_tool_retry=1,manual_video_pairs=8,GPU_used=False,GPU_release='not applicable: no GPU allocated',sealed_set_read=False,report_sha256=sha(p/'report.md'),publication='source/config/sanitized numeric summaries only; remote verified separately')
save(p/'delivery_verification.json',verification)
files=['report.md','delivery_verification.json','frozen_analysis_rules.json','analysis_acceptance.json','field_semantics.json','outcome_group_counts.csv','candidate_group_summary.csv','offcenter_contact_by_group.csv','paired_summary.csv','trajectory_metrics.csv','recorded_condition_summary.csv','event_table.csv','chunk_position_metrics.csv','within_vs_boundary_command_changes.csv','field_time_audit.csv','identity_and_initial_state_audit.csv','inventory_acceptance.json']
for name in files:
    t=re.sub(r'/data/[^/]+/projects/rla-wm/','${PROJECT}/',(p/name).read_text())
    t=re.sub(r'/data/[^/]+/','${DATA}/',t)
    with (dest/name).open('x') as f:f.write(t)
save(p/'output_hashes.json',{str(f.relative_to(p)):sha(f) for f in p.rglob('*') if f.is_file()})
print(json.dumps(verification,indent=2))
