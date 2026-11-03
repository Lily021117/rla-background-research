"""Record the main agent's completed fixed-frame review; not automated human certification."""
from common import *
import shutil
def main():
 rows=[];seconds=0;frames=0;size=0
 notes={
 'libero_spatial':'Seven fixed times, both views: robot/bowls/plate/cabinet remain excluded; same-state Clean/H+S geometry visually aligned; table/wall hue visibly changes. Old video path differences are descriptive.',
 'libero_object':'Seven fixed times, both views: floor/wall recolored, basket and all grocery objects excluded; milk transport and occlusion remain plausible; wrist background stays geometrically aligned.',
 'libero_goal':'Seven fixed times, both views: bottle/cabinet/cutting board/knife excluded; background changes around them; preserved source sequence includes first retained raw index1 without shifting or deleting failures.',
 'libero_10':'Seven fixed times, both views AFTER correction: two moka pots and stove excluded; burner now correctly red for button>=0.5 in both Clean and H+S; cupboard background remains excluded; no obvious object/camera geometry mismatch between pair.'}
 for suite in SUITES:
  p=RUN/'trajectories'/suite/'traj_000010';r=json.loads((p/'result.json').read_text());a=json.loads((p/'loader_audit.json').read_text())
  assert r['status']=='NUMERIC_PASS_PENDING_VISUAL' and a['status']=='PASS'
  assert (p/'loader_audit.json').stat().st_mtime>=(p/'result.json').stat().st_mtime
  rows.append(dict(suite=suite,trajectory_id='000010',result_sha256=sha(p/'result.json'),loader_audit_sha256=sha(p/'loader_audit.json'),review_image=str(p/'fixed_comparison.jpg'),review_image_sha256=sha(p/'fixed_comparison.jpg'),fixed_frame_ids=sorted(set(np.linspace(0,r['frames']-1,7,dtype=int).tolist())),review_type='AI visual inspection by main agent via view_image',human_review=False,whole_video_manually_reviewed=False,full_sequence_numeric_checks=True,visual_pass=True,notes=notes[suite]))
  seconds+=r['seconds'];frames+=r['frames'];size+=sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
 dump(RUN/'pilot_admission.json',dict(all_four_pass=True,scope='Only four complete ORIGINAL HDF5-state pilots; not full 1693 coverage and not old-video exact reproduction',rows=rows,approved_batch='Only source-joined and split-nonconflicting entries; no model task'))
 frame_total=sum(int(x['frames']) for x in csv.DictReader(open(OLD/'delivery/source_mapping.csv')))
 dump(RUN/'resource_estimate.json',dict(measured_pilot_frames=frames,sum_worker_seconds=seconds,seconds_per_frame=seconds/frames,author_manifest_frames=frame_total,serial_hours_if_all_frames=frame_total*seconds/frames/3600,ideal_8worker_hours=frame_total*seconds/frames/3600/8,caveat='Extrapolation only; compilation, disk, concurrency and download overhead not guaranteed. Confirmed subset may be smaller.',pilot_bytes=size,conservative_full_data_bytes_by_trajectory=size/4*1693*1.5,free_disk_bytes=shutil.disk_usage(RUN).free,cpu_workers=8,osmesa_threads_per_worker=2,gpu_use=False,min_disk_free_gib=100))
 print('four original-state pilots PASS; CPU batch released',flush=True)
if __name__=='__main__':main()
