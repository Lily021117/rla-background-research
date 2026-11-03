"""Preserve own first pilot attempt before correcting a known visual-state omission."""
from common import *
def main():
 source=RUN/'trajectories/libero_10/traj_000010';dest=RUN/'failed_attempts/libero_10/traj_000010_missing_burner_state'
 assert source.is_dir() and not dest.exists();dest.parent.mkdir(parents=True,exist_ok=True)
 records=[dict(old=str(source),new=str(dest),reason='AI fixed-frame review found burner off despite button>=.5; static XML alone omits official qpos-derived visualization')]
 source.rename(dest)
 for app in ['clean','background']:
  src=RUN/'official_loader'/app/'libero_10/traj_000010'
  if src.exists():
   dst=dest/('previous_loader_'+app);src.rename(dst)
   # Preserve these failed-attempt links to the failed images, not the upcoming corrected images.
   for stream in STREAMS:
    link=dst/(stream+'_rgb');assert link.is_symlink();link.unlink();link.symlink_to(dest/app/(stream+'_rgb'),target_is_directory=True)
   records.append(dict(old=str(src),new=str(dst)))
 dump(RUN/'visual_state_repair_record.json',dict(records=records,previous_result_sha256=sha(dest/'result.json'),source_code=str(OLD/'libero_official/libero/libero/envs/objects/articulated_objects.py'),source_code_sha256=sha(OLD/'libero_official/libero/libero/envs/objects/articulated_objects.py'),correction='qpos-derived display only; same source states/actions/camera/lighting; no environment search or replay',repeated_source_frames=383,old_attempt_preserved=True))
if __name__=='__main__':main()
