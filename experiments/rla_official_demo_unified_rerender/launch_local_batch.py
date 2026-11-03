"""Render already verified local sources while finite scoped downloads continue."""
from common import *
import subprocess,time
def main():
 assert (RUN/'pilot_admission.json').exists() and not (RUN/'batch_local_runtime.json').exists()
 with open(RUN/'batch_local_console.log','x') as log:
  child=subprocess.Popen(['/usr/bin/nice','-n','5','/data/lyc/projects/vla-adapter-ref/.venv-cpu/bin/python','-u',str(Path(__file__).with_name('batch.py')),'--phase','local'],cwd=Path(__file__).resolve().parents[2],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,close_fds=True)
 dump(RUN/'batch_local_launch.json',dict(pid=child.pid,time_unix=time.time(),phase='local source-confirmed subset',batch_lock='Exclusive shared with later full-source batch',detached=True));print(child.pid)
if __name__=='__main__':main()
