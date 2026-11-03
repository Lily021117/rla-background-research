"""Start only the bounded data continuation in an independent CPU session."""
from common import *
import subprocess,time
def main():
 p=RUN/'pipeline_runtime.json';assert not p.exists(),'Inspect existing task; do not launch duplicates'
 with open(RUN/'pipeline_console.log','x') as log:
  child=subprocess.Popen(['/usr/bin/nice','-n','5','/data/lyc/projects/vla-adapter-ref/.venv-cpu/bin/python','-u',str(Path(__file__).with_name('pipeline.py'))],cwd=Path(__file__).resolve().parents[2],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,close_fds=True)
 dump(RUN/'pipeline_launch.json',dict(pid=child.pid,time_unix=time.time(),detached_session=True,scope='finite data downloads already running; lineage inventory; source-gated paired rendering; loader checks; sanitized report and git sync; no models'))
 print('CPU data pipeline PID',child.pid,flush=True)
if __name__=='__main__':main()
