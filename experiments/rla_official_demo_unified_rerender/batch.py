"""Resumable, bounded CPU batch. Stop affected source file after any failure."""
from common import *
import subprocess, concurrent.futures, threading, time, shutil

def main():
 approval=json.loads((RUN/'pilot_admission.json').read_text());assert approval['all_four_pass']
 source=RUN/'source_manifest.csv';rows=list(csv.DictReader(open(source)))
 assert len(rows)==1693 and sum(x['split']=='train' for x in rows)==1653
 frozen=RUN/'batch_frozen_manifest.csv'
 if frozen.exists():assert sha(frozen)==sha(source),'Mapping changed after batch freeze'
 else:shutil.copy2(source,frozen)
 jobs=[x for x in rows if x['source_status']=='MULTI_EVIDENCE_JOIN' and x['split_conflict']=='False']
 lock=threading.Lock();blocked=set();results=[];code=Path(__file__).parent;logdir=RUN/'batch_logs';logdir.mkdir(exist_ok=True)
 def one(x):
  key=x['source_sha256'];suite=x['suite'];tid=x['trajectory_id'];out=RUN/'trajectories'/suite/f'traj_{tid}'
  with lock:
   if key in blocked:return dict(suite=suite,trajectory_id=tid,status='WITHHELD_AFFECTED_SOURCE_FAILURE')
  if (out/'result.json').exists():return dict(suite=suite,trajectory_id=tid,status='REUSED_COMPLETE',result=str(out/'result.json'))
  if out.exists():
   with lock:blocked.add(key)
   return dict(suite=suite,trajectory_id=tid,status='PRESERVED_INCOMPLETE_NO_AUTO_RERUN')
  if shutil.disk_usage(RUN).free<100*2**30:
   with lock:blocked.add(key)
   return dict(suite=suite,trajectory_id=tid,status='DISK_GATE_STOP')
  started=time.time();log=logdir/f'{suite}_{tid}.log'
  with open(log,'x') as f:
   process=subprocess.run(['/data/lyc/projects/vla-adapter-ref/.venv-cpu/bin/python','-u',str(code/'render.py'),'--suite',suite,'--id',tid],stdout=f,stderr=subprocess.STDOUT)
  if process.returncode:
   with lock:blocked.add(key)
  return dict(suite=suite,trajectory_id=tid,status='COMPLETE' if process.returncode==0 else 'FAILED_SOURCE_STOP',exit_code=process.returncode,seconds=time.time()-started,log=str(log))
 # Eight isolated CPU workers, two OSMesa threads each; no GPU/other jobs touched.
 dump(RUN/'batch_runtime.json',dict(pid=os.getpid(),jobs=len(jobs),workers=8,source_manifest_sha256=sha(source),status='RUNNING',started_unix=time.time()))
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  futures=[pool.submit(one,x) for x in jobs]
  for f in concurrent.futures.as_completed(futures):
   row=f.result();results.append(row);dump(RUN/'batch_results.json',results);print(json.dumps(row),flush=True)
 dump(RUN/'batch_complete.json',dict(status='COMPLETED_WITH_EXCLUSIONS' if blocked else 'COMPLETE_CONFIRMED_SUBSET',jobs=len(jobs),blocked_source_hashes=sorted(blocked),rows=results))
if __name__=='__main__':main()
