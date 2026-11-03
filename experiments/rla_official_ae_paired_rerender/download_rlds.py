"""Download ONLY three missing RLDS suites at already audited public revision."""
import os,json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from prepare import ROOT,RUN,sha,dump
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='120'
from huggingface_hub import hf_hub_download
def main():
 meta=json.loads((ROOT/'artifacts/rla_policy_closed_loop_pilot/rlds.json').read_text())
 assert meta['sha']=='6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551'
 allowed=['libero_object_no_noops/','libero_goal_no_noops/','libero_10_no_noops/']
 files=[x for x in meta['siblings'] if any(x['rfilename'].startswith(s) for s in allowed)]
 out=RUN/'source_rlds';out.mkdir(exist_ok=True);logs=[]
 dump(RUN/'rlds_download_plan.json',dict(repository=meta.get('id','openvla/modified_libero_rlds'),revision=meta['sha'],files=files,total_bytes=sum(x['size'] for x in files)))
 def fetch(x):
  t=time.time();p=out/x['rfilename'];expected=x.get('lfs',{}).get('sha256')
  cached=p.is_file() and p.stat().st_size==x['size'] and (expected is None or sha(p)==expected)
  if not cached:
   errors=[]
   for attempt in range(3):
    try:
     p=hf_hub_download('openvla/modified_libero_rlds',repo_type='dataset',revision=meta['sha'],filename=x['rfilename'],local_dir=out);break
    except Exception as e:
     errors.append(repr(e));dump(RUN/('download_error_'+x['rfilename'].rsplit('/',1)[-1]+'.json'),dict(file=x['rfilename'],errors=errors))
     if attempt==2:raise
  got=sha(p);expected=x.get('lfs',{}).get('sha256');assert Path(p).stat().st_size==x['size'];assert expected is None or got==expected
  return dict(file=x['rfilename'],path=str(p),bytes=x['size'],sha256=got,expected_sha256=expected,seconds=time.time()-t,reused_already_downloaded=cached)
 with ThreadPoolExecutor(max_workers=3) as pool:
  for row in pool.map(fetch,files):
   logs.append(row);dump(RUN/'rlds_download_progress.json',logs);print(json.dumps(row),flush=True)
 dump(RUN/'rlds_download_complete.json',dict(files=len(logs),bytes=sum(r['bytes'] for r in logs),records=logs))
if __name__=='__main__':main()
