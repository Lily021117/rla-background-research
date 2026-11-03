"""Bounded source-file completion; never reads an unverified HDF5."""
from common import *
import requests, shutil, time, concurrent.futures

PREVIOUS=ROOT/'experiments/rla_official_demo_unified_rerender/run_20261103_01'

def main():
 plan=json.loads((PREVIOUS/'download_plan.json').read_text())
 dump(RUN/'download_plan.json',plan)
 verified=json.loads((PREVIOUS/'source_files.json').read_text())
 present={x['suite']+'/'+x['filename']:x for x in verified}
 assert len(present)==28
 outcomes=[]
 def fetch(x):
  name=x['filename'];folder=RUN/'download_attempts'/name;folder.mkdir(parents=True,exist_ok=True)
  if (folder/'outcome.json').exists():return json.loads((folder/'outcome.json').read_text())
  if name in present:
   p=Path(present[name]['path']);assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256']
   result=dict(**x,path=str(p),verified=True,reused=True,attempts=0)
   dump(folder/'outcome.json',result);return result
  cache=PREVIOUS/'downloads/.cache/huggingface/download'/Path(name).parent
  partials=[p for p in cache.glob('*.incomplete') if x['sha256'] in p.name and 0<p.stat().st_size<x['bytes']] if cache.exists() else []
  resume=max(partials,key=lambda p:p.stat().st_size) if partials else None
  # Size-correct hash-mismatch files are NOT resume sources. Old caches remain untouched.
  for attempt in [1,2]:
   recordpath=folder/f'attempt_{attempt}.json'
   if recordpath.exists():
    old=json.loads(recordpath.read_text())
    if old.get('verified'):
     dump(folder/'outcome.json',old);return old
    continue
   target=folder/f'attempt_{attempt}.part'
   assert not target.exists(),'Unrecorded interrupted attempt requires explicit recovery'
   if resume:shutil.copy2(resume,target)
   offset=target.stat().st_size if target.exists() else 0
   record=dict(**x,attempt=attempt,verified=False,reused=False,started_unix=time.time(),resume_source=str(resume) if resume else None,resume_bytes=offset,network_bytes=0)
   dump(recordpath,dict(record,status='STARTED'))
   try:
    url=f"https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/{x['revision']}/{name}?download=true&source_completion_attempt={attempt}"
    headers={'Accept-Encoding':'identity','Cache-Control':'no-cache'}
    if offset:headers['Range']=f'bytes={offset}-'
    with requests.Session() as session:
     session.mount('https://',requests.adapters.HTTPAdapter(max_retries=0))
     with session.get(url,headers=headers,stream=True,timeout=(40,120)) as response:
      response.raise_for_status();record['http_status']=response.status_code
      record['content_range']=response.headers.get('Content-Range');record['etag']=response.headers.get('ETag')
      if offset:
       if response.status_code==206:assert response.headers.get('Content-Range','').startswith(f'bytes {offset}-')
       elif response.status_code==200:
        # Preserve unused copied prefix, then accept complete response in a fresh file.
        target.rename(folder/f'attempt_{attempt}_unused_prefix.part');offset=0
       else:raise RuntimeError('Invalid HTTP range response')
      with target.open('ab' if offset else 'wb') as output:
       for block in response.iter_content(2**20):
        if block:output.write(block);record['network_bytes']+=len(block)
        if time.time()-record['started_unix']>1800:raise TimeoutError('Per-attempt wall budget 1800s')
    record['actual_bytes']=target.stat().st_size;record['actual_sha256']=sha(target)
    assert record['actual_bytes']==x['bytes'] and record['actual_sha256']==x['sha256'],'SIZE_OR_SHA256_MISMATCH'
    dest=RUN/'downloads'/name;dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();target.rename(dest)
    record.update(verified=True,path=str(dest),status='VERIFIED');resume=None
   except Exception as e:
    record.update(status='FAILED_PRESERVED',error=type(e).__name__+': '+str(e)[:1500])
    # Only truncated transfers can resume; complete corrupt files require a fresh second attempt.
    resume=target if target.exists() and 0<target.stat().st_size<x['bytes'] else None
   record['seconds']=time.time()-record['started_unix'];dump(recordpath,record)
   print(name,attempt,record['status'],flush=True)
   if record['verified']:break
  dump(folder/'outcome.json',record);return record
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for result in pool.map(fetch,plan['files']):
   outcomes.append(result);dump(RUN/'download_records.json',outcomes)
 dump(RUN/'download_complete.json',dict(files=len(outcomes),verified=sum(x['verified'] for x in outcomes),max_new_attempts_per_missing_file=2))
if __name__=='__main__':main()
