"""Download only HDF5 task files named by the frozen 1693-entry lineage list."""
from common import *
import time, argparse
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='120'
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi, hf_hub_download
REPO='yifengzhu-hf/LIBERO-datasets'
def main(download):
 rows=list(csv.DictReader(open(OLD/'delivery/source_mapping.csv')));wanted=sorted({x['suite']+'/'+Path(x['rlds_episode_file_path']).name for x in rows})
 local=json.loads((RUN/'source_files.json').read_text());present={x['suite']+'/'+x['filename']:x for x in local}
 planpath=RUN/'download_plan.json'
 if planpath.exists():plan=json.loads(planpath.read_text())['files']
 else:
  metadata={};plan=[]
  for rev in ['97773100c1474cd0d686ebd173cc0e4fd5442466','f13aa24a3da8c43c7225569f28c562979fa0e35a']:
   info=HfApi().repo_info(REPO,repo_type='dataset',revision=rev,files_metadata=True);assert info.sha==rev
   metadata[rev]={x.rfilename:x for x in info.siblings}
  for name in wanted:
   rev='f13aa24a3da8c43c7225569f28c562979fa0e35a' if name.startswith('libero_10/') else '97773100c1474cd0d686ebd173cc0e4fd5442466'
   entry=metadata[rev][name];assert entry.lfs
   existing=present.get(name);assert existing is None or existing['sha256']==entry.lfs.sha256,(name,'Local source differs from published revision')
   plan.append(dict(filename=name,revision=rev,bytes=entry.size,sha256=entry.lfs.sha256,existing_path=existing['path'] if existing else None))
  dump(planpath,dict(repository=REPO,files=plan,new_bytes=sum(x['bytes'] for x in plan if not x['existing_path']),scope='Only task files referenced by frozen author list; unused demos not added to dataset'))
 print('download plan',len(plan),'new bytes',sum(x['bytes'] for x in plan if not x['existing_path']),flush=True)
 if not download:return
 records=[]
 def fetch(x):
  started=time.time();dest=RUN/'downloads'/x['filename'];reused=True
  try:
   if x['existing_path']:p=Path(x['existing_path'])
   elif dest.exists() and dest.stat().st_size==x['bytes'] and sha(dest)==x['sha256']:p=dest
   else:
    reused=False;p=Path(hf_hub_download(REPO,repo_type='dataset',revision=x['revision'],filename=x['filename'],local_dir=RUN/'downloads'))
   actual_bytes=p.stat().st_size;actual_sha256=sha(p)
   assert actual_bytes==x['bytes'] and actual_sha256==x['sha256'],dict(actual_bytes=actual_bytes,actual_sha256=actual_sha256,expected_bytes=x['bytes'],expected_sha256=x['sha256'])
   record=dict(**x,path=str(p),verified=True,reused=reused,seconds=time.time()-started)
  except Exception as e:
   record=dict(**x,verified=False,error=repr(e),seconds=time.time()-started)
  return record
 with ThreadPoolExecutor(max_workers=2) as pool:
  for f in as_completed([pool.submit(fetch,x) for x in plan]):
   record=f.result();records.append(record);dump(RUN/'download_records.json',records);print(record['filename'],record['verified'],flush=True)
 dump(RUN/'download_complete.json',dict(records=len(records),verified=sum(x['verified'] for x in records),new_verified_bytes=sum(x['bytes'] for x in records if x['verified'] and not x['reused']),wire_bytes='not instrumented; unique verified-file bytes only'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');main(p.parse_args().download)
