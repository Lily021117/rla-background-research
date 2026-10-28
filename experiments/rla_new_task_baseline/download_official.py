"""Single explicitly authorized public file, pinned revision; no snapshot download."""
import argparse,hashlib,json,time,traceback
from pathlib import Path
from huggingface_hub import hf_hub_download

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 spec=dict(repo_id='yifengzhu-hf/LIBERO-datasets',repo_type='dataset',revision='97773100c1474cd0d686ebd173cc0e4fd5442466',filename='libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5',expected_bytes=702223367,expected_sha256='20252c7cf98cd7437061f7f200ae7b6cb6219fabbd53b4536dfaa8abda6ab737')
 dest=a.out/'downloads';dest.mkdir(exist_ok=False);start=time.time()
 try:
  f=Path(hf_hub_download(**{k:spec[k] for k in ['repo_id','repo_type','revision','filename']},local_dir=dest,cache_dir=a.out/'cache/huggingface',token=False))
  h=hashlib.sha256()
  with f.open('rb') as stream:
   for b in iter(lambda:stream.read(8*1024**2),b''):h.update(b)
  spec.update(path=str(f),actual_bytes=f.stat().st_size,actual_sha256=h.hexdigest())
  assert spec['actual_bytes']==spec['expected_bytes'] and spec['actual_sha256']==spec['expected_sha256']
  spec['status']='PASS'
 except Exception:
  spec.update(status='FAILED',error=traceback.format_exc());raise
 finally:
  spec['seconds']=time.time()-start
  with (a.out/'download_provenance.json').open('x') as f:json.dump(spec,f,indent=2)
if __name__=='__main__':main()
