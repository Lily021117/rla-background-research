"""Only the four fixed task files; pinned upstream identities before reading."""
import os,time,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from prepare import RUN,sha,dump
os.environ['HF_HUB_DISABLE_XET']='1';os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='120'
from huggingface_hub import hf_hub_download,HfApi
FILES=[('97773100c1474cd0d686ebd173cc0e4fd5442466','libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo.hdf5'),
 ('97773100c1474cd0d686ebd173cc0e4fd5442466','libero_object/pick_up_the_milk_and_place_it_in_the_basket_demo.hdf5'),
 ('97773100c1474cd0d686ebd173cc0e4fd5442466','libero_goal/put_the_wine_bottle_on_top_of_the_cabinet_demo.hdf5'),
 ('f13aa24a3da8c43c7225569f28c562979fa0e35a','libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5')]
def main():
 plans=[]
 if (RUN/'hdf5_download_plan.json').exists():
  plans=json.loads((RUN/'hdf5_download_plan.json').read_text())['files']
  assert [(p['revision'],p['filename']) for p in plans]==FILES
 else:
  for rev,name in FILES:
   meta=HfApi().dataset_info('yifengzhu-hf/LIBERO-datasets',revision=rev,files_metadata=True)
   entry=next(x for x in meta.siblings if x.rfilename==name)
   plans.append(dict(revision=rev,filename=name,bytes=entry.size,expected_sha256=entry.lfs.sha256))
 dump(RUN/'hdf5_download_plan.json',dict(repository='yifengzhu-hf/LIBERO-datasets',reason_libero10_revision_diff='97773100 contains only spatial/object/goal; source published later, pin exact current commit before file download; do not substitute Cosmos',files=plans))
 def fetch(x):
  t=time.time();p=RUN/'source_hdf5'/x['filename'];reused=p.exists() and p.stat().st_size==x['bytes'] and sha(p)==x['expected_sha256']
  if not reused:
   for attempt in range(3):
    try:
     p=hf_hub_download('yifengzhu-hf/LIBERO-datasets',repo_type='dataset',revision=x['revision'],filename=x['filename'],local_dir=RUN/'source_hdf5');break
    except Exception as e:
     dump(RUN/f"hdf5_error_{x['filename'].split('/')[0]}_{attempt}.json",dict(file=x['filename'],attempt=attempt,error=repr(e),timestamp=time.time(),source='official HF; same pinned revision'))
     if attempt==2:raise
  got=sha(p);assert got==x['expected_sha256'] and Path(p).stat().st_size==x['bytes']
  r=dict(**x,path=str(p),sha256=got,seconds=time.time()-t,reused_already_downloaded=reused);dump(RUN/f"hdf5_{x['filename'].split('/')[0]}.json",r);print(json.dumps(r),flush=True);return r
 with ThreadPoolExecutor(max_workers=2) as pool: records=list(pool.map(fetch,plans))
 dump(RUN/'hdf5_download_complete.json',dict(records=records,total_bytes=sum(x['bytes'] for x in records)))
if __name__=='__main__':main()
