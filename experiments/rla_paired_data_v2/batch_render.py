"""Two CPU render workers, no GPU allocation."""
import concurrent.futures,subprocess,sys
from common import *
def launch(e):
 log=ROOT/(e['episode']+'_render.log')
 with log.open('x') as f:
  subprocess.run([sys.executable,str(HERE/'render.py'),'--mode','full','--episode',e['episode']],stdout=f,stderr=subprocess.STDOUT,check=True)
 return e['episode']
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for name in pool.map(launch,read(ROOT/'source_manifest.json')):print('COMPLETE',name,flush=True)
