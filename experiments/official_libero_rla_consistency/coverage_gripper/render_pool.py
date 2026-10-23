"""Bounded parallel CPU export of the frozen newly admitted episode selection."""
import argparse,json,os,sys,subprocess,concurrent.futures
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
selection=a.out/'dataset_selection.json';entries=json.loads(selection.read_text())['episodes'];script=Path(__file__).resolve().parents[1]/'recover_sequence.py'
def render(e):
    if e['existing_frames']:return dict(episode=e['episode'],status='REUSED',root=e['existing_frames'])
    out=a.out/'episodes'/e['episode'];out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',LP_NUM_THREADS='2',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    with (out/'render.log').open('x') as f:
        cp=subprocess.run([sys.executable,str(script),'--project',str(a.project),'--out',str(out),'--episode',e['episode'],'--split',e['split'],'--selection-manifest',str(selection)],env=env,stdout=f,stderr=subprocess.STDOUT)
    return dict(episode=e['episode'],status='PASS' if cp.returncode==0 else 'INVALID',root=str(out),exit_code=cp.returncode)
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
    for value in pool.map(render,entries):
        results.append(value);print(json.dumps(value),flush=True)
        with (a.out/'render_progress.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
with (a.out/'render_results.json').open('x') as f:json.dump(results,f,indent=2)
assert all(x['status'] in ('PASS','REUSED') for x in results),'Failed episodes retained; do not admit them'
