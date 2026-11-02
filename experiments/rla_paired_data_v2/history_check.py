"""Bounded read-only manifest crosscheck, no additional episode contents."""
import importlib.metadata as metadata
from common import *
def main():
 eps=read(ROOT/'source_manifest.json');ae=P/'experiments/rla_ae_from_scratch/run_20261029_01/data_manifest.json';drawer=P/'experiments/rla_drawer_baseline_gap/run_20261101_01/data_source_audit.json'
 ae_eps=read(ae)['episodes'];da=read(drawer)
 used={(e['source_hdf5'],e['source_demo']) for e in ae_eps}
 used|={(da['source'],d) for d in da['train']+da['development']}
 checks=[{'episode':e['episode'],'overlap_with_scratch_AE_or_latest_drawer':(e['source_hdf5'],e['source_demo']) in used} for e in eps]
 assert not any(r['overlap_with_scratch_AE_or_latest_drawer'] for r in checks)
 save(ROOT/'additional_history_check.json',{'status':'PASS','evidence':[{'path':str(p),'sha256':sha(p)} for p in [ae,drawer]],'checks':checks,'scope':'registered common-ancestor source IDs, not a proof of absent unknown upstream pretraining exposure'})
 names=['numpy','scipy','Pillow','h5py','mujoco','robosuite','imageio','imageio-ffmpeg']
 save(ROOT/'environment_inventory.json',{'packages':{n:metadata.version(n) for n in names},'device':'CPU OSMesa','cuda_visible_devices':'','reflection_scene_default_verified':1,'threads_per_render_process':2,'max_render_processes':2,'code_sha256':{p.name:sha(p) for p in HERE.glob('*.py')}})
if __name__=='__main__':main()
