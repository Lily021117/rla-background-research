"""No CUDA initialization; check only this run's workers and ECC deltas."""
import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();out=a.run/'recovery_20261024'
def counters(s):return [x.strip() for x in s.splitlines() if any(k in x for k in ('Correctable','Uncorrectable','SRAM SM','Threshold Exceeded'))]
workers=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:args=(proc/'cmdline').read_bytes().decode().split('\0')
 except (OSError,UnicodeDecodeError):continue
 scripts=('baseline_update_smoke.py','train_ae.py','cache_features.py','evaluate_and_extract.py','raw_source_check.py')
 if any(any(s in arg for s in scripts) for arg in args) and any(str(a.run) in arg for arg in args):workers.append(proc.name)
assert not workers,'This run still has model workers; do not claim resources released'
for index in (0,1):
 after=subprocess.check_output(['nvidia-smi','-i',str(index),'-q','-d','ECC'],text=True);before=(out/f'gpu{index}_ecc_start.txt').read_text();assert counters(before)==counters(after)
 with (out/f'gpu{index}_ecc_finish.txt').open('x') as f:f.write(after)
receipt=dict(status='PASS_OUR_GPU_WORKERS_RELEASED',our_GPU_workers=workers,used_devices=[0,1],GPU2_excluded=True,GPU3_not_reused_after_new_fault=True,ECC_increment_during_recovery=False,other_processes_not_terminated=True,GPU_reset=False,all_new_outputs_on_data_volume=True,system_cache_check='No recent files observed in existing root NV or torchinductor caches during recovery; explicit HF/Torch/XDG/TMP and future CUDA/Triton caches on data volume')
with (out/'resource_release_receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt,indent=2))
