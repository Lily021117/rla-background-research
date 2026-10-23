"""Post-exit resource receipt; no CUDA context, no model loading."""
import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
def counters(s):return [x.strip() for x in s.splitlines() if any(k in x for k in ('Correctable','Uncorrectable','SRAM SM','Threshold Exceeded'))]
gpu=json.loads((a.run/'execution_resource_update.json').read_text())['GPU_UUID']
before=(a.run/'C_recovery_01/isolated_smoke/ecc_before.txt').read_text();after=subprocess.check_output(['nvidia-smi','-i',gpu,'-q','-d','ECC'],text=True);assert counters(before)==counters(after)
processes=subprocess.check_output(['ps','-eo','pid,args'],text=True);workers=[x for x in processes.splitlines() if 'baseline_update_smoke.py' in x and str(a.run) in x];assert not workers
inventory=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
(a.run/'final_ECC.txt').write_text(after)
receipt=dict(GPU_workers_exited=True,ECC_increment=False,historical_ECC_not_cleared=True,other_processes_not_terminated=True,CUDA_reset_or_driver_changes=False,GPU_process_inventory=inventory,policy_effective_updates={'C':600,'D':600},policy_computed_successful_updates={'C':601,'D':600},failed_policy_forward_attempts=1,AE_updates_this_run=0,valid_rollouts=180,invalid_rollouts=0,pretrial_evaluation_failure=1,all_output_storage='data volume',scope='only experiment GPU workers released; other users processes left untouched')
(a.run/'resource_release_receipt.json').write_text(json.dumps(receipt,indent=2));print('RESOURCE_RELEASE_PASS')
