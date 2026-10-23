"""Preserve an ECC stop; never treats unpersisted compute as final budget."""
import argparse,csv,json,subprocess,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();out=a.run/'hardware_incident_gpu3';out.mkdir(exist_ok=False)
uuid='GPU-ddacf049-4261-ed6e-5772-bc75bd71dcf9'
(out/'ecc_after.txt').write_text(subprocess.check_output(['nvidia-smi','-i',uuid,'-q','-d','ECC'],text=True))
rows=[]
for arm in ('C','D'):
 root=a.run/'stove/ae_run'/('AE_'+arm);metrics=list(csv.DictReader((root/'training_metrics.csv').open()));maximum=max(int(x['step']) for x in metrics)
 rows.append(dict(task='stove',arm=arm,logged_successful_optimizer_updates=maximum,durable_resume_step=0,unpersisted_logged_updates=maximum,possibly_inflight_update_unconfirmed=True,final600_complete=False,resume_checkpoint=str(root/'resume_step0000.pt'),optimizer_scheduler_note='AdamW and model/master/RNG/sample position saved; no scheduler or AMP scaler in fixed-loss-scale AE recipe',maximum_logged_seconds=float(metrics[-1]['seconds'])))
record=dict(status='STOPPED_NEW_ECC_INCREMENT',GPU_UUID=uuid,volatile_correctable_before=8,volatile_correctable_after=9,aggregate_correctable_before=31,aggregate_correctable_after=32,uncorrectable_aggregate_unchanged=15,reason='Any ECC increment triggers stop, even correctable',cache_status='drawer partial unaccepted, stove complete accepted on GPU1 before fault',AE=rows,policy_optimizer_updates=0,new_policy_rollouts=0,other_processes_terminated=False,device_reset=False,automatic_GPU_failover=False,historical_outputs_modified=False)
(out/'incident.json').write_text(json.dumps(record,indent=2))
text='''# New GPU 3 ECC increment — incomplete experiment\n\nBoth new tasks passed full-sequence paired-data acceptance:20 train and5 development episodes each. Twenty new development reset states per task passed static geometry checks, with three predeclared unseen palettes. No sealed contents were read.\n\nGPU 3 correctable ECC increased8→9 (aggregate31→32); its previous uncorrectable aggregate15 did not increase. The drawer cache checker stopped automatically. Both stove AE processes on the same GPU were explicitly stopped; no other task was terminated and no GPU was reset.\n\nAE-C and AE-D each logged44 successful updates, but only the full step0 recovery checkpoint is durable. These are incomplete training runs, not600-step results. Any recovery must preserve their logs and record replay of these44 logged updates; an in-flight update at interruption is unconfirmed. No policy update or new closed-loop trial occurred.\n\nThe partial drawer feature cache is not accepted. A healthy-device cache recovery and complete AE recovery are required before sidecar extraction and policy training. GPU2 remains excluded. No automatic device failover was performed.\n'''
(out/'report.md').write_text(text)
print(json.dumps(record,indent=2))
