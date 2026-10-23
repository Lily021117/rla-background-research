"""Exact frozen input identities, no GPU or model modification."""
import argparse,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
    return h.hexdigest()
root=a.project/'downloads/dinov3-vitl16-pretrain-lvd1689m';records=[dict(name=n,sha256=sha(root/n),bytes=(root/n).stat().st_size) for n in ('model.safetensors','config.json','preprocessor_config.json')];config=json.loads((root/'config.json').read_text())
data=a.project/'experiments/official_libero_rla_consistency/run_20261023_04';report=a.project/'experiments/official_libero_rla_consistency/run_20261023_06/report.md'
with (a.run/'frozen_input_inventory.json').open('x') as f:json.dump(dict(DINO=dict(local_directory=str(root),files=records,model_type=config['model_type'],frozen=True,compile=False,preprocessing='official uint8/255,ImageNet normalization,no new crop/rotation,224px,14x14 patches,FP32 cached outputs'),B20_report_sha256=sha(report),data_manifest_sha256=sha(data/'frozen_dataset_manifest.json'),shared_action_proprio_statistics_sha256=sha(data/'normalization_statistics_expanded_train_only.json'),frozen_original_inputs_not_modified=True),f,indent=2)
