"""Freeze zero-update policy readiness inputs and next-turn sampling provenance."""
import argparse,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output-name',default='policy_readiness');a=p.parse_args();out=a.run/a.output_name;out.mkdir(exist_ok=False);base=a.project/'experiments/official_libero_rla_consistency/run_20261023_06';data=a.project/'experiments/official_libero_rla_consistency/run_20261023_04'
stats=json.loads((data/'normalization_statistics_expanded_train_only.json').read_text())
with (out/'normalization_statistics_train_only.json').open('x') as f:json.dump(stats,f,indent=2)
inputs=dict(project=str(a.project),B20_step0=str(base/'B20/formal_B/step0000.pt'),B20_sampling_order=str(base/'B20/formal_B/sampling_order.json'),B20_sampling_sha256=hashlib.sha256((base/'B20/formal_B/sampling_order.json').read_bytes()).hexdigest(),data_manifest=str(data/'frozen_dataset_manifest.json'),action_proprio_statistics=str(data/'normalization_statistics_expanded_train_only.json'),no_B20_final_weights=True,policy_optimizer_updates=0)
with (out/'readiness_inputs.json').open('x') as f:json.dump(inputs,f,indent=2)
