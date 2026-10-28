"""Hash only the executed model/loss/data path, without importing model code."""
import hashlib,json,sys
from pathlib import Path
out=Path(sys.argv[1]);src=Path(__file__).resolve().parents[1];project=Path('/data/lyc/projects/rla-wm')
paths=[src/'rla_lr_budget_sweep/bootstrap.py',src/'control_preserving_rla/interface.py',src/'control_preserving_rla/common.py',src/'rla_aux_interference_diagnostic/worker.py',src/'official_libero_rla_consistency/stove_centering/training.py',src/'official_libero_rla_consistency/stove_centering/store.py',src/'official_libero_rla_consistency/coverage_gripper/dataset.py',project/'worktrees/rla-official-0fc6e97/rebuttal/exp2_vla_adapter/vla_adapter_overlay/rla/loss.py',project/'worktrees/rla-official-0fc6e97/rebuttal/exp2_vla_adapter/vla_adapter_overlay/rla/action_head.py',project/'experiments/official_libero_rla_consistency/run_20261023_02/vla-official/vla-scripts/finetune.py']
rows=[]
for p in paths:
 content=p.read_bytes();rows.append(dict(path=str(p),sha256=hashlib.sha256(content).hexdigest(),symbols=[dict(line=i,text=s.strip()) for i,s in enumerate(content.decode().splitlines(),1) if s.lstrip().startswith(('def ','class '))]))
with (out/'executed_source_inventory.json').open('x') as f:json.dump(rows,f,indent=2)
