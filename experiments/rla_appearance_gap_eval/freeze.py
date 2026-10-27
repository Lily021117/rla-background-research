"""Freeze generated XML, asset and protocol bytes before any scoring."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rla_appearance_gap_eval.common import read,save,sha
out=Path(sys.argv[1]);assert read(out/'static_acceptance.json')['status']=='PASS'
paths=[out/n for n in ['evaluation_protocol.json','decision_rule.json','asset_manifest.json','checkpoint_inventory.json','policy_registry.json','run_plan.json','normalization.json','static_acceptance.json','preview_review.json','development_protocol/evaluation_protocol.json']]
paths+=list((out/'assets').glob('*.png'))+list((out/'development_protocol').glob('seed_*/*.xml'))+list((out/'development_protocol').glob('seed_*/state.npy'))
save(out/'frozen_artifact_hashes.json',{str(p):sha(p) for p in paths});print('FROZEN_BEFORE_SCORING',len(paths))
