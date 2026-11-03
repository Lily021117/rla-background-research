"""Freeze incremental scope, retain historical results and initial audit evidence."""
from common import *
from complete_sources import PREVIOUS
import shutil,subprocess

def main():
 assert not (RUN/'protocol.json').exists()
 previous=json.loads((PREVIOUS/'protocol.json').read_text())
 previous.update(name='官方示范统一重渲染派生版：来源补齐与统一核验',parent_commit='ef99dd5aadf231407641b71612193a3a1aa7cbcb',previous_run=str(PREVIOUS),previous_manifest_sha256=sha(PREVIOUS/'source_manifest.csv'),previous_rendered=955,download_scope='12 missing files only; fixed parent plan; max two new attempts each; prior verified 28 hash-checked and reused; old caches immutable',source_review='All 1693 under same conservative evidence rule. v2 fixes diagnostic interpretation, not thresholds. Do not automatically admit on unique actions alone. Full initial state and generation mapping absent remains explicit.',reuse='Source SHA, demo, filtered indices and split unchanged, state/model hashes and PNG structure/counts verified; recheck official loader without writing old results',model_forwards=0,updates=0,backward=0,policy_rollout=0,source_action_replay=0)
 dump(RUN/'protocol.json',previous)
 for name in ['pilot_admission.json','source_manifest.csv','source_files.json','source_coverage.csv','split_conflicts.json','resource_estimate.json']:
  shutil.copy2(PREVIOUS/name,RUN/name)
 evidence=[]
 for path in [OLD/'libero_official/scripts/create_dataset.py',Path('/data/lyc/projects/vla-adapter-ref/experiments/robot/libero/regenerate_libero_dataset.py'),PREVIOUS/'protocol.json',PREVIOUS/'source_manifest.csv',PREVIOUS/'download_plan.json']:
  evidence.append(dict(path=str(path),sha256=sha(path)))
 dump(RUN/'generation_code_evidence.json',dict(files=evidence,regeneration_code_commit=subprocess.check_output(['git','-C','/data/lyc/projects/vla-adapter-ref','rev-parse','HEAD'],text=True).strip(),metainfo='No *_metainfo.json in pinned local generation repository. No per-episode demo ID/initial scene state in archived RLDS metadata. No new replay or environment-version search.',phases=dict(source='env.step(action) precedes obs append; saved states indexed at corresponding pre-action index',regenerated='set orig_states[0]; 10 settling actions; filter noops without executing; record obs then execute retained action',initial_exception='regenerated states[0] copies original state before settling; its first obs is after settling. Do not equate these fields.')))

if __name__=='__main__':main()
