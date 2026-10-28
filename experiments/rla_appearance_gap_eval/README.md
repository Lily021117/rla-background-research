# Frozen C appearance-gap evaluation

This branch evaluates the existing two balanced-sampling C policies only.
There is no training entry point: optimizer steps and backward are forbidden
in the evaluation worker. The historical deployed evaluator is reused with
an explicit protocol override for exactly 200 trials.

Required local inputs are the accepted C checkpoints, their Prismatic setup,
the original 20 development reset states, and the public simulator's generic
color-texture resources. Model weights, states, videos and texture files are
not distributed in this repository. Follow the parent project's license and
dependency setup; use the already verified isolated environment.

Set `OUT` to a fresh data-disk directory and run, from this worktree:

```sh
python experiments/rla_appearance_gap_eval/prepare.py --out "$OUT"
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa CUDA_VISIBLE_DEVICES='' \
  python experiments/rla_appearance_gap_eval/static_checks.py --out "$OUT"
```

Review the two fixed static contact sheets, record `preview_review.json`,
and save `decision_rule.json` before scoring. Both records are delivered with
this run. The rule reports all backgrounds and uses physical-state clustered
bootstrap intervals; the two sampling orders are not independent full seeds.

```sh
python experiments/rla_appearance_gap_eval/freeze.py "$OUT"
python experiments/rla_appearance_gap_eval/run.py "$OUT"
```

The GPU launcher checks authorized device UUID, 26 GiB startup memory and ECC.
Each rollout resets the complete frozen state and uses unchanged 220-step,
8-action execution, camera, proprio, action decoding and success criterion.
Original reflection=1 and MSAA=4 remain enabled. Only the table/wall texture
files change. All four backgrounds are chosen by a fixed lexicographic family
rule before any policy evaluation, excluding previously used background assets.

`run.py` performs exactly 200 valid trials if there is no infrastructure
failure, then analyzes and publishes sanitized small outputs. Infrastructure
errors stop the worker and retain evidence; failed tasks are valid results.
No task-failure retry, checkpoint selection, parameter change or extra trial
is performed. Do not infer universal robustness from a small development set.

## Existing-method completion on the frozen four backgrounds

`method_comparison.py` reuses the preceding run's C160 background scores,
all120 existing Clean scores, and every frozen state/XML/asset byte. Only
the four specified D-center/shared-target final600 checkpoints run:4x20x4,
exactly320 new trials. It does not regenerate or select backgrounds.

```sh
python experiments/rla_appearance_gap_eval/method_comparison.py --out "$OUT" --mode prepare
# Commit reviewed source/configuration before scoring, then:
python experiments/rla_appearance_gap_eval/method_comparison.py --out "$OUT" --mode run
```

Each reused row has its original ledger/hash and device provenance. Joint
physical-state bootstrap keeps every method, appearance and both orders in
the same resampled cluster. The background2 focus is explicitly exploratory
after the C results. All backgrounds and Clean costs are reported; no new
training or automatic follow-up is performed.
