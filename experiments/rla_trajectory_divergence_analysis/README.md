# Existing paired-trajectory analysis (CPU only)

Base commit: `3a3a3c091e707cd5372d763ed80ff1f86eee9849`.
Dependencies: Python, numpy, pandas, matplotlib, Pillow, imageio and imageio-ffmpeg.
No policy, torch, LIBERO or MuJoCo import is needed. No weights are loaded.

Set CUDA_VISIBLE_DEVICES empty and all temporary/cache directories to the data disk.
`GAP_RUN` denotes the existing appearance-gap `run_20261028_01`; its verified
reuse index locates the previous C and historical shared Clean results.
`OUTPUT` must be a new independent directory; existing outputs are not overwritten.

```bash
python experiments/rla_trajectory_divergence_analysis/analyze.py --source "$GAP_RUN" --out "$OUTPUT"
python experiments/rla_trajectory_divergence_analysis/supplement.py "$OUTPUT"
python experiments/rla_trajectory_divergence_analysis/video_sheets.py "$OUTPUT"
```

The first command freezes rules and the 200-pair inventory before statistics.
The supplement checks original initial-state array hashes and summarizes existing
contact/height/centering predicates without introducing new thresholds.
The video command only decodes the eight predetermined pairs into local review
sheets. Review limitations and observations are recorded in the report.

`finalize.py` is the run-specific protection and publication step; it checks every
registered source hash and publishes an allowlist of small summaries. Its report
template contains findings from this frozen run and is not a generic report for
different datasets. Do not overwrite the checked-in delivery when reproducing.

Raw trajectories, video sheets, curve images, weights and absolute source paths
are deliberately absent from the repository. Public CSVs are derived metrics,
not the original state/action trajectories. Threshold sensitivity is fixed at
3/5 cm with eight control steps; no outcome-driven threshold search is performed.
