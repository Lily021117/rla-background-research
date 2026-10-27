# C sampling coverage: historical evidence and frozen intervention

This is a sampling scheme intervention, not merely exchanging the order of an
identical multiset. No model, AE target, loss weight, optimizer or LR change.

## Evidence scope

The historical full per-item access logs do not exist. Counts below use the saved
4800-item sampling indices, exactly reconstructed from the original RandomState
permutation algorithm. Both C terminal checkpoints confirm global_step=600,
next_physical_sample=4800, and their configuration points to the same index data.
Both logs contain consecutive updates1–600. The actual loader is synchronous
`DerivedEpisodeDataset.sample`; no DataLoader workers or shuffle buffer participate.
This supports executed-order reconstruction, not an independent per-item historical
trace. Historical bootstrap reference batches are not used as evidence of access.

Both C historical runs are uninterrupted600-update records. The general restoration
logic resumes complete model/optimizer/scheduler/RNG state and uses update*8 as
the next physical index. Replayed calculations are not new independent samples.

## Counts

| Sequence | Total per-episode exposures | First200 | Middle200 | Last200 | Last100 |
|---|---|---|---|---|---|
| Historical20261023 |209–280|67–89|61–97|67–95|33–47|
| Historical20261029 |212–279|64–93|62–100|64–95|27–49|
| Balanced20261101 |240 each|80 each|80 each|80 each|40 each|
| Balanced20261102 |240 each|80 each|80 each|80 each|40 each|

All four sequences cover all2555 eligible windows. Historical windows occur1–2
times. Balanced windows occur1–3 times; within each episode the minimum and maximum
counts differ by at most1, and every permutation is exhausted before reshuffling.
Both appearances occur once per physical sample:4800 Clean and4800 Background.
Relative-time decile statistics and every window's counts are saved separately.
Deciles are NOT semantic approach/grasp/placement stage labels.

There is no evidence here that either old sequence omitted an episode or all of
a temporal region. Exposure imbalance alone does not establish the cause of failure.

## Weighting and fairness

The original long-run window probability is1/2555 and episode probability n_e/2555.
The new episode probability is1/20; per-window probability becomes1/(20*n_e).
Episode lengths range112–147 eligible windows. The relative window weight changes
by factors0.8690476–1.140625. Short episodes are upweighted, long episodes downweighted.
All eligible windows remain available; no stage/outcome-specific selection occurs.

Sampler order seeds20261101/20261102 use private RandomState objects, do not consume
global NumPy/model RNG, and are frozen before new outcomes. Both arms strictly load
the exact historical common step0 with its saved Python/NumPy/CPU/CUDA RNG. Effective
batch16, microbatch8, accumulation2,600 updates and all numerical settings unchanged.
Actual sample IDs are now logged after every successful update and checked against
the frozen index sequence. No new backward or update is introduced for logging.

Interpretation is limited to two order seeds, one task and previously tuned development
states. The historical high and low outcomes are both retained as references.
