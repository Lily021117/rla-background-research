# C episode-balanced sampling stability

Two sampler order seeds, 20261101/20261102, chosen before new outcomes. Both arms
strictly load the SAME historical step0 including model RNG. No AE or D updates.
600 successful updates/arm, original schedule and losses, followed by60 development
rollouts/arm. No dynamic extension or checkpoint selection.

Each of240 cycles visits each of20 training episodes once in a shuffled order.
Within each episode, exhaust a shuffled permutation of eligible windows before
reshuffling. Exactly240 physical exposures per episode and40 per last100 updates.
Clean/background each occur once per physical window; effective batch16 unchanged.
Only an independent NumPy RandomState generates the sampler. Uniform episode
weighting differs from historical uniform window weighting: this is a sampling
scheme intervention, not simply a reorder of an identical multiset.

Historical orders are verified against algorithm reconstruction, checkpoint config
and cursor, and600 logged updates, but absent per-item logs are not invented. The
only trainer instrumentation change records actual successfully consumed sample
IDs without changing tensors, RNG, gradients or update semantics.

```
python experiments/rla_sampling_stability/experiment.py --out <NEW_RUN> --mode prepare
python experiments/rla_sampling_stability/experiment.py --out <NEW_RUN> --mode run
```

Use the existing isolated environment and data-disk caches. Controller is CPU-only;
the reused launcher selects authorized GPU0 with26GiB and ECC checks. On failures,
stop and preserve evidence; no silent retry/configuration change. Source/history
remain read-only. Reports are exploratory on reused development states.
