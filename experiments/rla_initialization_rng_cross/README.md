# C initialization × training randomness crossover

Queued only after the fixed-AE policy-seed replication finishes and publishes.
New cells: historical initialization + seed29 training RNG/order, and seed29
initialization + historical training RNG/order. Existing diagonal cells are reused.

Each new cell: 600 updates, then 20 development states × 3 appearances. Frozen AE-C,
original LR/schedule/auxiliary coefficient, no teacher/keep or new losses. The saved
step0 parameter tensors and Python/NumPy/CPU/CUDA RNG states are composed separately;
optimizer state is fresh. No trained checkpoint supplies initialization.

Run in the existing isolated environment with `CUDA_VISIBLE_DEVICES=''` for the
controller; the audited launcher selects only authorized GPU0, checks 26 GiB and ECC.

```
python experiments/rla_initialization_rng_cross/queued_experiment.py --out <NEW_RUN> --mode prepare
python experiments/rla_initialization_rng_cross/queued_experiment.py --out <NEW_RUN> --mode run
```

The new experiment never changes an in-flight predecessor. Any numerical failure
stops the queue rather than changing configuration or silently replaying. Comparison
is exploratory; it does not select a winning seed or prove a unique mechanism.
