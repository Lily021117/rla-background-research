# Frozen step1800 independent-state admission

This is a metadata-only audit, not an evaluation runner. It opens registered
candidate/use manifests and hashes the six exact final checkpoints. It never
loads sealed state arrays, model weights into a model, or an environment.

Run with the local project root (artifacts are intentionally not distributed):

```bash
python experiments/frozen1800_state_readiness/audit.py \
  --project /path/to/local/project \
  --output /path/to/new/output
```

Base commit: `442448c56c6eb075a895692033798395c36d8da8`.
The previous delivery was successfully pushed to `experiments/rla-low-lr-continuation`
and its remote ref matched that exact commit. The failed push record was preserved;
the local run has a separate append-only synchronization recovery record.

The audit does not certify the remaining candidates merely because no overlap was
found in selected manifests. Official init indices cannot be treated as source
demonstration IDs without evidence. A source crosswalk is necessary before pooling
the two registries. No scoring protocol was frozen or executed because admission
did not establish the minimum of 20 independent sources.
