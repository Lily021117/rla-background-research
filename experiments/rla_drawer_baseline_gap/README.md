# Open-middle-drawer capability and appearance-gap study

Fixed development study: BC-standard and frozen scratch AE-C-low, full policy
seeds 31001/31002/31003. Six fresh policies, 1800 updates each. LR 2e-4 for
updates 1–600, then 2e-5; fixed auxiliary coefficient 0.11204268292682927.
Only final1800 is scored. Each method must attain Clean >=16/20 on **all three
seeds** before its 240 background trials. Maximum 600 valid trials overall.

Use `RLA_PROJECT` and `DRAWER_GAP_ROOT` to locate local assets; see frozen protocol
for exact identities. Existing accepted drawer data are read-only. AE-C final2000
is a frozen representation learned on three bowl tasks, **not** drawer opening.
Original standard/RLA architecture differences are preserved and disclosed.

Execution: `prepare.py`, then `publish.py protocol`, then `run.py` in the validated
policy environment. Sidecar extraction uses the isolated sidecar environment.
The pipeline stops on any numerical/infrastructure failure and never retries a
valid task failure. No AE/DINO update, sealed-data read, or automatic search.

The adapters assert exact source replacements against versioned accepted
modules, and save effective-source hashes. All graph, action processing and
drawer deployment logic are reused. Source retains upstream repository licenses.
No weights, images, states or sidecars belong in the Git repository.
