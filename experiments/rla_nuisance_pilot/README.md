# Frozen C/D nuisance pilot

No optimizer, AE/DINO update or backward. Six fixed stove step1800 policies.
Select25 previously evaluated states by SHA256("nuisance_pilot_v1:"+decimal ID).
Three new background pairs and three static lighting parameter sets, selected
by metadata hash and technical rendering only. No whole-environment replacement.
900 new rollouts; 150 matching historical Clean trials reused, not rerun.

CPU `prepare.py` freezes candidates, hashes and100 balanced development anchors.
`run.py` validates selected-state renderings, commits/pushes the protocol, renders
saved demonstration states without integration, runs frozen AE/policy checks,
and the exact historical220-step/20Hz/chunk8 deployment. Then paired-state and
per-demonstration statistics are generated and sanitized outputs published.

Primary: equal-three-background/equal-three-policy-seed D-C success difference.
Secondary: equal-three-static-lighting difference. No adverse-C gate. These are
reused states/new appearance exploratory results, not new sealed tests.
Existing negative stove/drawer conclusions are preserved. No follow-up training.
Use the accepted separate sidecar and policy environments; all caches and large
artifacts remain outside Git under the independent run directory.
