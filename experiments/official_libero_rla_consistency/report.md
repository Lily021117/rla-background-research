# Route revision delivery

Status **PARTIAL**. Official baseline provenance and released LIBERO AE checkpoint verified; formal paired training **BLOCKED** pending concrete interface gates, not authorized by this report.

Author components: dual-camera Plücker AE16×64, anchored8 targets, pretrained Prismatic policy initialization, official vanilla/RLA overlay and two-stage loss.
New components: paired-appearance reconstruction wrapper, latent-MSE consistency, variant-isolated store selector, custom background dataset and evaluation. No BC-ResNet port, Spatial-Pro repair, action-consistency or physical loss in this route.

Custom paired data cannot currently be described as losslessly compatible: complete horizons missing,128px source, malformed K export and unverified image/ray orientation, FK-vs-raw join difference. Strict CPU loading of released90000 passed; paired AE forward and full VLA pipeline are not yet accepted.
This is materially different from Step5.7: official16×64 multiview/h1…8 instead of ManiSkill32×64/h8; pretrained VLM rather than finished Spatial-Pro; official overlay rather than unified wrapper; declared stage budgets rather than100-update warmstart. None proves it will solve previous policy degradation.

See data_interface_audit.json and sidecar_alignment_audit.json for measured counts, checkpoint_acceptance.json for strict loading, and next_training_plan.md for gates.
Optimizer updates=0; backward=0; new rollouts=0; held-out data reads=0. Old branches/data/checkpoints remain in place. No GPU process was started.

