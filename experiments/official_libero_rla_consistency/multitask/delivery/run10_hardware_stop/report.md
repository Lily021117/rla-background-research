# Three-task extension: hardware stop, not completed

## Frozen design and accepted data

The two additional tasks are bowl retrieval from an already-open drawer and bowl retrieval from a stove, followed by placement on a plate. Selection used source completeness, sufficient legal demonstrations and manipulation context, not model outcomes. All three tasks still share bowl pick-and-place semantics; this is not broad primitive coverage.

Each new task passed20 training and5 development episodes, with complete paired dual-view sequences, exact physical/camera/segmentation/depth matching and nonzero background intervention in every view/frame. Splits exclude protected identities before reading arrays. Each task has20 statically checked reset development states; unseen palettes and state assignments were fixed before model scoring. Static reset checks are not policy rollouts or new training demonstrations. They do not prove independence from all historical distributions.

The AE and policy recipes and fixed coefficients are unchanged. Per-task train-only normalization is shared acrossB/C/D. AE updates are fixed at600 per arm, policy updates600 per arm; final checkpoints only. C/D targets must differ only by the defined AE objective. No per-task coefficient recalibration is authorized.

## Hardware interruption

GPU3 correctable ECC increased8→9 (aggregate31→32), triggering the cache guard. Both stove AE processes on that same GPU were stopped. Their logs each contain44 successful updates, but only step0 has a complete recovery checkpoint. Those updates are not a completed600-step result. A potentially in-flight update at interruption is unconfirmed. A recovery must retain these logs and record replay separately from effective final updates.

The stove DINO cache completed on GPU1 without an ECC increment. The drawer cache is partial and not accepted. No sidecars or new policy checkpoints were generated. Policy optimizer updates=0; new closed-loop trials=0. Our GPU processes were released. No device reset or termination of another task occurred. No automatic failover occurred.

## Remaining work

Recover on an explicitly authorized safe device; finish both tasks' fixed AE budgets, representation diagnostics and sidecars, then full policy preflight andB/C/D600 updates each, followed by180 frozen development rollouts per task. None of these unfinished results may be presented as method evidence.

The future replication plan is three matched seeds per task, changing both AE and policy training seeds together. This run authorizes only the first matched seed. Policy-only reseeding estimates stability conditional on fixed AEs, not complete-method replication. State-cluster intervals are conditional on three fixed tasks and cannot establish generalization across LIBERO tasks. Single flips and bootstrap zero bounds are not evidence of stable improvement. Historical cookie unseen backgrounds use one palette; new tasks use a frozen three-palette mix, which must remain explicit in combined summaries.
