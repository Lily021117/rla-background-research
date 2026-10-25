# GPU0 continuation pause — partial result

D-center resumed from full step200 with unchanged training semantics and successfully computed updates201–396. C remains complete at600. D's latest complete checkpoint is300, verified on CPU including optimizer/scheduler/sampling position; 301–396 are logged but not persisted and must be counted as replay if resumed. No substitute weight-only restart is permitted.

Training stopped advancing while a GPU0 status query blocked in the driver (`os_acquire_rwlock_write`). Only this task's trainer and status query received SIGTERM. The cause is unresolved: no evidence establishes OOM or a new ECC error. Device-memory release could not be verified while the query was blocked. No other process was terminated and no device reset was performed.

This attempt:196 successful training computations,0 rollout attempts. Across the continuation:C200+D196=396 successful computations, including16 known C replays;380 previously unfinished logged update indices. D has96 additional unpersisted computations. An in-flight397 is unconfirmed and excluded. AE updates0, teacher updates0, extra diagnostic backward0, sealed-test reads0. Training backward was performed as authorized.

The requested120 valid development rollouts have NOT run. No method conclusion, final600 D comparison, or success-rate claim is available. Resume only after GPU safety is re-established, from full D300 state with original1200-step scheduler and original sampling/RNG, targeting600. Do not repeat C or silently disregard the96 replays.

Historical protected inputs were rehashed unchanged; see delivery_verification.json. Failed attempts and all checkpoint/log evidence are retained.
