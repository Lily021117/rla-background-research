# Fixed scratch-AE low-weight policy seed replication

Only C-low and D-low, three new complete policy initialization/training seeds,
600 successful updates each and 600 paired development rollouts. Frozen official
Prismatic backbone is strictly loaded; random modules follow the established
baseline-to-RLA constructor recipe. Constant-zero LoRA/gates stay zero. One shared
step0 and RNG snapshot per seed is loaded by both arms. AE-C/D formal step2000,
existing own train-only target normalization and lambda0/2 remain frozen.

`prepare.py` audits registered seeds and hashes existing assets, generates the
unchanged episode-balanced indices and freezes all five evaluation conditions.
`run.py` uses GPU UUID/ECC/26GiB guards, serial initialization, full-batch zero-update
preflight, six trainings, all 600 evaluations, analysis and sanitized publication.
No automatic retries or device switching. Complete training checkpoints include
optimizer/scheduler/RNG/sample cursor. CUDA errors stop rather than silently resume.

Training and evaluation functions reuse the preceding accepted implementation;
exact asserted source replacements and their hashes are recorded. Training changes
only C-low source recognition and fixed half weight. Evaluation changes only the
finite admitted budget. No future target enters action inputs.

Python environment, upstream imports, licenses and data placeholders are inherited
from the repository. Run prepare/analyze/publish with the existing analysis Python;
run the supervisor with the established VLA environment. All artifacts stay on the
data disk. Do not upload weights, raw states, images, videos or target caches.

Report each seed independently, sample SD/range across three seeds, and paired
state-cluster intervals separately (not training-seed confidence). Historical two
sampling orders are references only, not extra training seeds. No independent-test
or cross-task generalization claims. Stop after this fixed budget.
