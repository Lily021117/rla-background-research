# Frozen C-low V2 development evaluation

Six existing final1800 policies, two tasks, 20 historical development states each, four fixed revision2 interventions. No training, backward, AE loading or target extraction. New valid rollout budget480;120 exact historical Clean results are reused after identity checks.

`prepare.py` verifies weights/state hashes and copies exact V2 material/texture/light definitions by semantic names into historical Clean XMLs. It never modifies physics, source states or policies. `identity.py` verifies the two historical inference/proprio functions have identical ASTs except a local variable name. `evaluation.py` reuses those function bodies, checks actual ControlEnv reset configuration and scores current online observations. It restores the initial state only once per trial. No prototype frames or demonstration actions are used online.

Before scoring: freeze all protocol files, publish sanitized copies with `publish.py protocol`, commit and record that commit and local protocol hash in `protocol_commit.json`. Launch `run.py` in the accepted environment with `RLA_PROJECT_ROOT` and `V2_EVAL_ROOT` set. GPU UUID/free memory/ECC checks are mandatory; the serialized runner stops on infrastructure failure rather than silently retrying. Successful and failed task trials are both retained as valid.

The statistical unit is the physical initial state, retaining every policy seed and condition in each bootstrap draw. Tasks are reported separately. The intervals exclude full training-seed, AE, condition-sampling and task uncertainty. No10pp/three-seeds-same-direction gate is used. This evaluation cannot establish whether augmentation training solves a deficit or whether D adds value.

After completion use `analyze.py`, then inspect its full results and add evidence-bounded decision text. `publish.py final` exports only sanitized small results; weights, states, XMLs, images and videos remain local. `policy_forward_attempts` counts calls to the historical `predict_action` path, normally one per8-action chunk; it is not a count of internal neural-network layers.
