# Frozen stove D failure diagnosis

This directory diagnoses the existing `run_20261023_10/stove` run. It does not train AE or policy models, run a simulator control loop, alter targets, or read sealed demonstrations. Closed-form ridge readouts are fitted only on the permitted training episodes.

Keep the source run read-only. Choose a new `OUT` directory under the data volume. All scripts use exclusive-create for evidence tables. Source dependencies remain the exact versions recorded in the existing run: RLA `0fc6e97212442ebccffa06f1d42f5481e5f22950`, VLA upstream `23fa0c9c159e2aa04341cdd3e924f44061311060`, and the accepted derived loader/STOP correction. No original recipe implementation is patched.

## Environment and order

Set `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TORCH_HOME`, `XDG_CACHE_HOME`, `TMPDIR`, `CUDA_CACHE_PATH`, `TRITON_CACHE_DIR`, `MPLCONFIGDIR` to independent data-volume paths before Python. Set `PYTHONDONTWRITEBYTECODE=1`, `TOKENIZERS_PARALLELISM=false`, and limit BLAS threads to four. CPU stages use `CUDA_VISIBLE_DEVICES=''`.

From this repository, with `SRC=experiments/official_libero_rla_consistency/stove_diagnosis`:

```bash
"$AE_PYTHON" "$SRC/prepare.py" --source "$SOURCE_STOVE" --out "$OUT"
"$AE_PYTHON" "$SRC/checkpoints.py" --out "$OUT"
"$VLA_PYTHON" "$SRC/representation.py" --out "$OUT"
"$AE_PYTHON" "$SRC/target_scale.py" --out "$OUT"
"$VLA_PYTHON" "$SRC/rollouts.py" --out "$OUT"
```

`AE_PYTHON` is the existing PyTorch2.8 environment for trusted full checkpoint loading. `VLA_PYTHON` is the accepted PyTorch2.2/TensorFlow2.15/MuJoCo environment; do not install packages into the historical environments. No MuJoCo renderer or GPU is used for the CPU label/trajectory checks; only `mj_forward` on saved states.

For the policy-only diagnostic, independently verify the authorized GPU UUID, memory and ECC baseline. The script enforces26GiB free at its own start, UUID equality, Math SDPA and ECC-increment stopping. Set `CUDA_VISIBLE_DEVICES` and `RLA_GPU_UUID` to the same explicitly permitted device. Set `RLA_STEPS=8`, `RLA_QUERIES=16`, `RLA_DIM=64`, `RLA_NORM=center`, `LAMBDA_ACT=1`, `LAMBDA_RLA=0.22408536585365854`, `RLA_LOG_EVERY=0`, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

```bash
"$VLA_PYTHON" "$SRC/policy.py" --out "$OUT" --result-name policy_verified
```

This constructs the accepted official RLA model, uses the actual derived loader and official forward/loss definitions, strictly reloads the existing trainable state over the verified frozen Prismatic initialization, and never instantiates an optimizer. It skips the historical bootstrap reference batch entirely. It records separate action and weighted auxiliary gradients on trainable VLM parameters, not shared head blocks; these are deterministic eval-mode derivatives, not replayed stochastic training gradients.

The delivered run contains two identical passes (`policy` and `policy_verified`),160 backwards each. The second logs the raw auxiliary loss directly before BF16 coefficient multiplication, correcting a first-pass logging approximation. Actions and gradient norms match exactly; float64 cosine reductions over unordered parameter-name sets differ by at most6.7e-16. This is diagnostic repetition, not additional updates. `finalize.py` validates these two receipts; for a new single-pass run, preserve a new finalizer variant instead of fabricating a repeat receipt.

```bash
"$VLA_PYTHON" "$SRC/finalize.py" --out "$OUT" --delivery "$NEW_SANITIZED_DELIVERY"
```

The finalizer rehashes protected inputs and requires the diagnostic GPU process to have exited. It publishes only sanitized report/configuration/small-summary files. Checkpoints, sidecars, images, video panels, full trajectories, and local per-step timing data must stay out of Git.

## Actual execution paths and interpretation

* Training identity: `ae_consistency/policy_training.py:initialize,execute`, `multitask/prepare_runs.py`, source `policy_run/run_plan.json`, `sampling_order.json`, saved checkpoint `configuration`, and sidecar provenance.
* Real batch: `coverage_gripper/dataset.py:DerivedEpisodeDataset.sample`; official `rla.sidecar:RlaStore.frame` provides targets only. Every batch checks stove language tokens and56 valid action tokens.
* Policy supervision: official `finetune.run_forward_pass` action L1; `rla.loss.make_run_forward_pass` adds fixed-weight RLA L1. The diagnostic computes the same two components independently.
* AE: `ae_consistency/common.py:forward`, `train_ae.py`, `evaluate_and_extract.py`. The reconstruction table is reused on precisely the existing400 development transitions; no new AE forward is needed.
* Deployment: `ae_consistency/policy_evaluation.py:execute,infer`; source per-trial state/XML/checkpoint hashes and recorded chunk/postprocessed commands are checked. Original phase classifications are preserved.

Normalized action errors are dimensionless. Decoded translation/rotation errors are controller-command units, not metric displacement/angle. Gripper threshold accuracy is separate. Raw/unit-normalized/common-C-centered latent distances answer different questions and are all reported. Squared-energy fractions of target means must not be interpreted as auxiliary L1 or gradient fractions.

Observed cached rollout body/site positions differ from recomputation of recorded poststep states by up to0.980mm. New contact checks are explicitly same-state recomputations, not measured forces or a silently corrected old timeline. Training FK labels passed their separate same-state check. No physical labels are inferred from RGB.

The proposed target-recentering intervention is documentation only. Do not run it from this diagnostic package.
