#!/usr/bin/env bash
set -euo pipefail
cd ${RLA_PROJECT_ROOT}/worktrees/rla-background-causal-unified-trainer
export RUN_DIR=${RLA_PROJECT_ROOT}/experiments/rla_background_invariance/step5_7_causal_training/run_20261022_01
export TMPDIR=${RLA_CACHE_ROOT}/step5_6
export HF_HOME=$TMPDIR/hf HF_HUB_CACHE=$TMPDIR/hf/hub HUGGINGFACE_HUB_CACHE=$TMPDIR/hf/hub
export TORCH_HOME=$TMPDIR/torch XDG_CACHE_HOME=$TMPDIR/xdg MPLCONFIGDIR=$TMPDIR/mpl
export CUDA_CACHE_PATH=$TMPDIR/cuda TRITON_CACHE_DIR=$TMPDIR/triton TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor NUMBA_CACHE_DIR=$TMPDIR/numba
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBERO_CONFIG_PATH=${LIBERO_ROOT}/.libero
export PYTHONPATH=$PWD:${LIBERO_ROOT}
export CUDA_VISIBLE_DEVICES=''
${VLA_REFERENCE_ROOT}/.venv-cpu/bin/python prepare_acceptance.py > "$RUN_DIR/prepare.log" 2>&1
export CUDA_VISIBLE_DEVICES=${GPU_UUID}
${VLA_REFERENCE_ROOT}/.venv-cpu/bin/python train_math_step57.py > "$RUN_DIR/training.log" 2>&1
