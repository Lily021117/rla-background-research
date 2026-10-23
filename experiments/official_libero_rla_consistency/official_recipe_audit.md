# Official recipe provenance

Status: PARTIAL; strict released AE checkpoint load passed, paired end-to-end admission has not passed.

Official repository: https://github.com/mlzxy/rla-wm
Pinned commit: `0fc6e97212442ebccffa06f1d42f5481e5f22950`.
All source paths below are relative to this immutable checkout.
Integration root: `rebuttal/exp2_vla_adapter/`.
Setup pins upstream VLA-Adapter `23fa0c9c159e2aa04341cdd3e924f44061311060` and LIBERO `8f1084e3132a39270c3a13ebe37270a43ece2a01`.
Dependencies: torch 2.2.0, torchvision .17, transformers 4.40.1, tokenizers .19.1, timm .9.10, peft .11.1, numpy 1.26.4, TF2.15, flash-attn2.5.6. AE has a separate existing environment; do not upgrade the policy environment.

## Policy

Read README, setup/setup_env.sh, overlay tools/train_eval.sh and tools/train_eval_rla.sh, rla/loss.py, rla/sidecar.py and upstream finetune.py.
Official initialization is pretrained `Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b`, not an already LIBERO-trained Spatial-Pro policy. Fresh action/proprio modules and rank64 LoRA are initialized in upstream finetune; the rest is pretrained VLM, not full random initialization.
Recipe uses Pro/minivlm, two images, language and proprio, L1 action regression, image augmentation, LR2e-4 and shuffle buffer80000.
RLA Stage1 is RLA-only20K (four suites); Stage2 action+RLA10K per suite. Stage1 adds real compute/data exposure.
Actual launchers differ in default accumulation: vanilla8/GPU×accum2 versus RLA8/GPU×accum1; recorded README baseline configurations may use accum1. Do not assert identical effective budgets without freezing overrides.

B sets RLA_STEPS=0 and does not install RLA structure. RLA extends the Pro sequence by128 latent tokens (8×16), alongside8 action tokens. Action and RLA attention/head structures therefore differ. Auxiliary supervision trains shared policy features; targets are never observation inputs. RLA token computation remains in the inference graph; predicted latent output can be discarded.

## Autoencoder and target

`configs/rla/16x64_libero.yaml`: DINOv3-L1024,224px, agentview+wrist, joint16×64 latent, 12blocks/16heads/1024width, Plücker positional inputs in encoder and decoder. Tokens are joint query tokens, NOT sixteen tokens per camera.
`rebuttal/src/trainers/rla_autoencoder_multiview_trainer.py`: inference_batch and training_losses (lines202–207) implement feature L1+MSE; subtraction is future minus current. Each appearance reconstructs its own future.
AE config uses AdamW1e-4/weight_decay0,batch128,max1M updates, two frames sampled within horizon64; config load_dir is null. This is not proof that all reported weights trained1M steps.
Official sidecar uses anchored h1…8, not eight copies of h8.
Extractor _reference_batch lines631ff and prepare_views lines875ff use actual source geometry, INTER_LINEAR AE resizing and cam-to-world matrices under historical key w2c. Current-frame rays condition the pair.
Published archive: https://huggingface.co/datasets/xyzhang368/RLA-WM/blob/44401dff2016a4aa66aaa99fc67e09987d5790db/rebuttal_ckpt_exp2_libero_rla.tar
1473515520bytes; SHA256 `24e177f9bcdda3ff0d73789c59987c9aed9bb618629130a4afa6fe931878b44e`, verified against LFS.
Released step90000 Encoder and Decoder strictly loaded on CPU with zero missing/unexpected keys and finite tensors. The paper's step40000 is not in this archive. Neither step90000 nor local step750 is an exact substitute for it.

## Auxiliary weighting

`vla_adapter_overlay/rla/loss.py:57–99`: auto:0.25 means
`lambda_eff=.25*stopgrad(action_L1)/max(stopgrad(rla_L1),1e-6)`.
Total is lambda_act×action_L1+lambda_eff×rla_L1. This gives auxiliary/action loss ratio .25 when unclamped, not gradient-norm matching or25% of total.
`rla/sidecar.py:RlaStore._scaling`: center subtracts per-coordinate mean, divides by a global std, not each coordinate std.
C/D different distributions imply different scales AND potentially different per-step effective coefficients. Same auto setting does not isolate consistency alone.

