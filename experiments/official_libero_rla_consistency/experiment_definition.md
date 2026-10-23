# Controlled experiment definition

Data: 基于 LIBERO 示范构建的跨外观配对数据.
Evaluation: 标准 LIBERO and 基于 LIBERO 的自定义背景材质干预协议, separately. NOT official LIBERO-Plus.
Old20 states are exploratory development; no independent-test claim.

B: official vanilla VLA-Adapter without RLA structure.
C: official VLA+RLA targets from AE-C: mean(clean reconstruction, background reconstruction).
D: same VLA+RLA as C, targets from AE-D: identical reconstruction plus joint-latent MSE.
Both appearances retain their own future feature targets. No cross-camera token matching, future-token equality, action-output consistency, physical loss, or independent dynamics model.

All policies receive identical paired observations, labels, language, sample order and cumulative policy update budget. B versus C estimates the full official RLA integration including head differences; C versus D isolates the target-producing intervention only if normalization/effective coefficient confounds are explicitly controlled.
AE-C/D share initialization/RNG/trainable scope/data/update count. Default official AE configuration starts from shared random initialization; released90000 is a reference checkpoint, not silently the training start. Using it as common continuation would require a named warm-start variant.
Implementation in paired_ae.py delegates inference to the official trainer. It is not yet certified against recovered paired camera inputs and must not launch training.

