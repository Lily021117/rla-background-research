# 官方四套件配对背景色相增强（数据准备，不是训练结果）

官方代码固定为 `mlzxy/rla-wm@0fc6e97212442ebccffa06f1d42f5481e5f22950`。
本目录实现：实际数据覆盖统计、官方 loader 回放、严格掩码接口、在线 RGB 色相变换、独立增强 RNG 与 CPU 测试。
数据准入现状见 `delivery/report.md`：原始数据可读，但可靠背景掩码缺失，因此**真实配对增强未通过，禁止正式训练**。

## 可执行命令

在本分支代码根目录执行；路径通过参数传入，不自动下载或修改原始数据：

```bash
python experiments/rla_official_background_hue/audit.py --root "$DATA_ROOT" --output "$NEW_AUDIT_DIR"
python experiments/rla_official_background_hue/test_pairing.py
python experiments/rla_official_background_hue/verify_official_loader.py --official "$OFFICIAL_CODE" --data "$DATA_ROOT" --output "$RUN_DIR"
python experiments/rla_official_background_hue/prepare_configs.py --official "$OFFICIAL_CODE" --output "$RUN_DIR/configs"
python experiments/rla_official_background_hue/preflight.py --config "$RUN_DIR/configs/AE-Aug-Cons.json"
```

`audit` 要求新目录；`verify` 需先把 audit 产物放在 RUN_DIR/audit。
使用已有 RLA 环境（本地 `.venv/bin/python`），无需安装/覆盖环境。以上均不启动模型训练。
`preflight` 当前应以退出码2拒绝放行。这是可执行准入命令，**不是已完成的正式训练启动器**。
三份 JSON 为完整实验规格，不能直接传给官方 train.py 冒充已集成 paired trainer。
正式 paired trainer 的端到端验收和共同 step0 文件创建留待掩码准入后；本轮不构造虚假的可运行训练承诺。

## 在线接入

`PairedHueReader.sample_training_record()` 调用官方随机 suite→trajectory→start→horizon 采样；调用方持久化 handle 和全局 occurrence（不能用 worker 内计数）。
`read(handle, occurrence, 'train')` 回放官方 RGB，获取逐帧、逐相机的已核验背景 mask，再应用唯一 hue 参数。
验证必须预先保存 handle，使用独立 val seed 和 occurrence=0。
官方 `capture_eval_handle` 内含 Python hash，故不能只保存整数 seed 后宣称跨进程复现；保存完整 handle。

掩码外字节严格不变；颜色变换中 S/V 不变，uint8 转回有量化误差。灰色或空背景不删样本。
mask registry 必须指向真实审查证据，包含 RGB 内容哈希、mask 文件哈希、原 RGB 对齐记录与 table/wall 白名单。
不能把官方 foreground 占位图取反，也不能把本轮合成单测的矩形区域用于正式数据。
不提供启发式“静止=背景”、模型分割或整图 ColorJitter 降级路径。

RGB 两分支必须在变换后分别经冻结 DINO 提取特征。Base 也执行(x,x)两路重建，以匹配物理样本和重建曝光；无改缓存捷径。
每路损失是未来 DINO 的全元素 mean L1 + mean MSE，两路取平均。Cons 额外加入 z 的 mean MSE。
decoder 是特征 decoder；DINO→RGB image_decoder 只做可视化，不是本实验要从零学习的重建目标。

## 从零初始化与恢复防护设计

按官方模型构造规则，仅创建 encoder 与 feature decoder，seed=20261102，decoder zero_init=true。
不得调用 fetch/load 训练过的 AE；DINO 加载预训练但冻结。先保存一份共同模型参数及 CPU/CUDA/Python/NumPy RNG，三组严格加载该文件。
官方 train.py 会把空 load_dir 设为 output_dir 并默认找 latest；未来启动必须新建空目录、明确 `--ckpt none`，不能仅在 YAML 写 load_dir:null。
本轮未创建该 step0，未调用模型，不能称为“作者原始 step0”。

## 限制

120组展示的是原始图与无效的白色占位 mask；增强/差分格明确标为 BLOCKED，不是假装增强通过的图。
源 HDF5/状态与 RLDS 经过 no-op 过滤、重新编号和图像朝向处理后的严格映射尚未建立。
历史 V2 重渲染链路可用于未来映射核查，但不能未经证据替换本任务原始 RGB。
