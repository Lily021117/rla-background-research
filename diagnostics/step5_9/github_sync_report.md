# GitHub 同步验收

- 登录身份：GitHub CLI 实际`api user`返回`Lily021117`，账号类型User。此前连接的其他账号未参与本次创建、提交或推送。
- 仓库：https://github.com/Lily021117/rla-background-research ，已核验PRIVATE。此前本项目remote属于作者公开仓库，不用于写入；已有个人公开项目不用于混放本研究。
- 新分支：`diagnostics/step5-9`。新空仓库中的研究快照分支，没有覆盖已有默认分支，没有force push。
- 已推送源码快照commit：`4b7255a0c2e327df46041bd849313685f38c4c32`。本文件及最终验收会作为后续文档提交；最终branch HEAD以本地`github_remote_verification.json`及远端记录为准，避免在commit内自引用自身SHA。
- 初始上传153个文件、约1.09MB文本。包括实际unified model/objective、Math SDPA trainer、真实dataset/target构造、FK/DINO/encoder target生成谱系、闭环入口、配置、必要VLA源码及本轮聚合诊断。不是仅上传报告。
- `source_inventory.csv`记录源SHA与脱敏后SHA；只机械替换服务器路径和GPU UUID为占位符。README明确本地化步骤、外部数据/checkpoint/官方overlay依赖及测量的运行环境版本。没有宣称在无数据的下载环境端到端复跑通过。
- 保留VLA-Adapter原MIT许可证及作者来源。官方RLA审计checkout无root LICENSE，其核心源码未擅自重新授权/打包；README给出固定commit与外部依赖。自定义研究源码未因本次私有归档被另行公开授权。
- 排除checkpoint、数据集、图像/视频、原始轨迹、逐窗口物理标签、缓存、认证配置、SSH文件及所有密钥。只分享聚合诊断CSV/JSON。服务器绝对路径已替换；无公司内网remote或地址。
- 上传前检查白名单、UTF-8/AST、文件大小、密钥模式、服务器路径及全Git对象历史。使用新独立快照仓库，未携带原项目历史，原项目及已有worktree的分支/remote/config均未更改。私有GitHub仓库由当前确认个人账号持有。
- 第一次本地预览快照未推送；发现过宽的`data/`忽略规则会漏掉源码适配器后，创建新的最终快照，规则锚定根目录并核验`source/data/sparse_rla`完整进入提交。第一次预览仍保留本地，不进入上传历史。最终快照还移除了无关机器人平台示例。
- Git凭据仅通过CLI的credential helper在本次git命令中使用，没有输出token、读取浏览器/VS Code凭据或更改项目/global Git认证配置。

本地完整诊断包含逐试验/逐转移证据；仓库按约定只包含可分享代码与脱敏聚合文档。
