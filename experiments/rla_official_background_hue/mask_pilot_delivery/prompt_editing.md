# 提示编辑与局部修正（下一次授权后执行，本轮不执行）

1. 查看audit_final/semantic_review.json中的具体video_id/frame；原图坐标256×256，x向右y向下。full overlay为四宫格，必须回到原图确定坐标，不直接使用512画布坐标。
2. 用编辑器另存prompts_v3.json；每个object的keyframes包含frame、positive=[[x,y],...]、negative、可选box=[x0,y0,x1,y1]。正点必须明确属于桌面；盘子/碗/机器人/抽屉/炉灶等负点。墙面未可靠标注继续不涂。
3. 保留原提示，记录何人依据何帧作了什么修改；不得把SAM置信度当语义签字。
4. 先检查GPU UUID/ECC/26GiB门槛，再运行同一隔离解释器；--only限制具体视频，--revision必须新名字，已有目录会拒绝覆盖。

```bash
CUDA_VISIBLE_DEVICES=GPU-205a3f5a-d91b-6385-c588-4e4cf3505f7e \
$DATA_ROOT/experiments/sam2-background-pilot-env/bin/python \
experiments/rla_official_background_hue/mask_pilot_run.py \
--run $RLA_ROOT/experiments/rla_official_background_hue/mask_pilot_20261102_01 \
--sam-code $DATA_ROOT/tools/sam2-background-pilot \
--checkpoint $DATA_ROOT/models/sam2.1_hiera_base_plus_background_pilot.pt \
--prompts prompts_v3.json --revision v3_spatial11_wrist \
--only libero_spatial_000011_wrist_camera
```

重新检查固定帧、异常帧及完整视频。先通过语义/对齐再新增registry；不能把整个视频的所有帧继承一个固定帧签字。
