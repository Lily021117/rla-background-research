# 修复演进与未覆盖项

1. 旧主重放在控制步边界保存 `time/qpos/qvel`，直接渲染尚未同步的几何缓存。用相同数组重新加载并 `forward` 会改变body/camera变换。741帧离线证据位于 `offline/`。
2. 为核对cached obs，包装Observable原sensor；首次Object包装漏拷贝 `__modality__`，在一个等待步物理执行后抛错。修复为 `tracked.__modality__ = orig.__modality__`。没有再次执行Object源动作；失败计数的 `wait_steps:0` 是返回后计数器未加一，交付明确校正为实际1，不改原failure.json。
3. Spatial一次完整捕获后，保存NPZ使用了重复的 `sensor_geometry` 名称。保留完整CSV/时间记录/视频/同步状态；缺少附加原始多路径数组。修复记录键为 `sensor_geometry_transforms`，不重放Spatial。
4. Spatial运行时 `MjData` 对XML重编译模型的恢复仍有微小几何/像素误差；Goal/10改为原运行时 `mj_saveModel` MJB快照，恢复几何和RGB精确相等。最终四条离线颜色渲染也统一使用精确MJB。
5. Object/Spatial缺少此前MJB：只构建相同BDDL环境、seed0、reset，核对导出的XML哈希与原记录完全相同，然后保存编译模型。`MjSim.step`被显式禁用，不重放源动作、不等待。
6. 早期Object颜色诊断通过XML编译两模型，保留在 `color/`，不作为最终物理模型精确性证据。最终 `color_final/` 从MJB复制模型，只修改专用背景纹理数据。全部非纹理数组逐字节相同；可见geom来自渲染器的GEOM实例分割。背景纹理若被前景共用则拒绝，当前四场景没有这种别名。
7. 复制模型审计的初次浮点 `array_equal` 遇到源模型已有NaN（Object mesh_polynormal有60个）而失败，尚未渲染。修复为精确字节比较（不是放宽容差），保留失败记录在 `color_v2/`。

颜色参数、来源轨迹、帧索引和原视频对应阈值没有根据结果改变。新颜色方案在任何本轮图像检查前冻结。最终A是同一同步Clean的自身保存/恢复，而B始终直接与作者原视频和RLDS机器人状态比较；不通过传递性间接判断B。

Object四路径的运行时记录未补齐；不能将其他套件的2ms/4ms测量冒充Object实测。其完整143帧仍完成旧状态恢复、同步Clean、精确模型保存/恢复、双视角原片比较与颜色/分割检查。
