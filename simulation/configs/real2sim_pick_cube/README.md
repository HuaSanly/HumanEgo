# pick_cube 的 Real2Sim 消融配置

这些配置来自 2026-10-23 的工作区快照，保留该时刻的模型与控制器设置。
后续按用户要求统一将抓取阈值设为 0.7，同时在 `pick_cube.yaml` 增加任务级覆盖；
原始 0.75 版本和结果仍保留在实验目录，`serve_bread.yaml` 保留用户原有设置。

| 配置 | 变化 |
|---|---|
| A | 当时的完整场景与控制器快照 |
| B | A + 暖白桌面、浅蓝塑料、区域颜色拟合、灯光、背景和定位标记、训练分辨率关键点 |
| C | A + 重建估计尺寸、圆角锥壁盒、训练内参、观察相机和名义布局 |
| D | B + C |
| E | D + 批量实验脚本中的有界视角与光照变化 |

`E.yaml` 单独用于 `run_episode.py` 时是 D 的名义场景；随机化由
`scripts/real2sim_ablation.py` 的逐 episode 配置实现。

相机变化只改变观察相机，保持机器人底座世界位姿不变，并重算手眼外参。
盒子尺寸约 15.2 × 23.0 × 9.6 cm，方块边长约 3.35 cm：来自可用训练 episode
中 17 组过滤后的轮廓三角化估计，尚未经过尺量确认。
同一实体道具的尺寸、质量和摩擦在 E 中保持固定。

```bash
python simulation/scripts/run_episode.py --config simulation/configs/real2sim_pick_cube/D.yaml --video

# 重新统计与拟合，使用新的输出目录；A 从当前 pick_cube 配置保存。
python simulation/scripts/real2sim_calibrate.py --output simulation/runs/my_real2sim
python simulation/scripts/real2sim_ablation.py --output simulation/runs/my_real2sim --n 10 --workers 3
# 同样参数重新运行会跳过已落盘的试验；执行错误保留在 result.json，单独统计。
```

原始实验目录为 `simulation/runs/real2sim_ablation_20261023/`：

- `baseline/`：原始配置、源代码、全部场景资产、模型副本与哈希、原有未提交 diff。
- `implementation/`、`implementation_manifest.json`：实验实现快照与哈希。
- `training_statistics.json`：配置的 73 个训练路径中，本地可用的 25 个 episode，包含缺失和异常记录。
- `protocol.json`：冻结的配置哈希、试验数、随机种子和并发设置。
- `episodes/{A,B,C,D,E}/epNNN/`：每次完整轨迹、终态、配置与结果；前两次还保存策略输入图。
- `comparison.png`、`visual_metrics.json`：名义场景与训练参考的实际模型输入对比。
- `summary.json`、`report.md`：严格成功率、物理里程碑、Wilson 区间、配对 bootstrap 与 Holm 校正。

每组默认运行 10 次。用户将原先 100 次的计划缩减为 10 次后，按预先编号选取
每组 `ep000`–`ep009`，不按成功或失败筛选；已多跑的结果保存在 `extra_trials/`，
原计划保留在 `protocol_history.json`。10 次结果用于初步比较，置信区间会较宽。

阈值 0.7 的重新评估在 `simulation/runs/real2sim_ablation_20261023/threshold_0p7/`，
仍为 A–E 各 10 次，场景编号与随机种子与 0.75 轮次相同。

恢复原始设置时，可从 `baseline/simulation/configs/` 复制两份 YAML；
原始仿真源码在 `baseline/simulation/humanego_sim/`，完整资产在 `baseline/assets/`。
还原前先保存后续修改；不要用 Git reset 丢弃其他工作。
