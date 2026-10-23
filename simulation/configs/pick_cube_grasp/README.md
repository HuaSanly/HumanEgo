# D 场景上的抓取改进

`pick_cube.yaml` 默认继承 `recommended.yaml`，后者继承原始 D。
桌面、颜色、相机、物体尺寸、质量和摩擦都来自 D；调整方块位置和抓取控制。

```bash
python simulation/scripts/run_episode.py --config simulation/configs/pick_cube.yaml --video

# 原始 D 基线
python simulation/scripts/run_episode.py --config simulation/configs/real2sim_pick_cube/D.yaml --video

# 复现参数研究，每组 10 次。必须使用原始 D 作为基线，避免默认配置迭代影响对比。
python simulation/scripts/d_grasp_study.py --config simulation/configs/real2sim_pick_cube/D.yaml \
  --variants simulation/configs/pick_cube_grasp/round1.json --output simulation/runs/d_study_round1 --n 10
python simulation/scripts/d_grasp_study.py --config simulation/configs/real2sim_pick_cube/D.yaml \
  --variants simulation/configs/pick_cube_grasp/round2.json --output simulation/runs/d_study_round2 --n 10
```

推荐配置的主要变化：

- 方块初始 `[x, y, yaw_deg]` 从约 `[0.233, -0.077, 0]` 移到 `[0.40, -0.20, 0]`，单位为米和度。
- 从感知的红块轮廓深度估计夹点，使用已有抓取辅助的到位、闭合保持和空抓重试机制。
- 俯抓世界姿态 RPY 为 `[180, 0, 180]` 度；跟踪时长系数 4，位置平滑 0.3，单步旋转上限 0.08 rad。
- 抓取阈值保留 0.7；关闭 `grasp_latch`，允许策略在放置时重新打开夹爪。

辅助控制只读取现有感知、标定和夹爪状态。评估中的真实接触力不回馈给控制器。
这属于策略加感知引导的控制辅助，不能解释成纯模型能力的提升。
方块位置朝机械臂可达区域移动，图像构图因此与原始 D 不完全相同。

2026-10-23 开发集和新种子复验均为推荐组每组 10 次：
双指持稳抬起 10/10，搬运 10/10，严格入盒 7/10；D 基线抬起 0/10。
新种子是在开发结果选定后固定的，复验后没有继续调参。10 次仍是小样本。
原始日志和汇总：`simulation/runs/d_grasp_study_20261023/`。
