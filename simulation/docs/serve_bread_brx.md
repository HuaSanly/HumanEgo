# BRX serve_bread 正前方布局与抓取实验

配置：`simulation/configs/serve_bread_brx.yaml`，默认右臂单臂。

- 整机底座 `[0.01361, -0.20, -0.32104]`，保持直立、面向世界 +X。
- 头部 `pan_deg: 0`、`tilt_deg: 57`，物理相机为 `right/head_view`。
  仅在本场景中调整固定头部姿态，原 pick_cube 场景和导入模型保持原值。
- 桌高 0.96 m，桌面世界 z=0.20 m；外置 D405 高度为世界 z=0.566 m。
  策略继续使用外置 D405，头部相机用于预览和录像。
- 面包默认 `[0.46, -0.17, 95]`，盘子 `[0.56, 0.10, 0]`，单位为米/度。
  物体尺寸、质量、软体参数和摩擦沿用 serve_bread。
- 场景专用 `rig.initial_joints.right` 对齐原 Trossen 在策略相机中的起始
  EE 位姿，回归检查要求位置差 <0.2 mm、旋转差 <0.002 rad。
  原 pick_cube 的 BRX 待机姿态在这一布局中会让手位于很右侧，且靠近关节限位。

官方权重、网络、动作解码、`T_align` 和 IK 算法均未修改。沿用 production
控制参数：抓取阈值 0.70、原有 latch、自动抬升和 idle-done。
`grasp_assist`、`release_assist`、`withdrawal` 均关闭。
权重 SHA-256：`7a90fee7dad1feb2a27c8ee6580f9811f42cdcc3836c77eafee17ab27be01385`。

## 9 个位置的实际结果

每个位置运行 25 s，环境种子 0、策略噪声种子 7，面包 yaw 均为 95°；
初始物体静置，位置实验关闭开场掉落动画。盘子、待机姿态和其他参数固定。

稳定夹起要求面包比初始中心高至少 2.5 cm，且两指对软面包的法向接触力均
大于 0.1 N，连续至少 0.1 s；另检查抬起持续至少 0.2 s。接触力只用于评估，
不反馈给策略或控制器。`grasped` 状态可能在空夹时为真，不能单独作为成功依据。

| 编号 | 面包 x | 面包 y | 最大抬升 cm | 双指持稳抬升最长时间 s | 稳定夹起 | 放盘成功 |
|---|---:|---:|---:|---:|---|---|
| 0 | 0.41 | -0.22 | 33.7 | 19.58 | 是 | 否 |
| 1 | 0.46 | -0.22 | 32.4 | 19.34 | 是 | 否 |
| 2 | 0.51 | -0.22 | 27.4 | 19.00 | 是 | 否 |
| 3 | 0.41 | -0.17 | 13.5 | 17.83 | 是 | 否 |
| 4 | 0.46 | -0.17 | 15.0 | 17.00 | 是 | 否 |
| 5 | 0.51 | -0.17 | 19.3 | 17.32 | 是 | 否 |
| 6 | 0.41 | -0.12 | 12.7 | 15.95 | 是 | 否 |
| 7 | 0.46 | -0.12 | 14.0 | 16.13 | 是 | 否 |
| 8 | 0.51 | -0.12 | 13.9 | 16.43 | 是 | 否 |

25 s 内夹起 9/9，放盘 0/9。搬运/放置阶段仍有每回合 2–107 次 IK 拒绝。
按实际 CLI 默认开场动画和 40 s 时限另跑中心位置，也能夹起，但到时仍持有
面包，未完成放盘。这是固定位置、单一噪声种子的开发实验，不是泛化成功率。

## 同位置对照

选择编号 0、4、8，保持同一布局、模型权重、相机、环境/噪声种子和 25 s 时限：

| 执行配置 | 稳定夹起 | 放盘成功 |
|---|---:|---:|
| BRX，沿用 pick_cube 原生待机姿态 | 0/3 | 0/3 |
| BRX，对齐原 Trossen 起始手部位姿 | 3/3 | 0/3 |
| Trossen，原 production 控制 | 2/3 | 1/3 |

BRX 两组只改变场景初始关节角。因而即使模型和推理代码不变，起始输入位姿
仍然会显著影响结果。另一次独立的机械检查用显式运动目标夹住并抬高面包
11.6 cm、双指持稳超过 2 s，验证了夹爪和接触模型；该检查不计入策略成功率。

## 运行与复查

从仓库根目录，在包含仿真依赖的环境中运行：

```bash
conda activate GlassEgo
python simulation/scripts/run_episode.py --config simulation/configs/serve_bread_brx.yaml --viewer

# 保存头部视角视频，策略输入仍为外置 D405。
python simulation/scripts/run_episode.py --config simulation/configs/serve_bread_brx.yaml \
  --video --camera_view right/head_view --out serve_bread_brx_head_s0

# 新输出目录；每个位置保存轨迹、策略输入、接触力及结果。
python simulation/scripts/serve_bread_brx_sweep.py --duration 25 \
  --out simulation/runs/serve_bread_brx_positions_01
```

脚本保存解析后的配置、权重哈希、种子、位置和判据，固定策略噪声并顺序运行。
它拒绝覆盖已有输出目录。原始实验与录像保存在本地忽略的 `simulation/runs/`：

- `serve_bread_brx_front_matched_grid_dev`：9 个位置的结果与接触记录。
- `serve_bread_brx_native_ready_control_dev`：同位置的原生待机姿态对照。
- `serve_bread_matched_trossen_control_dev`：同位置的 Trossen 对照。
- `serve_bread_brx_matched_ready_center_dev`：中心位置及 `grasp_front.mp4`、`grasp_head.mp4`。
- `serve_bread_brx_front_default_s0`：默认 CLI 的完整 40 s 回合。

九位置主组与 Trossen 对照是在加入 `protocol.json` 保存功能前启动的，
因此这两组目录只有配置、结果及轨迹，无法从各自目录独立核验运行时权重哈希；
上文哈希对应本次使用的本地权重。原生姿态对照及之后的新实验会保存完整协议。
