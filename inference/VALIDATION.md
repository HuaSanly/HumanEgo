# 服务端实现验收记录

验证日期：2026-10-26。实现分支：`codex/humanego-remote-inference`。

## 环境与产物

- 独立环境 `HumanEgo-inference`：Python 3.12.15、PyTorch 2.7.1+cu126、torchvision 0.22.1+cu126、Transformers 5.4.0、Hub 1.6.0、SAM2 1.1.0、ONNX Runtime GPU 1.26.0。
- GPU：NVIDIA GeForce RTX 3060，12 GB；驱动 595.99.02。
- LeRobot：`5517a326eeb9b457cfe07d59b9bfdf291799c8d9`。已将安装环境中的全部 LeRobot Python 文件与此 revision 的源码逐一比较，内容无差异。
- checkpoint：`HumanEgo_anchor/latest.pt`；严格加载原权重，20 步 ODE、单右手、absolute、50 步输出。
- DINO / SAM2 / LaMa Hub revision 分别为 `a2bb814dd30d776dcf7e30523b00659f4f141c71`、`7c218beaf0bb87874785f32b582f640134fc1c09`、`c3c0c9e468934d62e79c329e35d82dd09ff8c444`。
- 最终安装包：`runs/inference-validation/wheels-delivery/lerobot_policy_humanego-0.1.0-py3-none-any.whl`。只映射原目录和必要共享源文件，不包含另一份模型实现。
- 测试部署包：`runs/inference-validation/bundle-final/`；这是 **PICO 录制回放测试标定**，不能用作 BRX 正式标定。

生产 `cfg/inference/intrinsics.yaml`、`extrinsics.yaml` 的未测矩阵仍为 `null`，运行检查会拒绝启动。既有训练/仿真环境未升级；用户原有两份仿真配置和 `get-docker.sh` 的内容哈希与实施前一致。

## 验证结果

| 验证 | 结果 |
| --- | --- |
| 配置、输入边界与原骨架迁移检查 | 11 passed |
| 合成双目、场景生命周期、训练绘图对照 | 10 passed |
| 适配器连续 grasp / 非有限预测拒绝 | 3 passed |
| 现有仿真参考系与感知几何回归 | 27 passed |
| 实际 DINO/SAM2/LaMa CUDA 与原策略 | 通过，10 次正式请求 |
| 官方服务端进程 + Zenoh + 测试 Robot | 通过，连续 4 个动作块、136 个 30 Hz 动作记录 |
| 安装包在项目外运行、部署包迁移重载 | 通过 |
| `ruff check`、编译检查、`pip check`、shell 语法检查 | 通过 |
| 原 `ICTPolicy` AST 与实施前比较 | 相同；本轮 `policy.py` 无修改 |

真实感知使用 `spatial_pick_cube_000` 的原始双目录制第 100 帧，按各目相机元数据缩至 640×480，内参同步缩放。该录制包含人手，因此独立测试参数使用 `human arms . human hands .`；生产参数仍使用机器人双臂/夹爪 prompt。回放中没有连接电机。

测得物体有效关键点分别为 16 和 19，去臂 mask 非空。验证覆盖 RGB/BGR 转换、可见物体像素保护、空 mask 跳过、镜像增强双臂图像上的 mask 并集，以及每帧 SAM 图像编码恰好一次。双臂增强测试是图像级验证，不代表已验证 BRX 双臂外观。

适配器与直接调用原 `ICTPolicy` 使用同一已准备观测和同一固定噪声，输出最大绝对误差为 **0**；连续 grasp 逐值相同，样本范围约 **0.500008–0.731536**。工具变换、固定 reference 编码与解码保持一致。训练绘图对照复用训练侧夹爪分段和绘点函数，对开、闭两种状态逐像素比较通过。

场景测试覆盖双物体初始化、缺失初始化失败、物体移动、固定 reference 不漂移、只有 obj2 可抓取、闭合后遮挡传播、释放和视觉重定位、未抓取物体丢失超时。失败更新不提交半完成场景。

官方回环覆盖 lossless raw 双目、canonical float32 `[50,10]`、持续补货、错误 admission schema、错误网络观测 shape、reset 后确定性重放、过时代际请求拒绝、本地失效结果丢弃、session close 和 SIGTERM 正常退出。测试使用官方 `RemoteClient` 与 `ChunkRuntime`；测试 Robot 仅提供录制观测、记录动作。

迁移测试使用正式 wheel，在项目外工作目录载入复制的部署包，对原训练树的 `open()` 访问设置失败检查。模型、固定噪声及 processor 相对路径重载通过；模型与 noise 经 CPU/CUDA 往返后设备一致，eval 状态和确定性输出保持。正式 wheel 还在项目外重复完成真实 GPU 请求和官方跨进程回环。

## 延迟

下表取最终 wheel 在项目外的测试，感知权重已在 Hub 缓存中。初始化包含权重装载与两次全模型启动预热；逐请求耗时包括感知、立体几何、LaMa、绘图、原策略 20 步推理。采样量较小，仅供本机首版预算评估。

| 测量 | 样本 | 平均 | P95 |
| --- | --- | --- | --- |
| 完整 GPU 核心链路 | 10 请求 | 1.203 s | 1.232 s |
| 官方跨进程回环 | 4 请求 | 1.217 s | 1.246 s |
| 核心初始化及预热 | 1 次 | 14.669 s | — |

50 步／30 Hz 的动作范围为约 1.667 秒。本次普通 chunk 回环完成持续补货，客户端测试设置 `refill_seconds=1.65`、`max_observation_age_s=10`、`action_starvation_grace_s=3`。这些是录制回放测试参数，实际网络、场景变化和新客户端控制节拍需要重新测量。没有验证真机任务成功率。

详细结果与诊断图：

- `runs/inference-validation/gpu-delivery/report.json`、`clean.png`、`canvas.png`、`model.png`、`arms.png`、`dual_arm_mask.png`。
- `runs/inference-validation/remote-delivery/report.json`、`server.log`。
- `runs/inference-validation/relocation-release/report.json`。

大文件、图像、权重和日志保留在忽略的 `runs/` 下，不进入 Git。

## 复现

从项目根目录开始，激活独立推理环境。导出与安装目标使用新目录，已有非空导出不会被覆盖。

```bash
conda activate HumanEgo-inference
python -m pytest inference/test_scaffold.py inference/tests -q
ruff check inference --exclude policy.py --exclude build
python -m compileall -q inference
python -m pip check
bash -n inference/install.sh

# 独立测试输入与标定，来源是训练录制元数据；不会回填生产配置。
python inference/tests/make_recording_fixture.py \
  --source data/pick_cube/spatial/spatial_pick_cube_000/spatial_source.json \
  --params cfg/inference/params.yaml \
  --output runs/inference-reproduce/recording
humanego-export --config_path runs/inference-reproduce/recording/params.yaml \
  --output runs/inference-reproduce/bundle

HF_HUB_OFFLINE=1 python inference/tests/run_gpu_validation.py \
  --bundle runs/inference-reproduce/bundle \
  --observation runs/inference-reproduce/recording/observation.npz \
  --output runs/inference-reproduce/gpu --requests 10
HF_HUB_OFFLINE=1 python inference/tests/run_remote_validation.py \
  --bundle runs/inference-reproduce/bundle \
  --observation runs/inference-reproduce/recording/observation.npz \
  --output runs/inference-reproduce/remote

python -m pip wheel --no-deps --wheel-dir runs/inference-reproduce/wheels ./inference
python -m pip install --no-deps --target runs/inference-reproduce/installed \
  runs/inference-reproduce/wheels/lerobot_policy_humanego-0.1.0-py3-none-any.whl
HF_HUB_OFFLINE=1 python inference/tests/run_relocation_validation.py \
  --installed runs/inference-reproduce/installed \
  --bundle runs/inference-reproduce/bundle --training-root runs/pick_cube \
  --output runs/inference-reproduce/relocation
```

仿真回归使用原有含 pytest 和仿真依赖的环境：

```bash
conda run -n GlassEgo python -m pytest \
  simulation/tests/test_reference_frames.py simulation/tests/test_perception_geometry.py -q
```
