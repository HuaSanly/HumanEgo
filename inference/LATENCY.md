# 推理延迟与阶段优化（2026-10-27）

本轮在 RTX 3060 12GB、现有 Python 3.12 / PyTorch 2.7.1 CUDA 12.6 推理环境中测量。输入为录制双目 frame 100，每眼 RGB uint8 640×480；测试去臂提示词为 human arms / human hands，不能代表 BRX 外观的识别效果。

## 行为调整

`cfg/inference/params.yaml` 默认使用 `object_update_mode: events`，DINO / SAM 均为 `float16` 局部 autocast。

| 阶段 | DINO 次数 | 双目深度 | 物体状态 |
|---|---:|---|---|
| 初始化 / reset 后首帧 | 3 | 计算 | 确认 obj1、obj2，复制固定 reference |
| 稳定运行 | 1（双臂和夹爪） | 跳过 | 未抓取物体保持；被抓物体通过实测 EE 传播 |
| 释放后重新确认 | 2（obj2、双臂和夹爪） | 计算 | 定位 obj2；成功后回到稳定运行 |

重新定位失败时允许短暂沿用传播后的末次位置，并在之后请求重试；超过 1 秒失败。主动跳过定位的静止物体不因旧时间戳超时。未完成重新定位的 obj2 不允许再次建立抓取关联。reference 始终保持初始化值。

该模式要求未抓取物体不被外力移动。物体意外移动后须 reset；也可以显式配置 `continuous` 恢复原来的每帧定位。没有增加新的网络字段。

SAM 仍每帧编码一次图像。稳定阶段用当前物体三维关键点投影生成框，重新分割物体保护 mask；不会固定使用首次像素位置。此分割不重新定位物体，也不额外调用 DINO。投影提示下完全遮挡场景的 mask 质量仍需 BRX 实测。

LaMa 继续使用 FP32、512×512、原膨胀和融合方式。原 `policy.py` 没有修改；20 步 ODE、50 步动作、连续 grasp 均保持。

## 同一输入的分阶段测量

每组先运行 2 次，再统计 10 次。计时脚本在每个阶段前后同步 CUDA，以免异步工作被误计入后续阶段；这些同步只存在于诊断脚本，不进入正常服务。下表单位为 ms，均值；初始化不计入稳定阶段。

| 阶段 | 原 continuous FP32 | events FP32 | events FP16 |
|---|---:|---:|---:|
| DINO 网络（各次合计） | 989.9 | 327.7 | 234.8 |
| DINO 图像 / 文本预处理 | 75.1 | 22.9 | 25.5 |
| SAM 图像编码 | 116.5 | 115.8 | 51.6 |
| SAM mask 解码 | 21.4 | 22.2 | 14.0 |
| 双目 SGBM | 32.5 | 0 | 0 |
| 三维关键点 | 2.4 | 0 | 0 |
| LaMa（含准备 / 融合） | 209.6 | 209.9 | 212.1 |
| 绘图 | 1.0 | 0.9 | 1.0 |
| 原策略 ODE | 56.2 | 55.1 | 57.1 |
| 图像准备 / ICT / anchor / 动作解码 | 4.3 | 4.3 | 4.6 |
| 其他（输入处理、场景、同步等） | 33.6 | 18.8 | 19.5 |
| **总计** | **1542.6** | **777.5** | **620.1** |
| **总计 P95** | **1564.5** | **782.1** | **636.2** |

先消除重复物体检测，收益大于仅降低精度。当前剩余主要耗时为 DINO 和 LaMa，原策略约 57 ms。

## 实际官方跨进程回环

前后分别启动原生 `lerobot-policy-server`，同一录制输入，官方 `RemoteClient` + `ChunkRuntime`，每组接收 10 个动作块。首块需要建立场景，后 9 块统计稳定延迟。没有逐阶段 CUDA 同步，也没有连接电机。

| 指标 | 优化前 | 优化后 |
|---|---:|---:|
| 首块（场景初始化，模型已预热） | 1526.9 ms | 1183.4 ms |
| 稳定请求均值 | 1513.7 ms | 588.1 ms |
| 稳定请求 P95 | 1525.1 ms | 604.3 ms |
| worker 外开销均值 | 6.8 ms | 7.0 ms |

稳定回环均值减少 **61.1%**，约 **2.57 倍**加速。worker 外开销是客户端计时减去官方 queue / worker 时间，包含编解码、传输及客户端轮询，不能视作纯网络传输时间。30 Hz 是动作执行频率；每块 50 步覆盖约 1.67 秒，并不表示每 33 ms 完成一次模型请求。

本轮前后均在同一可编辑源码环境顺序测量；不以此前不同批次的历史耗时作为对照。

两组均通过持续补货、错误 schema、错误 payload、reset 确定性重放、旧 generation 拒绝、失效结果丢弃、session close 和 SIGTERM 停止。回环前后导出包各自保留自己的固定噪声；数值精度比较则使用下面的同一个原始 bundle 和同一固定噪声。

## 数值与质量检查

- 同一 bundle / 输入 / 噪声，continuous FP16 相比 continuous FP32：红块和双臂 mask IoU 为 1；容器为 0.99995；最终动作逐元素相同。
- 同一条件，events FP16 相比 continuous FP32：容器 / 红块 / 双臂 mask IoU 分别为 0.97627 / 0.99874 / 1；最终动作逐元素相同。容器差异来自投影框提示，不能解释为重新定位精度。
- BF16 实验中，最终动作最大位置差异约 2.36 mm，rotation6d 分量最大差异约 0.00977；因此默认采用 FP16。
- 独立真实 GPU 验证记录了 6 次请求的 DINO 次数 `[3,1,1,1,1,1]`，通过可见物体像素保护、双臂并集、单次 SAM 编码、原策略逐元素一致及 reset。CPU 回归 26 项、仿真参考系和感知几何回归 27 项全部通过。
- 上述结论限于当前录制样本，不构成机器人任务成功率或全数据集精度结论。

## 复现与产物

在仓库根目录、`HumanEgo-inference` 环境中运行：

```bash
HF_HUB_OFFLINE=1 python tests/inference/benchmark_latency.py \
  --bundle runs/inference-validation/bundle-final \
  --observation runs/inference-validation/recording/observation.npz \
  --object-update-mode continuous --requests 10 \
  --output runs/inference-validation/latency-precision-new

HF_HUB_OFFLINE=1 python tests/inference/benchmark_latency.py \
  --bundle runs/inference-validation/bundle-final \
  --observation runs/inference-validation/recording/observation.npz \
  --object-update-mode events --requests 10 \
  --output runs/inference-validation/latency-events-new

HF_HUB_OFFLINE=1 python tests/inference/run_remote_validation.py \
  --bundle runs/inference-validation/bundle-latency \
  --observation runs/inference-validation/recording/observation.npz \
  --chunks 10 --output runs/inference-validation/remote-latency-new
```

`benchmark_latency.py` 默认依次比较 FP32、FP16、BF16；可用 `--variants float32/float32 float16/float16` 指定 DINO/SAM 精度组合。每组报告包含阶段均值/P95、mask IoU、位置及 grasp 差异，并保存动作和 mask。

本轮本地产物（均不入 Git）：

- `runs/inference-validation/latency-precision/report.json`
- `runs/inference-validation/latency-events/report.json`
- `runs/inference-validation/remote-latency-before/report.json`
- `runs/inference-validation/remote-latency-after/report.json`
- `runs/inference-validation/bundle-latency/`：优化后的录制样本测试部署包，使用独立测试标定。

BRX 左右目外参是未实测的 URDF 名义候选值，内参及 hand 变换仍为 null。旧 bundle 的资产不会随源码配置更新；正式标定完成后按 README 重新导出，才能启用 events / FP16。
