# HumanEgo 远程推理服务端

服务端已实现双目深度、DINO/SAM2 分割、LaMa 去臂、场景更新、训练约定绘图、原策略推理和官方 LeRobot remote 接入。`policy.py` 是唯一策略实现，代码保持不变。网络、归一化、固定噪声、20 步 ODE、sigmoid grasp 与 done 解码均由它完成。

使用官方 LeRobot revision `5517a326eeb9b457cfe07d59b9bfdf291799c8d9`，日志入口为 `python -m lerobot_policy_humanego.server`，内部使用官方 `PolicyServer`、`SessionWorker` 和 Zenoh 传输，传输、session、reset、请求代际和动作缓冲均由上游负责。上游源码没有修改。首版只支持单右手、absolute、anchor frame、50 步普通 chunk、30 Hz；RTC 和语言功能关闭。

软件验收已使用 RTX 3060、真实训练录制双目、真实 GPU 模型及跨进程 Zenoh 回环完成。BRX 双目基线和校正内参由提供的相机标定计算，EE 与模型 hand 坐标系按约定重合。真机标定、闭环和任务成功率另行验收。

## 环境与安装

独立 Python 3.12 环境，PyTorch 2.7.1 / torchvision 0.22.1 / CUDA 12.6、Transformers 5.4.0。`requirements.in` 固定直接依赖及官方 revision，`constraints.txt` 固定验证环境中的依赖版本。Transformers 5.4.0 配套固定 Hugging Face Hub 1.6.0，避免较新 Hub 严格字段校验与 DINO 配置不兼容。

从项目根目录执行，目标环境必须尚不存在：

```bash
bash inference/install.sh /absolute/path/to/new-inference-env /path/to/python3.12
source /absolute/path/to/new-inference-env/bin/activate
```

安装脚本创建新 venv，不修改训练或仿真环境。SAM2 神经网络使用 CUDA；安装时关闭可选 CUDA 后处理扩展，与本次验证环境一致。模块通过 `inference/pyproject.toml` 映射注册为 `lerobot_policy_humanego`，安装包携带原策略依赖的六个共享源模块，不需要部署训练目录。

本机验证环境名为 `HumanEgo-inference`，可以执行 `conda activate HumanEgo-inference`。

## 配置与导出

参数集中在 `cfg/inference/params.yaml`，其中文件路径统一相对于这个 YAML 解析。内参是**实际传输的 640×480 校正图**的内参；禁止填入未缩放的原始相机内参。

| 文件 | 内容 |
| --- | --- |
| `params.yaml` | checkpoint、感知模型 revision、深度范围、场景超时、绘图与部署设置 |
| `intrinsics.yaml` | 左右校正内参、`image_size_hw: [480,640]`、`rectified: true` |
| `extrinsics.yaml` | 米制 `T_left_from_right`、`T_EE_from_hand` |

`T_left_from_right` 将右目 optical 点变换到左目。首版要求旋转为单位矩阵、右目原点位于左目正 X 轴、两目内参相同且无 skew；投影矩阵与米制深度直接从 K 和此变换构建，不另存一份基线。`T_EE_from_hand` 将模型 hand 坐标变到实际 EE 坐标，组合为 `T_cam_from_hand = T_cam_from_EE @ T_EE_from_hand`。启动时检查 SE(3)、单位、基线方向及缺失字段。实际图像是否完成正确校正，需要相机标定环节保证。

提供的原始相机内参、畸变和左右相机外参记录在 `cfg/inference/stereo_raw_calibration.yaml`，供标定使用，运行时不读取。`extrinsics.yaml` 的水平校正基线 0.060039595 m 是原始平移向量的长度，须核对客户端实际校正映射；左右校正内参由 `stereoRectify`（`CALIB_ZERO_DISPARITY`，`alpha=0`，640×480）的 P1/P2 左上角计算；原始文件同时记录 R1/R2 和 P1/P2。服务端接收已校正图像，不再次校正；上线前须核对客户端映射和实际图像。

```bash
python -m inference.config --config_path cfg/inference/params.yaml
python -m inference.config --config_path cfg/inference/params.yaml --validate-runtime

# 正式标定填妥后导出。已有非空目录会被拒绝覆盖。
humanego-export --config_path cfg/inference/params.yaml --output runs/deploy/humanego-anchor
python -m lerobot_policy_humanego.server --config_path runs/deploy/humanego-anchor/server.yaml
```

导出包含：原 checkpoint 与两个 sidecar（`policy/`）、固定噪声、已解析标定和感知/绘图参数（`assets/`）、原生配置、saved processors、版本和文件哈希清单、官方 `server.yaml`。启动不读取原训练目录。DINO/SAM2/LaMa 按固定 Hub revision 从缓存加载，首次使用允许按配置下载；缓存完整时可设 `HF_HUB_OFFLINE=1`。

移动部署包后，参数、噪声和 processor 路径仍相对于包内解析；修改启动路径即可：

```bash
python -m lerobot_policy_humanego.server --config_path /new/location/server.yaml \
  --model.repo_or_path=/new/location
```

原 checkpoint 的两个 sidecar 原样保存，其中历史训练数据路径只是训练元数据，推理不会访问它们。请使用 `humanego-export` 准备部署产物。

## 实时运行日志

使用上面的日志入口启动，部署包 `server.yaml` 中 `log_level: INFO` 即可。已有部署包可直接使用，无需重新导出；更改代码后需重启服务。安装更新后的插件还可使用 `humanego-policy-server --config_path <server.yaml>`。

同一 `request` / `session` / `generation` 关联以下事件：

| 事件 | 含义 |
| --- | --- |
| `observation_received` | 官方协议已解码观测；列出观测 ID、feature 形状及 dtype，尚不表示通过校验 |
| `inference_started` / `observation_validated` | worker 开始处理；数据校验通过，输出实测 EE 位置和夹爪状态 |
| `perception_completed` / `policy_started` | 感知完成及耗时；开始策略预测 |
| `policy_completed` | done、耗时、grasp 范围、预测首末动作；仍是完整 50 步预测摘要 |
| `action_ready` | 官方 runner 整理后的实际动作块已生成，列出形状和处理耗时 |
| `action_published` | 已调用 Zenoh 发布，列出请求、动作形状、字节数和官方阶段耗时；`delivery_ack=false` 表示没有客户端接收或执行确认 |
| `inference_failed` / `error_published` / `reply_publish_failed` | 推理异常、向客户端发布错误或传输发布失败 |

启动预热标记为 `warmup_probe`，不计为收到客户端观测。原始 `lerobot-policy-server` 入口仍可使用，但不包含新增的收包、请求关联及发布日志。日志默认输出到终端，不保存完整图像或动作数组。

## 详情诊断模式

在部署包 `assets/params.yaml` 中配置，重启日志入口后生效：

```yaml
deployment:
  diagnostic_dir: /home/huasan/HumanEgo/runs/inference-diagnostics
  diagnostic_mode: full
```

`canvas` 为默认模式，只保存画布及投影 JSON；`full` 保存关键过程数据，目录为 `run_<时间>_<实例>/<session>/<generation>/<request>/`，预热后的文本日志保存到该 run 目录的 `server.log`。`diagnostic_dir: null` 关闭文件诊断。导出仍会清空诊断目录，须在实际部署包中显式设置。

| 文件 | 数据 |
| --- | --- |
| `request.json`、`config.json` | 请求标识、采集时间、处理状态、异常及所用配置和标定 |
| `input/` | 原始左右 RGB 图、完整实测 `state.npy` |
| `detections/`、`masks/` | DINO 选框与分数、投影提示框、物体/机械臂/保护 mask |
| `stereo/` | 实际计算时的视差、米制深度、有效 mask、物体三维点；稳定阶段跳过深度时明确记录 |
| `lama/`、`clean.png` | 有修复时的有效 mask、512×512 LaMa 输入/输出及最终 clean 图；空 mask 会标记跳过 |
| `scene.json`、`render_model_canvas.*` | 固定参考、物体与手位姿、抓取关联、渲染画布、投影范围及绘点/裁剪统计 |
| `model/` | 实际 RGB 图与原始 `[1,3,240,320]` float32 tensor、ICT、bool mask、anchor UV、固定噪声、原策略输出和 done |
| `output/` | 完整 50 步解码预测、runner 截取后的动作及实际发布动作、阶段耗时 |
| `publication.json` | 最近一次发布的成功/失败、错误、字节数、时间及 `delivery_ack=false` |
| `publications/*.json` | 每次发布事件，保留重复请求的错误回复历史 |

处理中途失败保留已完成阶段的数据，不会补造未运行阶段文件；协议校验阶段即被拒绝的请求仅有错误发布记录和文本日志。NPY/NPZ 使用 `numpy.load(..., allow_pickle=False)` 读取。详情模式同步写图、压缩数组和复制 GPU 输入，会增加延迟和磁盘占用；结束排查后改回 `canvas` 或关闭目录。诊断写入失败会明确报日志，不改变策略结果。

## 客户端接口

通信、观测与动作格式、坐标及 EE 约定见 [客户端适配接口](CLIENT_INTERFACE.md)。

## 模块和执行链

| 模块 | 职责 |
| --- | --- |
| `config.py` / `data_types.py` | 配置检查、数据与场景容器；不构造模型或硬件 |
| `stereo.py` | SGBM128/窗口5、左右一致性、0.15–2.5m 深度过滤、内侧轮廓采样与 MAD 过滤 |
| `perception.py` | 常驻 DINO/SAM2/LaMa、复用训练 `pca1`、物体更新及抓取传播 |
| `rendering.py` | 1624×1232 训练画布、局部 -Y 夹爪、训练配色和点序 |
| `processor_humanego.py` | 输入检查、感知编排、场景 reset、saved processor 加载 |
| `configuration_humanego.py` | 原生 PreTrainedConfig 与固定能力声明 |
| `modeling_humanego.py` | 一个 ICTPolicy 实例、设备/噪声管理、原生动作块整理 |
| `export.py` | 自包含部署产物和官方 server.yaml |

```text
同步 RGB 双目 + 实测右 EE 状态
→ 检查尺寸、有限值、SE(3)、夹爪范围
→ 实测 EE × T_EE_from_hand
→ 机械臂分割；初始化/释放时执行物体检测和原始双目几何
→ 去臂、LaMa、保护可见物体、训练画布绘图
→ 原 policy.prepare_image() → 320×240 RGB 模型图
→ 原 build_ict(固定 reference) / compute_anchor_uv(当前 obj1)
→ 原 infer() / decode_ee_in_cam(同一固定 reference)
→ float32 [1,50,10]，连续 grasp 不变
```

初始化必须获得 obj1 蓝色容器和 obj2 红块，每个物体至少 8 个有效三维关键点。初始 obj1 被独立复制为 reference，之后物体位置更新不会改写它。默认 `object_update_mode: events`：初始化分别检测 obj1、obj2、机械臂，共三次 DINO；之后每帧仅检测机械臂一次，并跳过双目深度计算。该模式要求未抓取物体保持静止；外力移动物体后需要 reset 重新建立场景。`continuous` 可恢复逐帧定位。

只有 obj2 可被抓取关联：实测夹爪闭合且模型 hand 与物体距离小于 0.20m 时锁定相对变换，遮挡期间用实测手位姿传播；释放后保留末次位置并恢复视觉定位。释放触发 obj2 重新检测和双目定位；失败时逐请求重试，超过 `object_ttl_s`（默认 1 秒）明确失败。主动跳过检测的静止物体不计为视觉丢失，也不伪造新的观测时间。`continuous` 模式仍保留原来的逐帧定位与 1 秒丢失超时。

每帧 SAM 只编码一次 RGB 图，再处理物体和双臂/夹爪框。稳定阶段的物体框由当前三维关键点投影产生，因此物体保护 mask 仍逐帧分割，不复用过时的像素坐标；抓取时投影随实测 EE 移动。物体选择对应单实例，去臂取所有候选的 mask 并集。LaMa 使用 BGR、512×512、9×9 膨胀两次及训练侧 Gaussian 融合；修复区域保护可见物体。空去臂 mask 跳过修复并记录 `empty_mask`，不会记录为 `inpainted`。

官方启动零输入预热只在首次 reset 之前可用，不创建场景。预热执行真实感知模型和原策略计算；启动 reset 后永久关闭这条探测分支。之后即使任务字符串为 `warmup`，非法零位姿也会正常失败。reset 保留模型资源与固定噪声。

DINO 与 SAM 默认分别使用 `segmentation.dino_precision: float16`、`sam_precision: float16` 的局部 autocast；原策略和 LaMa 保持 FP32。可设置 `float32` 回退，`bfloat16` 仅用于显式实验。旧部署包缺少这些配置时保持原 FP32/continuous 行为，需要重新导出才能启用优化。

日志包含初始化、当前请求的 `localized_objects`、关键点数、mask 像素、阶段耗时和 done。`gripper_projection` 输出夹爪绘点数、深度裁剪数、画外点数及投影范围；无可见点时为 WARNING。`deployment.diagnostic_dir: null` 默认不逐帧写图；显式设置目录后才保存模型画布及同名 JSON，包含接收的 EE 位姿、模型 hand 位姿、画布内参和绘制统计。

## 保留行为和边界

训练 grasp target 与原推理 sigmoid 的关系仍待真机开合样本核验。本轮验证样本的连续输出约在 0.50–0.73，服务端没有通过修改算法或阈值处理这个现象。done 仅记录日志。几何/软件回归不代表 BRX 标定精度或任务成功率。

仿真继续共享 `inference.policy.ICTPolicy` 和兼容的 `ObjectState`。旧相机辅助代码保存在 `simulation/humanego_sim/legacy_inference/`，新服务端不依赖它们。
