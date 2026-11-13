# HumanEgo 远程推理服务端

服务端已实现双目深度、DINO/SAM2 分割、LaMa 去臂、场景更新、训练约定绘图、原策略推理和官方 LeRobot remote 接入。`policy.py` 是唯一策略实现，代码保持不变。网络、归一化、固定噪声、20 步 ODE、sigmoid grasp 与 done 解码均由它完成。

使用官方 LeRobot revision `5517a326eeb9b457cfe07d59b9bfdf291799c8d9`。`humanego-policy-server` 直接读取项目配置并构造 `PolicyRunner`，使用官方 `PolicyServer`、`SessionWorker` 和 Zenoh 传输。支持单右手、absolute、anchor frame、50 步普通 chunk、30 Hz。RTC 和语言功能关闭。

此前版本已在 RTX 3060 上使用真实训练录制双目和 GPU 模型完成跨进程 Zenoh 回环。BRX 双目基线和校正内参由提供的相机标定计算，EE 与模型 hand 坐标系按约定重合。真机标定、闭环和任务成功率另行验收。

## 环境与安装

独立 Python 3.12 Conda 环境，PyTorch 2.7.1 / torchvision 0.22.1 / CUDA 12.6、Transformers 5.4.0。`inference/requirements.txt` 固定直接依赖、已验证的间接依赖版本及官方 revision。Transformers 5.4.0 配套固定 Hugging Face Hub 1.6.0，避免较新 Hub 严格字段校验与 DINO 配置不兼容。

先创建并激活专用 Conda 环境，再从项目根目录执行：

```bash
conda create -n humanego-inference python=3.12 -y
conda activate humanego-inference
bash setup.sh inference
```

安装脚本向当前激活的 Conda 环境安装依赖；请勿在训练或仿真环境中运行。SAM2 神经网络使用 CUDA；安装时关闭可选 CUDA 后处理扩展，与本次验证环境一致。模块通过 根目录的 `pyproject.toml` 映射注册为 `lerobot_policy_humanego`，安装包携带原策略依赖的六个共享源模块，不需要部署训练目录。

本机验证环境名为 `HumanEgo-inference`，可以执行 `conda activate HumanEgo-inference`。

## 配置与启动

参数集中在 `cfg/inference/params.yaml`，其中文件路径统一相对于这个 YAML 解析。内参是**实际传输的 640×480 校正图**的内参；禁止填入未缩放的原始相机内参。

| 文件 | 内容 |
| --- | --- |
| `params.yaml` | checkpoint、感知模型 revision、深度范围、场景超时、绘图与部署设置 |
| `intrinsics.yaml` | 左右校正内参、`image_size_hw: [480,640]`、`rectified: true` |
| `extrinsics.yaml` | 米制 `T_left_from_right`、`T_EE_from_hand` |

`T_left_from_right` 将右目 optical 点变换到左目。首版要求旋转为单位矩阵、右目原点位于左目正 X 轴、两目内参相同且无 skew；投影矩阵与米制深度直接从 K 和此变换构建，不另存一份基线。`T_EE_from_hand` 将模型 hand 坐标变到实际 EE 坐标，组合为 `T_cam_from_hand = T_cam_from_EE @ T_EE_from_hand`。启动时检查 SE(3)、单位、基线方向及缺失字段。实际图像是否完成正确校正，需要相机标定环节保证。

提供的原始相机内参、畸变和左右相机外参记录在 `cfg/inference/stereo_raw_calibration.yaml`，供标定使用，运行时不读取。`extrinsics.yaml` 的水平校正基线 0.060039595 m 是原始平移向量的长度，须核对客户端实际校正映射；左右校正内参由 `stereoRectify`（`CALIB_ZERO_DISPARITY`，`alpha=0`，640×480）的 P1/P2 左上角计算；原始文件同时记录 R1/R2 和 P1/P2。服务端接收已校正图像，不再次校正；上线前须核对客户端映射和实际图像。

```bash
python -m inference.config --config_path cfg/inference/params.yaml --validate-runtime
humanego-policy-server --config_path cfg/inference/params.yaml
```

服务直接加载 `policy.ckpt` 及同目录的 `config.json` 和 `dataset_stats.json`。`policy.noise_seed` 默认是 `0`，用于生成每次启动和 reset 后都复用的固定噪声。需要使用以前保存的噪声时，在 `policy.noise_path` 指定对应的 `.npy`。修改配置后重启服务。

DINO、SAM2 和 LaMa 按固定 Hub revision 从缓存加载。缓存完整时可设 `HF_HUB_OFFLINE=1`。

## 实时运行日志

在 `cfg/inference/params.yaml` 中设置 `deployment.log_level`。服务启动时在 `deployment.log_dir` 下建立独立的 `run_*` 目录，默认位于 `runs/inference/`。`server.log` 从启动开始记录文本日志。预热前写入一份 `config.json`，包含解析后的配置和标定、权重及配套文件的路径和 SHA-256、固定噪声摘要，以及远程服务使用的模型标识。

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

终端和 `server.log` 会提示权重加载、预热完成及服务退出。`DEBUG` 级别可查看每次 `warmup_probe`。终端和 `server.log` 也记录请求摘要，完整图像与动作数组可通过详情诊断保存。

## 详情诊断模式

在 `cfg/inference/params.yaml` 中配置，重启服务后生效：

```yaml
deployment:
  diagnostic_dir: ../../runs/inference-diagnostics
  diagnostic_mode: full
```

`canvas` 为默认模式，保存画布及投影 JSON。`full` 保存关键过程数据，目录为 `run_<时间>_<标识>/<session>/<generation>/<request>/`，与日志目录的运行名称一致。`diagnostic_dir: null` 关闭逐请求诊断，启动配置快照和文本日志仍会保存。

| 文件 | 数据 |
| --- | --- |
| `request.json` | 请求标识、采集时间、处理状态和异常 |
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
| `processor_humanego.py` | 输入检查、感知编排、场景 reset、组装内存中的处理器 |
| `configuration_humanego.py` | 原生 PreTrainedConfig 与固定能力声明 |
| `modeling_humanego.py` | 一个 ICTPolicy 实例、设备/噪声管理、原生动作块整理 |
| `server.py` | 读取配置、构造并预热 LeRobot runner、写入运行配置快照、启动服务 |

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

初始化必须获得 obj1 蓝色容器和 obj2 红块，每个物体至少 8 个有效三维关键点。初始 obj1 被独立复制为 reference，之后物体位置更新不会改写它。默认 `object_update_mode: events`、`joint_object_detection: true`：初始化联合检测两个物体，再单独检测机械臂，共两次 DINO，并共用一次图像预处理。联合提示按各物体的 token 范围分别计分，机械臂仍使用独立提示。之后每帧仅检测机械臂一次，并跳过双目深度计算。该模式要求未抓取物体保持静止；外力移动物体后需要 reset 重新建立场景。`continuous` 可恢复逐帧定位。

只有 obj2 可被抓取关联：实测夹爪闭合且模型 hand 与物体距离小于 0.20m 时锁定相对变换，遮挡期间用实测手位姿传播；释放后保留末次位置并恢复视觉定位。释放触发 obj2 重新检测和双目定位；失败时逐请求重试，超过 `object_ttl_s`（默认 1 秒）明确失败。主动跳过检测的静止物体不计为视觉丢失，也不伪造新的观测时间。`continuous` 模式仍保留原来的逐帧定位与 1 秒丢失超时。

每帧 SAM 只编码一次 RGB 图，再将物体和双臂/夹爪框批量解码一次，按框所属类别分别合并 mask。SAM 保留 1024 输入。稳定阶段的物体框由当前三维关键点投影产生，因此物体保护 mask 仍逐帧分割，不复用过时的像素坐标；抓取时投影随实测 EE 移动。物体选择对应单实例，去臂取所有候选的 mask 并集。LaMa 使用 BGR、512×512、9×9 膨胀两次及训练侧 Gaussian 融合；修复区域保护可见物体。空去臂 mask 跳过修复并记录 `empty_mask`，不会记录为 `inpainted`。

官方启动零输入预热只在首次 reset 之前可用，不创建场景。预热执行真实感知模型和原策略计算；启动 reset 后永久关闭这条探测分支。之后即使任务字符串为 `warmup`，非法零位姿也会正常失败。reset 保留模型资源与固定噪声。

DINO 与 SAM 默认分别使用 `segmentation.dino_precision: float16`、`sam_precision: float16` 的局部 autocast，原策略和 LaMa 使用 FP32。可设置 `float32` 回退，`bfloat16` 仅用于显式实验。

默认 DINO 保留 800 短边、完整编码器/解码器和 900 queries。`dino_cuda_kernel_revision` 固定融合可变形注意力算子的 Hub commit，使用 `kernels==0.12.3` 加载匹配 Torch 2.7 / CUDA 12.6 的预编译算子；采样和累加输入统一为 FP32，避免 autocast 混合 dtype 导致错误。首次运行需联网填充算子缓存，完整缓存后支持 `HF_HUB_OFFLINE=1`。将 revision 设为 `null` 可使用原 PyTorch 算子。

LaMa 默认 `static_batch: true`、`cuda_graph: true`：固定 batch=1，在启动预热时捕获 CUDA Graph，后续请求更新固定 GPU 输入缓冲并回放；输出同样绑定固定地址。ORT 按调用线程缓存 Graph，因此启动预热和正式请求都通过同一个 LaMa 执行线程，避免 worker 首请求重新捕获。reset 保留这些缓冲，但每次请求都会更新图像和 mask。该实例与场景处理器一样按请求串行使用。

2026-11-02 在本机 RTX 3060 上，对四帧真实机械臂双目画面各测 10 次稳定请求，另测首次场景定位，并比较 mask、物体位置、修复图与完整动作。阶段计时使用 CUDA 同步，不含网络与文件诊断；首次定位不含服务启动预热。详细结果保存在 `runs/inference-validation/optimization/`。此对比的基线已使用 FP16 和 events，线上 full 诊断还会产生额外耗时。测试过的 DINO 缩图、减少 queries、LaMa 半精度和 SAM 缩图配置未作为默认值，原因是漏检、修复伪影或动作偏差。

| 测量项 | 原算子/逐物体 DINO | 当前配置 |
| --- | --- | --- |
| 本地首次场景定位，四帧平均 | 1098 ms | 645 ms |
| 本地稳定请求，40 次平均 | 595 ms | 478 ms |
| DINO，每次稳定请求 | 231 ms | 135 ms |
| LaMa，每次稳定请求 | 203 ms | 180 ms |

上表基线也已共用 DINO 图像预处理和批量 SAM 解码；更改前的原始 FP16 稳定请求另测为 614 ms。四帧中初始物体 mask 最低 IoU 98.75%，机械臂最低 99.76%；物体平移最大变化 0.29 mm，最终 50 步动作的位置最大变化 6.8 mm，grasp 最大变化 0.00212。这些数值是相对原实现的差异，未度量真机任务成功率。独立服务进程与官方客户端的六次 Zenoh 回环（真实训练录制双目，关闭文件诊断）首请求 631 ms、稳定平均 464 ms、稳定 P95 471 ms；网络部署的结果还取决于链路和诊断设置。完整诊断不适合作为延迟基准，日常可设 `diagnostic_mode: canvas` 或 `diagnostic_dir: null`。

日志包含初始化、当前请求的 `localized_objects`、关键点数、mask 像素、阶段耗时和 done。`gripper_projection` 输出夹爪绘点数、深度裁剪数、画外点数及投影范围；无可见点时为 WARNING。`deployment.diagnostic_dir: null` 默认不逐帧写图；显式设置目录后才保存模型画布及同名 JSON，包含接收的 EE 位姿、模型 hand 位姿、画布内参和绘制统计。

## 保留行为和边界

训练 grasp target 与原推理 sigmoid 的关系仍待真机开合样本核验。本轮验证样本的连续输出约在 0.50–0.73，服务端没有通过修改算法或阈值处理这个现象。done 仅记录日志。几何/软件回归不代表 BRX 标定精度或任务成功率。

仿真继续共享 `inference.policy.ICTPolicy` 和兼容的 `ObjectState`。旧相机辅助代码保存在 `simulation/humanego_sim/legacy_inference/`，新服务端不依赖它们。
