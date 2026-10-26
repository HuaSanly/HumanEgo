# HumanEgo 远程推理服务端骨架

当前阶段已建立配置、数据类型和模块边界。原 `policy.py` 保留为唯一策略实现；
双目感知、图像绘制、原生 LeRobot 注册及部署导出仍是显式未实现入口。
这不是可直接启动的 remote 服务，也没有连接相机或机器人。

## 基线与数据契约

- checkpoint：`runs/pick_cube/HumanEgo_anchor/latest.pt`，配套读取同目录
  `config.json`、`dataset_stats.json`，20 步 Euler ODE。
- 单右手、absolute 动作、固定初始 anchor reference；头部、躯干、底盘在 episode 内静止。
- 每眼传输 **640×480** 同步校正图；网络图像为 RGB uint8 `[480,640,3]`。
  模型图像由原 `ICTPolicy.prepare_image()` 生成 RGB float `[1,3,240,320]`。
- 当前右臂状态和单步目标均为 10 维：`[x,y,z,rot6d_0,...,rot6d_5,gripper]`。
  位置单位米，rotation6d 为 `R[:, :2].reshape(-1)`，gripper 为 `0=开、1=闭`。
- 对外位姿为左目校正 optical 系：X 向右、Y 向下、Z 向前；所有位姿使用右手 SE(3)。
- `obj1` 是蓝色容器、`obj2` 是红色方块。ICT 顺序为右手、obj1、obj2、padding，
  token 与 mask 形状分别为 `[1,8,20]`、`[1,8]`，region attention 使用 `[1,2]` anchor UV。
- 动作块目标形状为 float32 `[1,50,10]`，30 Hz 对应约 1.67 秒预测范围。
  时间戳同步及动作新鲜度最终由官方 remote 生命周期和新客户端共同保证。

## 文件职责

| 文件 | 职责与当前状态 |
| --- | --- |
| `policy.py` | **现有实现**：模型加载、图像准备、ICT、anchor UV、ODE 和动作解码 |
| `config.py` | **已实现**：YAML 读取、相对路径解析、标定及权重文件就绪检查 |
| `data_types.py` | **已定义**：双目、右臂状态、物体、场景和感知结果容器 |
| `stereo.py` | **待实现**：双目深度与米制反投影 |
| `perception.py` | **待实现**：物体初始化、追踪/抓取传播、双臂分割及 LaMa |
| `rendering.py` | **待实现**：复现训练侧虚拟夹爪与物体关键点 |
| `configuration_humanego.py` | 声明目标能力，**尚未注册为 LeRobot PreTrainedConfig** |
| `processor_humanego.py` | 已有形状检查与场景 reset，**感知编排待实现** |
| `modeling_humanego.py` | 持有单个 ICTPolicy，**官方 chunk 适配待实现** |
| `export.py` | **待实现**：准备官方可加载的部署产物 |

各模块平铺。配置和数据模块导入不会加载模型；骨架模块不下载权重、不初始化 GPU、
不连接硬件。调用未实现的业务入口会抛出 `NotImplementedError`，不会伪造图像或动作。
`HumanEgoPolicy` 目前不是官方 `PreTrainedPolicy`，不能直接传给 `lerobot-policy-server`。

## 配置

配置统一位于项目的 `cfg/inference/`：

- `params.yaml`：策略、相机尺寸、感知及目标部署参数。
- `intrinsics.yaml`：左右目**实际传输的校正图**的 3×3 内参，尺寸为 640×480。
- `extrinsics.yaml`：`T_left_from_right` 与 `T_EE_from_hand`；平移单位米。

所有文件路径相对于 `params.yaml` 所在目录解析，与启动工作目录无关。
模型维度、手部模式和归一化继续由原 policy 从 checkpoint sidecar 读取。
`deployment` 是规划参数，尚不是官方可直接消费的 server YAML。

```bash
# 从项目根目录读取配置；允许检查含 null 的标定模板，不加载任何模型
python -m inference.config --config_path cfg/inference/params.yaml

# 部署前的配置就绪检查；默认模板会明确报告缺失标定
python -m inference.config --config_path cfg/inference/params.yaml --validate-runtime
```

实际标定尚未提供，因此矩阵保持 `null`。不得用单位矩阵或旧客户端常量替代实测标定。
`validate_runtime_config()` 检查矩阵、单位、非零基线及 checkpoint 三个文件；
检查通过也不代表原生适配、感知或物理标定已经验收。

内参必须匹配缩放后的像素坐标。若从其它尺寸缩到 640×480，后续实现统一使用：

```text
A = [[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1]]
K_new = A @ K_source
P_new = A @ P_source
```

输入已经校正，外参也必须连接对应的校正 optical 系。图像缩放不改变三维位置、
基线或工具变换。骨架没有实现标定缩放或双目重建；640 下的定位精度需要实测。

## 数据流与策略复用

```text
同步原始双目与右臂状态
  → processor 校验
  → perception 初始化/更新场景
  → 分割双臂和夹爪 → LaMa 修复左图
  → rendering 绘制虚拟右手和物体关键点
  → PreparedObservation
  → 原 policy.prepare_image / build_ict / compute_anchor_uv
  → 原 policy.infer
  → 原 policy.decode_ee_in_cam
  → 适配层整理为 [1,50,10]
  → 官方 remote 传输与动作运行时
```

processor 不构建第二套 ICT，也不加载策略。modeling 只转发，不重新构建网络、
归一化或 ODE。最终入口使用官方 `lerobot-policy-server`，不实现自定义网络服务、
动作队列或电机控制器；本轮没有安装、注册或验证该运行时。

`T_EE_from_hand` 将模型 hand 局部坐标映到真实 EE，包含工具平移与旋转：

```text
T_cam_hand = T_cam_EE @ T_EE_from_hand
T_cam_EE_target = T_cam_hand_target @ inverse(T_EE_from_hand)
```

`SceneState.reference_anchor` 必须是初始 obj1 位姿的独立副本。`objects` 保存物体
当前位姿；初始化后移动物体不能修改 reference。编码调用现有
`build_ict(..., reference_anchor=scene.reference_anchor)`；解码也传同一固定 reference。
`anchor_uv` 则投影当前 anchor 实体。clean image 与 ICT 必须使用同一份场景状态。
物体局部关键点后续按当前 pose 重投影，不能一直缓存首次像素位置。

后续感知将从**原始双目**估计物体几何；LaMa 修复像素不参与深度测量。
抓取时锁定手到物体的相对变换，释放后保留最新物体状态再视觉重定位。
`HumanEgoProcessor.reset()` 清空整个场景，保留传入的 perception 实例。

## 与仿真的关系

仿真继续共享 `inference.policy.ICTPolicy` 和 `inference.data_types.ObjectState`。
`ObjectState` 的字段顺序、构造方式和含义保持兼容。
旧 RGB-D 相机及多帧反投影辅助代码和相机配置迁至
`simulation/humanego_sim/legacy_inference/`；新服务端不依赖这些硬件辅助代码。
旧同步实机循环、控制器和 Trossen 驱动已移除。

## 本轮保留的行为与后续验证

原 policy 的函数体、权重加载、固定噪声、Euler ODE、夹爪 sigmoid 和 done 解码保持不变。
以下问题需要后续独立验证，本轮不夹带算法修改：

1. 训练 grasp target 为 0/1，而模板推理又做 sigmoid；先核验该权重的开合样本。
2. 训练绘图和旧单帧夹爪 helper 的局部延伸轴分别为 -Y 和 -Z，需要重绘对照。
3. 640 分辨率的双目精度、PCA 物体系定义、mask 对物体的保护和工具中心重投影。
4. 原生 LeRobot 配置/processor 注册、预热、部署导出、动作块和 reset 回环。

测试骨架与仿真兼容性（使用包含现有仿真依赖的 Python 环境）：

```bash
python -m unittest inference.test_scaffold -v
python -m pytest simulation/tests/test_reference_frames.py simulation/tests/test_perception_geometry.py -q
```

骨架通过测试不表示完成真实感知、remote 接入或机器人任务成功率验证。
