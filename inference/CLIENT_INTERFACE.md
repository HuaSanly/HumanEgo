# HumanEgo 客户端适配接口

## 连接与会话

- 使用官方 LeRobot remote inference，revision `5517a326eeb9b457cfe07d59b9bfdf291799c8d9`，传输为 Zenoh；服务端监听地址、deployment 和语义标识以部署包 `server.yaml` 为准。默认地址 `tcp/127.0.0.1:7447`，跨主机需配置可访问地址。
- 默认 deployment：`humanego-pick-cube-anchor`；robot type：`humanego_brx`；语义标识：`humanego.right-ee.left-rectified-optical.m.row-major-rotation6d.continuous-grasp.v1`。
- 支持普通 `chunk` 模式，30 Hz；不启用 RTC 或语言。连接后通过官方 discovery/admission 校验 feature、语义、模式和动作间隔，再开始发送观测。
- 每个 episode 开始时 reset。reset 清空服务端场景和固定参考；旧 generation 的请求和结果不能用于新 episode。使用官方 runtime 处理请求、动作块补货和过期结果。

## 观测与动作

| Feature | 格式 | 内容 |
| --- | --- | --- |
| `observation.images.head_left` | `uint8 [480,640,3]` | 左目校正图，RGB，HWC |
| `observation.images.head_right` | `uint8 [480,640,3]` | 右目校正图，RGB，HWC |
| `observation.state` | `float32 [10]` | 右臂实测 EE 位姿和夹爪状态 |
| `action` | `float32 [10]`，每块 50 步 | 右臂绝对 EE 目标位姿和连续 grasp |

左右图与 EE 状态必须来自同一时刻；图像为与服务端标定一致的同步校正双目。使用官方 raw 无损图像编码。客户端保留采集开始时的单调时钟时间戳，由官方 runtime 检查观测和结果时效；不要求客户端与服务端单调时钟同步。

状态与每步动作的分量顺序均为：

```text
[x, y, z, R00, R01, R10, R11, R20, R21, grasp]
```

位置单位为米，旋转使用 `R[:, :2].reshape(-1)` 的行优先排列；两列必须是有限、单位且相互正交的向量，第三列由前两列叉积恢复。状态 `grasp` 为实测归一化夹爪状态，范围 `[0,1]`，`0=开`、`1=闭`。动作 `grasp` 是策略原始连续 sigmoid 输出，服务端不二值化；客户端负责映射到实际夹爪控制。

每次推理返回 `float32 [50,10]` 动作块，每步为绝对目标，按 30 Hz 执行；不是增量。服务端适配层内部形状为 `[1,50,10]`，官方运行时去掉 batch 维后传输。

## 坐标与 EE

输入与输出的 EE 位姿均在**左目校正 optical 坐标系**下：X 向图像右、Y 向图像下、Z 向前。客户端负责根据实测关节和相机位姿计算 `T_left_from_EE`，并将返回的左目坐标系 EE 目标转换为机器人控制坐标系。

BRX 右臂 EE 原点约定为两夹爪接触点的抓取中心。EE 轴向与模型 hand 轴对齐：+X 沿夹爪开合方向，+Y 从夹爪底座指向抓取中心，+Z 按右手系确定。相对 URDF `LinearclampinggripperJZ01_Link` 的候选方向为：EE +X 对应底座 +Y、EE +Y 对应底座 +X、EE +Z 对应底座 −Z。夹爪两侧的方向及具体接触位置尚待实机确认；客户端不得直接将 `ArmR08_Link` 或夹爪底座位姿当作上述 EE 位姿。

客户端发送 EE 位姿，服务端使用 `T_EE_from_hand` 转换为模型 hand 位姿，预测后用其逆变换返回 EE 目标。按 EE 和模型 hand 原点、轴向重合的约定，当前 `T_EE_from_hand` 为单位矩阵。左右目校正内参按客户端 `stereoRectify` 的 `CALIB_ZERO_DISPARITY`、`alpha=0` 计算，双目基线为 0.060039595 m，须与客户端实际校正映射和图像核对。原始左目 optical 位姿转换到校正左目时必须应用 `R1`：`T_left_rect_from_EE = diag(R1, 1) @ T_left_raw_from_EE`，其中 `R1` 记录在 `cfg/inference/stereo_raw_calibration.yaml`。

## episode 约束

头部、躯干、底盘在 episode 内保持静止。默认事件模式下未抓取物体保持静止；外力移动后应 reset 重新初始化。服务端返回无效或过期结果时不得执行该结果；停止、超时及动作耗尽由客户端控制流程处理。`done` 仅供服务端诊断，不作为网络动作字段。
