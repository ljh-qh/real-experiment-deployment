# Robot State

本文按 FrankyFrankaGym 的方式整理 robot state：SDK 能读到的状态先映射成字段化的数据结构，训练或控制侧的 `state_obs` 再按配置选择字段并拼接。

信息来源：

- Tianji SDK 本地源码：`/home/ubuntu/rl/PDM_test/PDMode_Test/SDK_PYTHON/fx_robot.py`
- Tianji 示例：`/home/ubuntu/rl/PDM_test/PDMode_Test/TEST_SCRIPT/`
- SharpaWave SDK 头文件：`/home/ubuntu/chensixiang/Codes/sharpa-manus-sdk/retargeting_alg_release_V4.0/include/SharpaWaveSDK_4.6.6/include/SharpaWaveSDK.h`
- SharpaWave Python samples：`/home/ubuntu/chensixiang/Codes/sharpa-manus-sdk/retargeting_alg_release_V4.0/include/SharpaWaveSDK_4.6.6/sample/python/`

## 使用方式

状态分两层：

- SDK state：尽量一一对应 SDK 能读到的字段，保留诊断、计数、错误码、配置回读等信息。
- Observation state：只挑选适合策略或控制使用的连续向量，例如 `joint_pos`、`joint_vel`、`joint_torque`。

当前代码和默认 recording 只把真正高频实时信息作为 robot state：Tianji 的实时包字段、Sharpa 的 joint packet 字段，以及独立的 tactile frame 流。设备身份、连接配置、工具配置、runtime 诊断和 tactile summary 不作为 recording robot state。

`state_obs` 使用 `side_part_state` token 配置最终拼接顺序，例如：

```yaml
states:
  - left_arm_joint_pos
  - left_arm_joint_vel
  - left_end_effector_joint_pos
  - left_end_effector_joint_vel
```

可用于 token 末尾的通用状态字段：

| 字段 | Tianji | Sharpa | 含义 |
|---|---|---|---|
| `joint_pos` | `fb_joint_pos` | `State.angles` | 关节位置，rad |
| `joint_vel` | `fb_joint_vel` | `State.velocities` | 关节速度 |
| `joint_torque` | `fb_joint_sToq` | `State.torques` | 关节扭矩/力矩 |

Tianji 额外可作为 observation 的向量字段：

| 字段 | SDK 来源 | 含义 |
|---|---|---|
| `joint_pos_external` | `fb_joint_posE` | 外编关节位置 |
| `joint_pos_commanded` | `fb_joint_cmd` | 位置关节指令反馈 |
| `joint_current` | `fb_joint_cToq` | 关节电流，SDK 采集接口注释标注为千分比 |
| `joint_temperature` | `fb_joint_them` | 关节温度 |
| `est_joint_friction` | `est_joint_firc` | 关节摩擦估计 |
| `est_joint_friction_derivative` | `est_joint_firc_dot` | 关节摩擦/扰动估计微分 |
| `est_joint_force` | `est_joint_force` | 关节力扰动估计值 |

## TianjiArmState

Tianji 实时状态主要来自：

```text
Marvin_Robot.subscribe(dcss) -> structure2dict(dcss)
```

`structure2dict()` 返回三组数据：`states`、`outputs`、`inputs`。每组都有 A/B 两臂数据，按 arm id 选择。

### 控制器状态

来源：`StateCtr`

| 字段 | SDK 字段 | 含义 |
|---|---|---|
| `cur_state` | `cur_state` | 当前机械臂状态 |
| `cmd_state` | `cmd_state` | 指令状态 |
| `err_code` | `err_code` | 机械臂错误码 |

常见 arm state：

| 值 | 含义 |
|---:|---|
| `0` | 下伺服 / idle |
| `1` | 位置跟随 |
| `2` | PVT |
| `3` | 扭矩 |
| `4` | 协作释放 |

### 实时输出

来源：`RT_OUT`

| 字段 | SDK 字段 | 维度 | 含义 |
|---|---|---:|---|
| `frame_serial` | `frame_serial` | scalar | 输出帧序号 |
| `traj_state` | `traj_state` | scalar | 规划状态，0 no traj，1 receiving，2 received，>=3 running |
| `tip_di` | `tip_di` | scalar | 末端数字输入 |
| `low_speed_flag` | `low_speed_flag` | scalar | 低速标志 |
| `joint_pos` | `fb_joint_pos` | 7 | 关节位置反馈，SDK 为 degree，结构内使用 rad |
| `joint_vel` | `fb_joint_vel` | 7 | 关节速度反馈 |
| `joint_pos_external` | `fb_joint_posE` | 7 | 外编关节位置 |
| `joint_pos_commanded` | `fb_joint_cmd` | 7 | 位置关节指令反馈 |
| `joint_current` | `fb_joint_cToq` | 7 | 关节电流 |
| `joint_torque` | `fb_joint_sToq` | 7 | 关节传感器扭矩，Nm |
| `joint_temperature` | `fb_joint_them` | 7 | 关节温度 |
| `est_joint_friction` | `est_joint_firc` | 7 | 关节摩擦估计 |
| `est_joint_friction_derivative` | `est_joint_firc_dot` | 7 | 关节摩擦/扰动估计微分 |
| `est_joint_force` | `est_joint_force` | 7 | 关节力扰动估计值 |
| `est_cart_force` | `est_cart_fn` | 6 | 末端笛卡尔空间力扰动估计值 |

### 实时输入

来源：`RT_IN`

`tool_kine` / `tool_dyn` 虽然能从 `RT_IN` 读到，但它们是工具运动学/动力学配置，不属于当前 `TianjiArmState`，默认 recording 不采集。

| 字段 | SDK 字段 | 维度 | 含义 |
|---|---|---:|---|
| `rt_in_switch` | `rt_in_switch` | scalar | 实时输入开关 |
| `imp_type` | `imp_type` | scalar | 阻抗类型 |
| `in_frame_serial` | `in_frame_serial` | scalar | 输入帧序号 |
| `frame_miss_cnt` | `frame_miss_cnt` | scalar | 丢帧计数 |
| `max_frame_miss_cnt` | `max_frame_miss_cnt` | scalar | 最大丢帧计数 |
| `sys_cyc` | `sys_cyc` | scalar | 系统周期计数 |
| `sys_cyc_miss_cnt` | `sys_cyc_miss_cnt` | scalar | 实时性 miss 计数 |
| `max_sys_cyc_miss_cnt` | `max_sys_cyc_miss_cnt` | scalar | 最大实时性 miss 计数 |
| `joint_cmd_pos` | `joint_cmd_pos` | 7 | 关节位置指令，SDK 为 degree，结构内使用 rad |
| `joint_vel_ratio` | `joint_vel_ratio` | scalar | 关节速度限制百分比 |
| `joint_acc_ratio` | `joint_acc_ratio` | scalar | 关节加速度限制百分比 |
| `joint_k` | `joint_k` | 7 | 关节阻抗刚度 K 指令 |
| `joint_d` | `joint_d` | 7 | 关节阻抗阻尼 D 指令 |
| `drag_sp_type` | `drag_sp_type` | scalar | 拖动/零空间类型 |
| `drag_sp_para` | `drag_sp_para` | 6 | 拖动/零空间参数 |
| `cart_kd_type` | `cart_kd_type` | scalar | 坐标阻抗类型 |
| `cart_k` | `cart_k` | 6 | 坐标阻抗刚度 K |
| `cart_d` | `cart_d` | 6 | 坐标阻抗阻尼 D |
| `cart_kn` | `cart_kn` | scalar | 零空间刚度 |
| `cart_dn` | `cart_dn` | scalar | 零空间阻尼 |
| `force_fb_type` | `force_fb_type` | scalar | 力控反馈源类型 |
| `force_type` | `force_type` | scalar | 力控类型 |
| `force_dir` | `force_dir` | 6 | 力控方向 |
| `force_pidul` | `force_pidul` | 7 | 力控参数 |
| `force_adj_lmt` | `force_adj_lmt` | scalar | 力控允许调节范围 |
| `force_cmd` | `force_cmd` | scalar | 力控指令 |
| `set_tags` | `set_tags` | 16 | SET 标记 |
| `update_tags` | `update_tags` | 16 | UPDATE 标记 |
| `pvt_id` | `pvt_id` | scalar | 设置的 PVT id |
| `pvt_id_update` | `pvt_id_update` | scalar | PVT id 更新状态 |
| `pvt_run_id` | `pvt_run_id` | scalar | 当前运行 PVT id |
| `pvt_run_state` | `pvt_run_state` | scalar | PVT 运行状态，0 idle，1 loading，2 running，3 error |

### 其他可读接口

| 接口 | 含义 |
|---|---|
| `collect_data()` | 高速采集接口，可采集关节位置、速度、外编位置、指令位置、电流、传感器扭矩、摩擦估计、关节外力估计、末端外力估计 |
| `get_servo_error_code(arm)` | 7 个关节的伺服错误码和描述 |
| `SDK_version()` | SDK 版本 |
| `get_param(type, paraName)` | 控制柜参数读取 |
| `get_tool_info()` | 工具动力学/运动学配置 |
| `get_485_data(arm, com)` | 指定来源的 485 数据 |

## SharpaHandState

当前 `SharpaHandState` 只对应 joint packet。DeviceInfo、运行诊断和 tactile summary 可通过 SDK 查询，但不属于 recording robot state；稠密触觉由独立 tactile stream 记录。

### Joint State

来源：`SharpaWave.get_states()` 返回的 `State`

| 字段 | SDK 字段 | 维度 | 含义 |
|---|---|---:|---|
| `sdk_timestamp` | `State.timestamp` | scalar | joint packet 时间戳 |
| `sequence` | `State.sequence` | scalar | joint packet 序号 |
| `joint_pos` | `State.angles` | 22 | 关节角，rad |
| `joint_vel` | `State.velocities` | 22 | 关节速度 |
| `joint_torque` | `State.torques` | 22 | 关节扭矩/力矩 |

便捷接口：

| 接口 | 含义 |
|---|---|
| `get_joint_position_rad()` | 从 `get_states()` 派生的 rad 关节位置 |
| `get_joint_position_degree()` | 从 `get_states()` 派生的 degree 关节位置 |
| `get_interpolate_mode()` | 插值模式 |

### DeviceInfo（不属于 robot state）

来源：`SharpaWave.get_device_info()` 返回的 `DeviceInfo`

这部分是设备身份和连接配置，不进入当前 robot state 抽象，也不在默认 recording 中采集。

| 字段 | SDK 字段 | 含义 |
|---|---|---|
| `manufacturer` | `manufacturer` | 厂商 |
| `pn` | `pn` | 产品号 |
| `sn` | `sn` | 序列号 |
| `firmware_version` | `firmware_version` | 固件版本 |
| `paired_sn` | `paired_sn` | 配对设备 SN |
| `control_mode` | `control_mode` | 控制模式 |
| `control_source` | `control_source` | 控制源 |
| `hand_side` | `hand_side` | 左手/右手 |
| `device_type` | `device_type` | 设备类型 |
| `mac` | `mac` | MAC 地址 |
| `ip` | `ip` | 设备 IP |
| `tcp_port` | `tcp_port` | TCP 端口 |
| `udp_port` | `udp_port` | UDP 端口 |
| `joint_data_config` | `joint_data_config` | joint 数据传输配置 |
| `tactile_data_config` | `tactile_data_config` | tactile 数据传输配置 |
| `debug_data_config` | `debug_data_config` | debug 数据传输配置 |
| `heart_data_config` | `heart_data_config` | heartbeat 数据传输配置 |
| `status` | `status` | heartbeat 状态 |

`DataTransferConfig`：

| 字段 | 含义 |
|---|---|
| `target_ip` | 目标 IP |
| `target_port` | 目标端口 |
| `transfer_mode` | 传输模式 |

`DeviceStatus`：

| 字段 | 含义 |
|---|---|
| `temperature` | 设备温度 |
| `battery` | 电量 |
| `error_code` | 错误码 |
| `error_joint` | 关联错误的关节 |
| `joint_lock_status` | 关节锁定状态 bitset |
| `temperature_levels` | 温度等级 bitset |

### Runtime Diagnostics（不属于 robot state）

来源：`SharpaWave.get_*()`、`is_*()` 诊断接口。

这部分是运行诊断接口，不进入当前 robot state 抽象，也不在默认 recording 中采集。

| 字段 | SDK 接口 | 含义 |
|---|---|---|
| `enable_state` | `get_enable_state()` | Running/Setting enable 状态 |
| `control_mode` | `get_control_mode()` | 当前控制模式 |
| `control_source` | `get_control_source()` | 当前控制源 |
| `interpolate_mode` | `get_interpolate_mode()` | 插值模式 |
| `current_coeff` | `get_current_coeff()` | 当前电流系数 |
| `speed_coeff` | `get_speed_coeff()` | 当前速度系数 |
| `fault_code` | `get_fault_code()` | fault code 与关联关节列表 |
| `sw_fault_code` | `get_sw_fault_code()` | SDK/软件层 fault code |
| `hand_ready` | `is_hand_ready()` | 手部控制是否 ready |
| `tactile_ready` | `is_tactile_ready()` | 触觉是否 ready |
| `state_counter` | `get_state_counter()` | state packet 计数 |
| `debug_counter` | `get_debug_counter()` | debug packet 计数 |
| `collision_protection_enabled` | `is_collision_protection_enabled()` | 碰撞保护状态 |
| `zero_calibration_progress` | `get_zero_calibration_progress()` | 零点标定进度 |

额外可读原始数据：

| 接口 | 含义 |
|---|---|
| `get_state_packet_raw()` | 原始 state UDP packet |
| `get_debug_packet_raw()` | 原始 debug UDP packet |
| `get_joint_packet_json()` | joint packet JSON |
| `get_debug_packet_json()` | debug packet JSON |
| `get_ptc_error_history()` | PTC 错误历史 |
| `get_parameter([...])` | 通用参数读取 |

### Tactile Interfaces（独立 tactile stream）

来源：`SharpaWave` tactile 接口。

`get_tactile_enable_state()` 和 `tactile_summary()` 是触觉诊断摘要，不作为 recording robot state。稠密触觉图像和触觉低维量由独立 tactile stream 记录。

| 字段/接口 | 含义 |
|---|---|
| `get_tactile_enable_state()` | tactile enable 状态 |
| `tactile_summary()` | packet/frame 统计摘要 |
| `fetch_tactile_frame(channel, timeout)` | 读取一个 tactile frame；当前 Python binding 实测可用 |
| `fetch_tactile_frame_compact(channel, timeout)` | 头文件声明的 compact tactile frame 接口；当前 Python binding 未暴露 |
| `set_tactile_callback()` | 注册 tactile frame callback |
| `set_tactile_callback_compact()` | 注册 compact tactile callback |
| `calib_tactile()` | tactile tare/calibration |
| `deform_map_uv()` | deform 图像坐标映射到 3D 接触点 |
| `deform_map_value()` | deform uint8 值映射为物理值 |

### Tactile Frame

`fetch_tactile_frame(channel, timeout)` 在当前 Python SDK 中返回 dict，当前能拿到的字段是：

| 字段 | 类型 | 含义 |
|---|---|---|
| `frame_id` | `int` | 触觉帧序号 |
| `channel` | `int` | 触觉通道 |
| `ts` | `float` | SDK 时间戳，单位秒 |
| `shape["RAW"]` / `content["RAW"]` | `Shape + uint8[]` | 原始触觉图像，当前实测为 `240x320` |
| `shape["DEFORM"]` / `content["DEFORM"]` | `Shape + uint8[]` | 形变图像，当前实测为 `240x240` |
| `shape["F6"]` / `content["F6"]` | `Shape + float[]` | 6 轴触觉/力信息 |
| `shape["CONTACT_POINT"]` / `content["CONTACT_POINT"]` | `Shape + float[] | None` | 接触点信息，通常按点展开 |

通道约定：

| 手 | 通道 |
|---|---|
| 右手 | `0-4` |
| 左手 | `5-9` |

手指顺序都是 `thumb / index / middle / ring / pinky`。
