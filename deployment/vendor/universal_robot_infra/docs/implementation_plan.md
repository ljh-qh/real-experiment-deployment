# unirobot_infra 实现计划

本文按最新阶段在最上面的顺序维护当前进度。自动化基线为 `uv run pytest`。

状态含义：

- `[已完成]`：仓库中已有实现和自动化测试覆盖。
- `[部分完成]`：代码链路已有，但阶段目标里的真机验证或数据闭环尚未完成。
- `[未完成]`：仓库中尚无对应生产模块或入口。

## Phase 12: 双臂双手真实 teleop 与稳定性 `[未完成]`

目标是在单组闭环稳定后扩展到 `dual_tianji_sharpa`，完成双臂双手短时真实 teleop、数据记录、后处理和 dry-run replay。

当前状态：

- 默认配置已支持 `dual_tianji_sharpa + manus_vive`。
- 双臂 Vive 和双手 Manus 的 action 合并、side 分发已有自动化测试。
- 双臂双手真实 teleop、真实 episode 记录、LeRobot 导出和 replay dry-run 尚未完成。

## Phase 11: 单组臂手真实 teleop 闭环 `[未完成]`

目标是完成一组 Tianji + Sharpa 的真实遥操作闭环，并打通记录、后处理和 replay dry-run。

当前状态：

- 单侧 `tianji_sharpa_left` / `tianji_sharpa_right` 配置和命令覆盖已存在。
- Manus -> Sharpa、Vive -> Tianji 的软件链路已实现。
- recorder、postprocess、replay 尚未实现，因此完整数据闭环尚未完成。

## Phase 10: 组合级真机人工验证 `[部分完成]`

目标是在已封装硬件 adapter 上，通过 `teleop` 入口验证 `sharpa_left`、`tianji_left`、`tianji_sharpa_left`、`dual_tianji_sharpa` 的组合机器人链路。

当前状态：

- 自动化测试已覆盖组合 robot 构建、action/observation shape、左右侧 action 分发和 wrapper 覆盖。
- 真机组合验证记录没有在仓库文档中固化；真实动作仍需人工确认现场安全、急停、初始姿态和执行权限。

## Phase 9: replay `[部分完成]`

目标是支持从记录数据中重放动作，包含 dry-run、shadow、live 三种模式。

当前状态：

- 已新增 `apps/replay_lerobot.py` 和 `configs/apps/replay/default.yaml`，支持从 LeRobot dataset 读取一条 episode 的 action，并按 dataset fps 或 `app.replay.rate_hz` 固定频率向机器人发送非阻塞 command。
- replay 入口不走 `env.step()`，回放阶段直接调用 controller 发送 action，避免每帧读状态反压控制频率。
- dry-run、shadow 模式和真机回放记录仍未完成；live replay 前需要人工确认现场安全、急停、初始姿态和执行权限。

## Phase 8: postprocess 与 LeRobot 导出 `[部分完成]`

目标是读取 raw episode，完成多频率时间戳对齐、降采样和 LeRobot dataset 导出。

当前状态：

- 已新增 `dataset/` 后处理模块，包含 raw episode reader、按多 stream 时间戳最近邻降采样、LeRobot converter、dataset card 和 Canvas resampling timeline HTML。
- 已新增 `apps/postprocess/convert_lerobot.py`、`apps/postprocess/visualize_lerobot.py`、`apps/postprocess/inspect_resampling_timeline.py`、`apps/postprocess/update_lerobot_readme.py` 和 `apps/postprocess/push_lerobot.py`。
- 自动化测试覆盖配置解析、HDF5 reader、resampler、timeline HTML 和 dataset card 描述。
- 真实大规模 LeRobot 转换、Rerun 交互可视化和 Hub push 仍需要在安装 `lerobot` / `rerun-sdk` 的环境中做人工验证。

## Phase 7: recorder 独立进程 `[未完成]`

目标是把 teleop 进程和数据记录进程分离，recorder 订阅 snapshot 并写 HDF5/MP4 raw episode。

当前状态：

- `core.types.Snapshot` 和 `core.messages.snapshot_message()` 已有基础数据结构。
- 仓库中尚无 `runtime/`、`recorder/` 生产模块。
- 尚无 `apps/record.py` 入口。

## Phase 6: teleop CLI 与真实控制确认 `[已完成]`

目标是形成完整单进程遥操作入口，所有输入都走 `TeleopInterventionWrapper -> env.step`，不保留 app 层的单独控制流。

已完成：

- `apps/teleop.py` 通过 Hydra 加载配置，并通过 `wrappers.builder.build_env(cfg)` 组装 env/wrapper 链。
- 默认入口为 `dual_tianji_sharpa + manus_vive`。
- 已支持 `manus`、`vive`、`manus_vive`、`keyboard`、`joint_gui` 五类 source。
- `JointGui` 使用本地 Web GUI，并作为 `source=joint_gui` 通过 wrapper 发送动作。
- `apps/vive_probe.py` 可独立检查 ViveTrackerServer ZMQ 信号。
- `apps/robot_ops/echo_robot_state.py` 和 `apps/robot_ops/move_tianji_tcp_local.py` 可做 Tianji 反馈读取和 TCP local 小位移复查。
- 真机入口接入 `app.sdk_output`，默认将 `[Marvin SDK]` 输出分流到运行日志目录。

主要测试：

- Hydra compose 覆盖单手、单臂、单侧组合、双臂、双臂双手和 GUI 场景。
- teleop loop、状态 GUI、Joint GUI、Vive 初始姿态门禁和 IK 失败策略均有测试覆盖。

## Phase 5: teleop mapping 与 wrappers `[已完成]`

目标是把 Manus/Vive/Keyboard/GUI 输入转换成 env action，并通过 wrapper 覆盖 policy action。

已完成：

- `TeleopInterventionWrapper` 统一读取 teleop source，并用 `RobotActionLayout.overlay_actions()` 覆盖 action segment。
- Manus 通过 `SharpaRetargeter` 生成 Sharpa 22 维手关节 action。
- Vive 通过 deadman trigger 建立 tracker/TCP 锚点，将 tracker local SE(3) 增量映射到 Tianji TCP local 6D，再经 IK 输出 q7。
- Keyboard 使用 `robot_world` 平移约定和 Tianji IK 做 Cartesian jog。
- Composite source 可合并同侧 arm 和 end_effector action，并拒绝重复 part action。

主要测试：

- wrapper 覆盖、inactive 行为、组合 source 合并、Vive local axis 映射、IK 失败策略均有测试覆盖。

## Phase 4: controllers 与 envs `[已完成]`

目标是建立参考 `FrankyFrankaGym` 的 Gym-style 环境与 action 分发。

已完成：

- `TianjiSharpaEnv` 暴露 `reset/step/close/action_space/observation_space`。
- `RobotActionLayout` 定义 flat action 与 `RobotAction` 的转换。
- `StateObservationWrapper` 支持通过 `wrappers.state_obs.states` 配置 observation 字段。
- `ArmJointController`、`HandJointController`、`RobotGroupController` 负责 q7/22 维关节动作分发。
- `envs.builder.build_base_env(cfg)` 按 `cfg.robot` 构建裸 env。

当前默认 action/observation shape；默认 observation 是 `joint_pos + joint_vel`：

- `robot=sharpa_left`：action `(22,)`，observation `(44,)`
- `robot=tianji_left`：action `(7,)`，observation `(14,)`
- `robot=tianji_sharpa_left`：action `(29,)`，observation `(58,)`
- `robot=dual_tianji_sharpa`：action `(58,)`，observation `(116,)`

## Phase 3: devices 层 `[已完成]`

目标是实现遥操作输入设备抽象和可测试数据源。

已完成：

- `ManusDevice` 订阅 Manus mocap ZMQ 数据，可按配置自动启动 `SharpaManusClient.out`。
- `ViveTrackerDevice` 订阅 ViveTrackerServer ZMQ 数据，并支持绑定指定 network interface。
- `KeyboardDevice` 提供按键状态帧。
- `JointGui` 提供本地 Web 关节控制页面。

主要测试：

- Manus protobuf 转换、空侧处理、Vive JSON topic 解析、Keyboard key binding、Joint GUI 状态和动作回调均有测试覆盖。

## Phase 2: robots 层 `[已完成]`

目标是实现被控硬件抽象，以及 Tianji/Sharpa 的可组合机器人结构。

已完成：

- `TianjiRobot` 封装 SDK 连接、反馈读取、position mode、q7 关节命令发送和错误检查。
- `TianjiArm` 负责单侧 arm id 映射和 rad/deg 转换。
- `SharpaHand` 封装 SDK import、SN/IP/左右手过滤、连接、使能、读写 22 维关节。
- `RobotUnit` 和 `RobotGroup` 支持左/右侧组合。

主要测试：

- Tianji arm index、读反馈、rad/deg 发送、send failure、command readback warning、position mode、Sharpa side/IP/SN 匹配逻辑均有测试覆盖。

## Phase 1: core 类型与 Hydra 配置 `[已完成]`

目标是定义共享数据结构、单位约定、时间戳和 Hydra 配置分组。

已完成：

- `core.types` 定义 `Pose`、`JointState`、Tianji/Sharpa 专用 robot state、`DeviceFrame`、`RobotState`、`RobotAction`、`Snapshot`。
- `core.units` 和 `core.time` 提供单位转换与时间戳工具。
- Hydra 配置按 `apps/`、`robot/`、`robot_hardware/`、`device/`、`env/`、`wrappers/` 分组。
- robot 配置用 `sides.<side>.parts` 表达硬件组合。

主要测试：

- 多 robot 组合、常用 override、缺失必要配置报错和配置层级约束均有测试覆盖。

## Phase 0: 项目骨架与环境 `[已完成]`

目标是建立可运行、可测试、可配置的最小项目。

已完成：

- 使用 `uv` + `pyproject.toml` 管理依赖和包信息。
- 建立 `src/unirobot_infra/`、`configs/`、`tests/` 和基础 CLI。
- 日志系统使用 Python `logging` + `rich`。
- pytest 配置落在 `pyproject.toml`。

## 通用规则

- 真实硬件动作前必须人工确认现场安全、急停、初始姿态、限位和执行权限。
- 真机动作测试不能做成无人值守自动化测试。
- `src/` 和 `configs/` 只放生产代码与生产配置；测试替身只放在 `tests/`。
- 配置缺失、SDK 路径错误、source 与 robot side 不匹配、设备数据格式错误应 fail-fast。
- 进入 recorder/postprocess/replay 前，应先确认数据 schema、进程边界和 teleop 控制进程不被记录 IO 反压。
