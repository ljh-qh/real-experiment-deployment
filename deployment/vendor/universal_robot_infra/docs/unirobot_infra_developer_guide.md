# unirobot_infra 开发者文档

本文描述当前仓库已经落地的架构、模块边界和代码公约。

## 1. 目标

unirobot_infra 面向 Tianji 机械臂、Sharpa 灵巧手、Manus 手套、Vive Ultimate Tracker，目标是提供：

- 遥操作
- 数据记录
- 数据后处理降采样
- 数据重放

当前仓库已经完成遥操作主链路；记录、后处理和重放仍是后续阶段。

工程风格参考：

- 核心遥操作逻辑参考 `../sharpa-tj-teleop`。
- `envs`、`wrappers`、记录/后处理组织参考 `../FrankyFrankaGym`。

## 2. 当前模块

当前生产模块：

```text
unirobot_infra/
  apps/          # CLI: teleop, calibration, perception, demo, robot_ops, postprocess
  core/          # 类型、单位、时间戳、消息
  devices/       # 遥操作输入设备: Manus, Vive Ultimate Tracker, Keyboard, Joint GUI
  robots/        # 被控机器人硬件与组合: Tianji arm, Sharpa hand
  controllers/   # action -> robot command
  envs/          # Gym-style 环境: reset/step/close/action_space/observation_space
  wrappers/      # 环境包装器: teleop intervention, state obs
  teleop/        # teleop mapping: Manus retarget, Vive local SE(3), Tianji IK, keyboard jog
  replay/        # LeRobot episode 读取、回放循环和显式诊断工具
  utils/         # logging, SDK paths, SDK output router, status GUI
```

后续规划模块：

```text
runtime/       # 进程、通信、生命周期
recorder/      # 独立记录进程
postprocess/   # 降采样与 LeRobot 导出
replay/        # 数据重放
```

这些规划模块尚未作为生产目录存在；在实现前不要创建空模块或占位入口。

## 3. robots 与 devices 边界

`devices/` 只放遥操作输入设备：

- `ManusDevice`
- `ViveTrackerDevice`
- `KeyboardDevice`
- `JointGui`

`robots/` 只放被控硬件：

- `robots/arm/TianjiArm`
- `robots/end_effector/SharpaHand`
- `robots/compositional/RobotUnit`
- `robots/compositional/RobotGroup`

组合关系：

```text
RobotGroup
  left:  RobotUnit(arm=TianjiArm | None, end_effector=SharpaHand | None)
  right: RobotUnit(arm=TianjiArm | None, end_effector=SharpaHand | None)
```

组合命名规范：

- `sharpa_left`：只控制左手 Sharpa
- `sharpa_right`：只控制右手 Sharpa
- `tianji_left`：只控制左臂 Tianji
- `tianji_right`：只控制右臂 Tianji
- `tianji_dual`：左右 Tianji
- `tianji_sharpa_left`：左侧 Tianji + Sharpa
- `tianji_sharpa_right`：右侧 Tianji + Sharpa
- `dual_tianji_sharpa`：左右两组 Tianji + Sharpa

## 4. envs

`envs/` 是统一控制环境层。它不直接读取 Manus、Vive、Keyboard 或 GUI。

当前实现：

```text
envs/
  builder.py
  layout.py
  spaces.py
  tianji_sharpa_env.py
```

`TianjiSharpaEnv` 职责：

- 接收 `RobotGroupController` 和 `RobotActionLayout`
- 暴露 `action_space` 和 `observation_space`
- `reset()`：连接硬件并读取当前状态
- `step(action)`：执行一帧 robot action，返回 observation
- `close()`：释放硬件

`RobotActionLayout` 负责 flat action 与 `RobotAction` 的转换。action 顺序按配置遍历：

```text
for side in robot.sides:
  for part in robot.sides[side].parts:
    append part joints
```

默认 observation 由 `wrappers.state_obs.states` 的显式 token 列表生成。每个 token 形如 `left_arm_joint_pos`、`right_end_effector_joint_vel`，列表顺序就是最终拼接顺序。

- `robot=sharpa_left`：action `(22,)`，observation `(44,)`
- `robot=tianji_left`：action `(7,)`，observation `(14,)`
- `robot=tianji_sharpa_left`：action `(29,)`，observation `(58,)`
- `robot=dual_tianji_sharpa`：action `(58,)`，observation `(116,)`

`joint_pos` 对应 `q_rad`，`joint_vel` 对应 `dq_rad_s`。如果只需要左手关节位置，可以覆盖 `wrappers.state_obs.states=[left_end_effector_joint_pos]`。

## 5. wrappers

`wrappers/` 围绕 env 叠加功能，而不是把所有逻辑写进 env。

当前实现：

```text
wrappers/
  base.py
  builder.py
  state_obs.py
  teleop_intervention.py
```

`TeleopInterventionWrapper` 是遥操作主扩展点：

- 读取 `TeleopActionSource`
- 通过 `RobotActionLayout.overlay_actions()` 覆盖 policy action 的相关 segment
- 在 `info` 中记录 `policy_action`、`device_frames`、`target_actions`、`sent_actions`、`teleop_active`

`StateObservationWrapper` 负责把 `info["robot_states"]` 转成 flat observation：

- `states` 中每一行是一个 `side_part_state` token
- `states` 配置顺序就是 observation 拼接顺序
- `missing_segments=skip` 时，当前 robot 不存在的 segment 会跳过；`missing_segments=error` 时会报错
- 通用字段支持 `joint_pos`、`joint_vel`、`joint_torque`
- Tianji 额外支持 `joint_pos_external`、`joint_pos_commanded`、`joint_current`、`joint_temperature`、`est_joint_friction`、`est_joint_friction_derivative`、`est_joint_force`

teleop source 名称只表示输入设备或设备组合：

- `manus`
- `vive`
- `manus_vive`
- `keyboard`
- `joint_gui`

不要把目标机器人、左右臂或 mapping 语义混进 source 名称。目标机器人由 `robot` 配置选择，左右侧由 `robot.sides` 或 `mapping.side` 选择，转换方式由 `mapping.type` 表达。

## 6. teleop mapping

当前 mapping：

- `sharpa_retarget`：Manus keypoints -> Sharpa 22 维关节 action。
- `tianji_vive_local_se3_ik`：Vive tracker local SE(3) 增量 -> Tianji TCP local 6D 目标 -> Tianji IK -> q7。
- `tianji_cartesian_jog`：Keyboard command -> robot_world 平移增量 / SDK abc 旋转增量 -> Tianji IK -> q7。
- `direct_joint_action`：Joint GUI 直接生成机器人关节 action。
- `manus_vive`：组合 Manus hand action 和 Vive arm action。

Vive 默认使用 deadman trigger：

```text
trigger 未按下：不发送 Tianji 遥操作动作
trigger 按下：记录当前 tracker pose 和当前 TCP pose 为锚点
trigger 按住：用 tracker local SE(3) 相对变换控制 TCP local 6D
trigger 松开：停止发送并清空锚点
```

## 7. controllers

`controllers/` 定义 action 如何变成 robot command。

当前实现：

- `ArmJointController`：发送 Tianji q7 关节目标。
- `HandJointController`：发送 Sharpa 22 维关节目标。
- `RobotGroupController`：按 side 分发多个 `RobotAction`。

Tianji 主路径是：

```text
Vive/Keyboard input -> TCP target -> Tianji IK -> q7 -> set_joint_cmd_pose + send_cmd
```

如果后续确认 Tianji SDK 有可靠笛卡尔伺服/阻抗接口，应新增 controller，不改 env 和 wrapper 边界。

## 8. Hydra 配置

配置按启动入口组织。`configs/apps/` 下的一个文件或目录对应 `apps/` 下的一个启动脚本。

当前主要配置：

```text
configs/
  apps/
    demo/left_grasp_cup.yaml
    demo/right_grasp_bottle.yaml
    demo/dual_grasp_cup_bottle.yaml
    perception/perception_loop.yaml
    calibration/visualize_camera.yaml
    robot_ops/echo_robot_state/default.yaml
    robot_ops/move_tianji_tcp_local/default.yaml
    robot_ops/move_tianji_tcp_pose/left.yaml
    robot_ops/move_tianji_tcp_pose/right.yaml
    robot_ops/sharpa_anchor_probe/default.yaml
    teleop/default.yaml
    teleop/readme.md
    vive_probe.yaml
  robot/
    sharpa_left.yaml
    sharpa_right.yaml
    tianji_left.yaml
    tianji_right.yaml
    tianji_dual.yaml
    tianji_sharpa_left.yaml
    tianji_sharpa_right.yaml
    dual_tianji_sharpa.yaml
  robot_hardware/
    arm/tianji.yaml
    end_effector/sharpa.yaml
  device/
    vive/default.yaml
  env/
    default.yaml
  wrappers/
    state_obs/default.yaml
    teleop_intervention/*.yaml
```

配置层级约定：

- app 配置只控制入口运行参数，例如 logging、状态窗口、SDK 输出路由。
- robot 硬件参数放 `robot.hardware` 下，由 `robot/` 配置组合。
- 遥操作输入设备配置放 `wrappers.teleop_intervention.device` 下。
- mapping 配置放 `wrappers.teleop_intervention.mapping` 下。
- observation 配置放 `wrappers.state_obs.states` 下。
- `robot/` 配置用 `sides.left/right.parts` 描述左右侧硬件组合，不额外配置重复 unit 名称。

常用入口：

```bash
python -m unirobot_infra.apps.teleop
python -m unirobot_infra.apps.teleop robot=sharpa_left wrappers/teleop_intervention=manus wrappers.teleop_intervention.device.manus.frame_side=left
python -m unirobot_infra.apps.teleop robot=tianji_left wrappers/teleop_intervention=keyboard
python -m unirobot_infra.apps.vive_probe
python -m unirobot_infra.apps.robot_ops.echo_robot_state
```

## 9. SDK 与输出

硬件 SDK 不复制进仓库。

- Sharpa / Manus SDK 根路径由 `robot.hardware.end_effector.sdk.vendor_root` 指定。
- Tianji SDK 根路径由 `robot.hardware.arm.sdk_dir` 指定。
- SDK 目录结构由 `utils/sdk_paths.py` 和 Tianji/Sharpa adapter 按配置推导。

SDK 原生输出可能直接写 `stdout/stderr`。当前真机入口通过 `app.sdk_output` 启用 fd 级输出路由：

- 默认运行目录在 `logs/<timestamp>_<run_name>/`。
- 当前默认规则只把包含 `[Marvin SDK]` 的输出写入 `tianji_marvin.log`。
- 未匹配输出继续留在原终端。
- 配置错误、正则错误、目录不可写必须直接报错。

Manus client 是独立进程，其输出写入 `wrappers.teleop_intervention.device.manus.client.log_path`。

## 10. 状态显示

teleop 入口使用 `utils/status_gui.py` 启动 CustomTkinter 状态窗口，显示 robot、source、mapping、frame、teleop active/idle 和使用说明。

Joint GUI 遥操作使用 `devices/joint_gui.py` 提供本地 Web 页面。它不是单独控制流，必须作为 `source=joint_gui` 通过 `TeleopInterventionWrapper` 和 `env.step` 发送动作。

## 11. 后续进程模型

记录阶段尚未实现。后续设计仍保持 teleop 与 recorder 分进程：

```text
Process A: teleop
  - build env + wrappers
  - own Tianji/Sharpa SDK connections
  - read Manus/Vive/Keyboard/GUI through TeleopInterventionWrapper
  - publish snapshots

Process B: recorder
  - subscribe snapshots
  - capture cameras
  - write episodes
  - never own robot SDK
```

这个设计用于避免 recorder 和 teleop 抢机器人 SDK 连接。实现前必须先明确 snapshot schema、通信方式和 recorder 崩溃时的隔离策略。

## 12. 数据流

当前遥操作数据流：

```text
Manus/Vive/Keyboard/JointGui
  -> TeleopActionSource
  -> teleop mapping / retarget / IK
  -> TeleopInterventionWrapper
  -> TianjiSharpaEnv.step
  -> RobotGroupController
  -> TianjiArm / SharpaHand
```

当前 `env.step()` 和 wrapper `info` 已包含：

- raw-ish device frame
- target action
- sent action
- robot state
- teleop active/idle

后续 recorder 应从确认后的 snapshot 发布接口订阅这些数据。原始高频数值数据保存为 HDF5，图像数据保存为 MP4，后处理训练数据导出为 LeRobot dataset。

## 13. 代码公约

项目环境使用 `uv` 和 `pyproject.toml` 管理：

- 依赖、包信息、CLI 入口写在 `pyproject.toml`。
- 锁文件使用 `uv.lock`。
- 安装依赖使用 `uv sync`。
- 运行脚本时先进入 `Universal_Robot_Infra/` 仓库根目录，再执行 `uv run python -m ...` 或激活 `.venv` 后执行 `python -m ...`。

通用约定：

- 日志系统使用 Python `logging` + `rich`，实现放在 `utils/logging.py`。
- `devices` 只表示输入设备。
- `robots` 只表示被控硬件。
- `envs` 负责统一 step/reset 接口。
- `wrappers` 负责叠加遥操作输入、观测、后续安全和发布能力。
- `core` 不 import 任何厂商 SDK。
- 所有物理量字段必须带单位后缀，例如 `q_deg`、`q_rad`、`xyz_m`、`xyz_mm`、`timestamp_ns`。
- 控制主链路不写 episode 或高频数据；入口级 SDK 输出日志由 `app.sdk_output` 显式配置。
- `src/` 和 `configs/` 只放生产代码与生产配置。
- `apps/demo/` 只放可复现实机 demo 入口；每个 demo 必须有同名 `configs/apps/demo/*.yaml` 和文档说明。
- 测试替身只放在 `tests/`。
- 人工检查脚本可以放在 `scripts/`，但必须避免伪装成生产入口；涉及真机动作时必须显式说明和配置动作幅度、侧别、目标对象。
- 除 `apps/demo/` 约定入口外，生产代码、生产配置和生产入口不得使用测试或临时语义命名，例如 `fake`、`mock`、`smoke`、`test`。
- 不为未来功能预留空模块、空配置、占位入口或临时 app；没有实现的模块不建文件。

配置和错误处理：

- 读取配置直接用点号访问，缺字段应自然报错，不用 `getattr` 或硬编码默认值隐藏问题。
- 读取字典直接用 `[]`，不用 `.get()` 写隐式默认值。
- 尽量少用 `try/except`，不要吞掉错误；需要捕获时必须有明确恢复策略。
- 不在代码中埋 fallback 行为；替代路径必须由配置显式选择。
- 设备缺数据、IK 失败、source 与 robot side 不匹配等运行错误必须 fail-fast；如果确实需要恢复策略，必须由配置显式选择并在日志或状态里暴露。

设计和命名：

- 命名必须贴近真实领域概念；如果字段、类、模块不能清楚回答“代表什么真实对象或行为”，就不要加入。
- 配置层级必须清晰：app 配置只控制一个启动入口；可复用硬件参数放子配置；高层配置可以覆写子配置；禁止重复表达同一概念。
- teleop source 名称只表示输入设备，禁止使用 `manus_sharpa`、`vive_tianji_left` 这类把设备、机器人和侧别混在一起的命名。
- GUI 遥操作也必须作为 `source=joint_gui` 走 `TeleopInterventionWrapper` 和 `env.step`，不得在 app 层维护单独控制流。
- 注释风格跟随上下文，只保留和代码设计、理解有关的内容。
- 代码尽可能简洁，避免无用、臃肿、废弃代码和过度设计。

真机安全：

- 真机相关代码默认保守。
- 真实动作必须由明确配置开启，并且动作幅度、目标关节、侧别、硬件对象都必须从配置或显式参数看得出来。
- 真实硬件动作前必须人工确认现场安全、急停、初始姿态、限位和执行权限。
- 真机动作测试不得做成无人值守自动化测试。

## 14. 当前一致性结论

当前代码与本文档的核心边界一致：

- `envs/` 没有直接读取 Manus/Vive/Keyboard/GUI。
- `robots/` 没有依赖 `devices/`。
- `apps/teleop.py` 是薄入口，实际控制流走 `wrappers.builder.build_env(cfg)`。
- GUI 遥操作走 `source=joint_gui -> TeleopInterventionWrapper -> env.step`。
- SDK 输出路由已在 `teleop`、`echo_robot_state`、`move_tianji_tcp_local`、`replay_lerobot` 等真机入口接入。
- `apps/replay_lerobot.py` 是薄入口，LeRobot episode 读取、回放循环和 TCP 轨迹诊断在 `replay/` 下实现；dry-run 和 shadow replay 仍未实现。
