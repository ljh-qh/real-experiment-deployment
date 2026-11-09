# tianji-sharpa-infra → unirobot_infra 模块吸收清单

日期：2026-10-26。分支：`dev/tianji`。状态：已实施完成（8 个切片全部落地，全量验证通过）。

来源：`tianji-sharpa-infra` main 的 `MuJoCo-Marvin/` 包（当前在用的控制/遥操/采集主链路，
约 4300 行源码 + 59 个测试 + 78 MB 资产）。主导：本仓库 main 分支的模块结构与代码风格。

## 总原则

- 模块吸收：功能按本包既有模块边界落位，`mujoco_marvin` 包解散，不留独立子包。
- 旧 ROS2 内容（`tianji-sharpa-infra/src/` 全部）不纳入。
- 第三方厂商依赖（Tianji SDK、PICO SDK/XRoboToolkit、Sharpa SDK、pyrealsense2）
  一律按路径/环境变量运行时加载，不拷贝第三方源码；`hand_input.py`（ROS2 桥）不纳入。
- 源测试的 fake tracker / fake SDK 思路原样沿用，不依赖硬件。
- 代码风格遵循 `AGENTS.md`（不藏错、无 fallback、`.`/`[]` 直接读配置）。

## 模块映射表

| 来源（mujoco_marvin/） | 落位（unirobot_infra/） | 说明 |
| --- | --- | --- |
| `ik.py` | `src/unirobot_infra/kinematics/differential.py` | wrist 优先差分 IK；`kinematics/` 在 main 上为空目录，属填空。包 `__init__.py` 注明两种 IK 方式（SDK 侧在 `teleop/mapping/tianji_kinematics.py`，不改） |
| `robot_description.py` | `src/unirobot_infra/robots/arm/marvin_model.py` | 模型/配置路径解析、home 关节、手安装、工具动力学；服务 IK、真机后端与遥操 |
| `tracking.py` | 拆两部分：`src/unirobot_infra/devices/pico_tracker.py`（tracker 帧读取、TrackerActivity、PoseFreezeGuard、ForearmRedundancy）+ `src/unirobot_infra/calibration/tracker_alignment.py`（fit/save/load 方向标定矩阵） | |
| `calibration.py` | `src/unirobot_infra/calibration/tracker_alignment_gui.py` | tkinter 交互式方向标定，核心逻辑在 tracker_alignment.py |
| `tianji_backend.py` | `src/unirobot_infra/robots/arm/tianji_marvin_backend.py` | 按文件路径 importlib 加载厂商 SDK，7 轴关节流后端 |
| `teleop.py`（MarvinTeleop、build_model、dispatch_command） | `src/unirobot_infra/teleop/marvin_teleop.py` + 入口 `src/unirobot_infra/apps/teleop_pico.py`（hydra 风格，对齐 apps/teleop.py） | Object 模式主链路（URDF→MJCF、IK loop、仿真/真机/both） |
| `body_input.py` | `src/unirobot_infra/devices/pico_body.py` | PICO Full Body JSON 回调 + replay/demo 源 |
| `input_state.py` | `src/unirobot_infra/devices/pico_body_state.py` | Body 输入准入/新鲜度状态机 |
| `body_retarget.py` | `src/unirobot_infra/teleop/mapping/pico_body_retarget.py` | 全身重定向 |
| `body_view.py` | `src/unirobot_infra/visualization/body_skeleton.py` | MuJoCo 骨架绘制 |
| `body_teleop.py` | 入口 `src/unirobot_infra/apps/teleop_pico_body.py`；编排逻辑随 `teleop/marvin_teleop.py` 复用 | Full Body 模式 |
| `episode.py` | `src/unirobot_infra/recorder/pico_episode.py` | episode 录制统一进 recorder/；HDF5/manifest 约定与 `recorder/writers.py` 对齐 |
| `diagnostics.py` | `src/unirobot_infra/recorder/teleop_trace.py` | tracker/IK 指令 JSONL 复现记录 |
| `wrist_camera.py` | 拆两部分：采集/录制 → `src/unirobot_infra/camera/realsense_d405.py`（接 camera/ 既有管理器约定）；Tk 实时视图 → `src/unirobot_infra/visualization/realsense_view.py` | |
| `plot_recording.py` | `src/unirobot_infra/visualization/plot_recording.py` | 离线 JSONL 绘图 |
| `home_pose_study.py` | `scripts/home_pose_study.py` | 离线 home 姿态研究，对齐 scripts/ 风格 |
| `cli.py` | 消解：各子命令对应到 apps/ 下 hydra 入口 | 不保留统一 CLI 包 |
| `hand_input.py` | 不纳入 | ROS2 桥（rclpy），超出范围 |
| `scripts/pico-teleop-local.sh` | `scripts/pico_teleop_local.sh` | 参数化原外层仓库布局依赖（venv、PICO SDK 路径） |
| `scripts/{marvin_brakes,marvin_smoke,probe_pico_drift}.py` | `scripts/` 同名 | 参数化 SDK 路径解析 |
| `assets/marvin_dual_arm/`（28 MB） | `assets/robot/marvin_dual_arm/` | 双臂 URDF + STL |
| `assets/wuji-description/`（50 MB） | `assets/robot/wuji_description/` | 厂商受限资产，团队内部使用，沿用 assets/README 约定 |
| `config/tianji_robot.yaml` 等 | `configs/robot/marvin_robot.yaml`、`configs/device/pico_input.yaml.template` 等 | 保留原有 yaml 加载方式（不强转 hydra） |
| `tests/`（7 文件 59 测试） | `tests/` 下对应 `test_marvin_*.py`、`test_pico_*.py` | 移植，fake 思路沿用 |

## 录制与显示统一方式

- 录制统一：Object/Full Body 遥操的 episode 录制与 teleop trace 全部落位 `recorder/`，
  HDF5 写入复用 `recorder/writers.py` 的约定，状态类型复用 `core/types.py`
  （`TianjiArmState`、`SharpaHandState`、`SharpaTactileFrame` 已存在）。
- 显示统一：Tk 相机视图、MuJoCo 骨架、离线 plot、遥操诊断文本集中进 `visualization/`；
  遥操运行态监视复用 `utils/status_gui.py` 的 StatusGui（apps 入口接入）。

## 依赖处理

- `pyproject.toml` 新增：`mujoco==3.3.7`（核心依赖）；`pyrealsense2/imageio/imageio-ffmpeg/Pillow/h5py/pyzmq`
  归入 episode extra；`matplotlib` 归入 plot extra。厂商 SDK 一律运行时按路径加载。
- 环境变量约定沿用来源：`MARVIN_SDK_DIR`、`PICO_SDK_DIR`、`TIANJI_ROBOT_IP`、
  `SHARPA_SDK_PY`、`SHARPA_PILOT_BIN` 等。

## 实施切片（依赖顺序）

1. 基础设施：`kinematics/differential.py`、`robots/arm/marvin_model.py`、资产与配置拷贝、pyproject 依赖
2. PICO 输入与方向标定：`devices/pico_tracker.py`、`calibration/tracker_alignment*.py`
3. Tianji 真机后端：`robots/arm/tianji_marvin_backend.py`
4. Object 遥操主链：`teleop/marvin_teleop.py`、`apps/teleop_pico.py`
5. Full Body 链：`devices/pico_body*.py`、`teleop/mapping/pico_body_retarget.py`、`visualization/body_skeleton.py`、`apps/teleop_pico_body.py`
6. 录制与相机统一：`recorder/pico_episode.py`、`recorder/teleop_trace.py`、`camera/realsense_d405.py`、`visualization/realsense_view.py`
7. 显示与工具：`visualization/plot_recording.py`、`scripts/` 工具
8. 收尾：测试全量验证、py_compile、本清单定稿、commit

## 实施结果（2026-10-26）

- 8 个切片全部落地，源包功能全部按上表落位；迁移以逐字节保真为准则，
  改编仅限 import 路径、路径常量、懒加载目标与 ROS2 排除点。
- 全量验证：`pytest tests/ -q` → **456 passed, 1 skipped, 2 failed**；
  2 个失败（`test_policy.py::test_online_and_offline_policy_images_have_the_same_contract`、
  `test_replay_lerobot.py::test_dataset_config_rejects_local_path_repo_id`）已在 main 基点
  `e89b85a` 复现，属 main 既有问题，与本次合并无关。唯一 skip 为
  `requires_tianji_sdk`（厂商 SDK 不在本机）。35 个改动 .py 全部 `py_compile` 通过。
- 唯一 skip 之外的厂商依赖：Tianji SDK（`MARVIN_SDK_DIR`）、PICO SDK（`PICO_SDK_DIR`）、
  Sharpa SDK（`SHARPA_SDK_PY`）均运行时按路径加载，本机不可用不影响测试。
- 入口：`uv sync` 后可用 console script `unirobot-infra-teleop-pico`
  （等价原 `mujoco-marvin` CLI：pico/body/preview/sim/calibrate/arms/both）。
- 开发环境注意：录制/显示测试依赖 `episode`、`plot` 两个 dependency group，
  `uv sync` 默认会裁剪它们，需 `uv sync --group episode --group plot`。

## 录制范式对齐说明（有意保留的差异）

吸收后的 episode 录制（`recorder/pico_episode.py`）与 main 现有
`recorder/data_recorder.py` 同属 recorder/ 并共享 `core/types.py` 状态类型，
HDF5 低层约定一致（`robot_state/<stream>` 组、`sys_time_ns` int64、`field_order`
attr），但上层范式有意不强行合并：

- 源范式：每 episode 一个目录 + `manifest.json`，gzip 压缩、逐条 resize、
  触觉写 `robot_state/sharpa_tactile/<key>`；
- main 范式：单一多模态 HDF5，预分配块、close 截断，触觉写 `tactile/<side>/<finger>`。

两者对应不同采集链路（PICO 手动触发 episode vs 环境 step 录制），schema 并集
需真机数据验证后再定，届时在 recorder/ 内统一。

## 未纳入项与原因

| 项 | 原因 |
| --- | --- |
| `tianji-sharpa-infra/src/` 全部 ROS2 包 | 用户明确排除；主链路不需要 ROS2 |
| `hand_input.py`（ROS2 手指桥） | 依赖 rclpy，属 ROS2 内容 |
| `XRoboToolkit-Teleop-Sample-Python`、`wujihandros2`、`wuji-retargeting` 子模块 | 第三方上游源码，运行时外部依赖，不拷入 |
| 旧 `tools/marvin_sim/` | 旧平行拷贝，功能已被 MuJoCo-Marvin 取代 |
