# unirobot_infra

`unirobot_infra` 是一个通用机器人硬件工具链，围绕真实机器人接入、环境封装、动作执行、控制器封装、遥操作、数据录制和数据后处理组织。

当前已落地支持 Tianji 机械臂和 Sharpa 灵巧手，主要能力包括：

- 基于 env、wrapper、controller 的机器人控制链路封装
- 机器人 action layout、observation layout 和动作执行
- Tianji 单臂、双臂，以及 Tianji + Sharpa 组合控制
- Manus 手套、Vive Ultimate Tracker、Keyboard、Joint GUI 遥操作输入
- 多相机、Sharpa 触觉、机器人状态与 action 录制
- 感知定位、相机标定和 Tianji + Sharpa 自适应抓握 demo
- raw episode 到 LeRobot dataset 的转换、检查和可视化
- LeRobot episode action 到真实机器人的固定频率 replay

## 开发者入口

核心控制流：

```text
apps -> wrappers -> envs -> controllers -> robots
```

主要模块：

- `apps/`：CLI 入口，例如 teleop、teleop_record、realtime_capture、postprocess
- `wrappers/`：环境包装器，例如 teleop intervention 和 state observation
- `envs/`：统一 `reset/step/close` 控制环境和 action/observation layout
- `controllers/`：把 `RobotAction` 分发为机械臂、灵巧手命令
- `robots/`：Tianji、Sharpa 真实硬件适配
- `devices/`：Manus、Vive、Keyboard、Joint GUI 输入设备
- `recorder/`、`dataset/`：录制写盘和数据集转换

详细开发说明见 [docs/unirobot_infra_developer_guide.md](docs/unirobot_infra_developer_guide.md)。机器人状态字段见 [docs/robot_states.md](docs/robot_states.md)。

## 环境

项目 Python 版本约束为 `>=3.12,<3.13`。

```bash
uv sync --python /home/ubuntu/.local/bin/python3.12
source ./.venv/bin/activate
```

Python 依赖写在 [pyproject.toml](pyproject.toml)，锁定在 [uv.lock](uv.lock)。

## 快速开始

默认遥操作：

```bash
python -m unirobot_infra.apps.teleop
```

默认录制：

```bash
python -m unirobot_infra.apps.teleop_record
```

更多常用命令见 [docs/quick_start.md](docs/quick_start.md)。

## 常用场景

| 场景 | 入口 | 说明 |
| --- | --- | --- |
| 遥操作 | `python -m unirobot_infra.apps.teleop` | 默认双臂 Tianji + Sharpa，Manus + Vive 输入 |
| 录制遥操作数据 | `python -m unirobot_infra.apps.teleop_record` | 录制 robot state、action、camera、tactile |
| 实时相机预览 | `python -m unirobot_infra.apps.realtime_capture` | 检查相机配置和画面 |
| 读取机器人状态 | `python -m unirobot_infra.apps.robot_ops.echo_robot_state` | 检查 Tianji 当前反馈 |
| Vive 信号检查 | `python -m unirobot_infra.apps.vive_probe` | 检查 Vive ZMQ 数据可达性和延迟 |
| 相机标定可视化 | `python -m unirobot_infra.apps.calibration.visualize_camera` | 检查 head camera、SDK base/TCP frame 和 Sharpa 安装姿态 |
| 左臂抓杯 demo | `python -m unirobot_infra.apps.demo.left_grasp_cup` | 感知 cup、移动到 target pose、自适应抓握、向上抬升 |
| 右臂抓瓶 demo | `python -m unirobot_infra.apps.demo.right_grasp_bottle` | 感知 bottle、移动到 target pose、自适应抓握、向上抬升 |
| 双臂抓取 demo | `python -m unirobot_infra.apps.demo.dual_grasp_cup_bottle` | 左抓 cup、右抓 bottle，双侧抬升后松手复位 |
| 数据集转换 | `python -m unirobot_infra.apps.postprocess.convert_lerobot` | raw episode 转 LeRobot dataset |
| 数据集回放 | `python -m unirobot_infra.apps.replay_lerobot` | 按固定频率发送 LeRobot action 到机器人 |

Teleop 的单臂、单手、Keyboard、Joint GUI、Vive tracker override 命令见 [configs/apps/teleop/readme.md](configs/apps/teleop/readme.md)。
感知自适应抓握从标定到运行的完整流程见 [docs/perception_adaptive_grasp_demo.md](docs/perception_adaptive_grasp_demo.md)。

## 配置与硬件

配置集中在 [configs](configs)，通过 Hydra 组合：

- `configs/apps/`：入口运行参数
- `configs/robot/`：机器人组合
- `configs/robot_hardware/`：Tianji、Sharpa 硬件参数和 SDK 路径
- `configs/wrappers/`：observation、teleop source 和 mapping
- `configs/camera/`、`configs/recording/`：采集和录制配置

硬件 SDK 不复制进仓库，运行时通过配置指定路径。硬件配置说明见 [docs/hardware_config.md](docs/hardware_config.md)，Manus 手套标定文件流程见 [docs/manus_glove_calibration.md](docs/manus_glove_calibration.md)，Vive 标定记录见 [docs/vive_calibration_notes.md](docs/vive_calibration_notes.md)。

模型权重统一放在 [models](models)，权重文件由 `.gitignore` 忽略，只提交目录说明。默认感知模型路径见 [models/README.md](models/README.md)。

## 数据录制与后处理

`teleop_record` 录制 raw episode；后处理工具负责重采样、时间戳检查、LeRobot dataset 写入和可视化。

详细流程见 [docs/convert_dataset.md](docs/convert_dataset.md)，采样频率和时间戳说明见 [docs/recording_freq.md](docs/recording_freq.md)。

## 文档索引

- [docs/quick_start.md](docs/quick_start.md)：常用启动命令
- [docs/unirobot_infra_developer_guide.md](docs/unirobot_infra_developer_guide.md)：架构、模块边界和开发约定
- [docs/hardware_config.md](docs/hardware_config.md)：硬件、SDK、网络和相机配置
- [docs/manus_glove_calibration.md](docs/manus_glove_calibration.md)：Manus 手套 `.mcal` 标定文件生成、部署和验证
- [docs/calibration.md](docs/calibration.md)：head camera hand-eye 标定、验证和 Rerun 可视化
- [docs/camera_debug.md](docs/camera_debug.md)：相机占用、设备路径和 USB 拓扑排障
- [docs/perception_adaptive_grasp_demo.md](docs/perception_adaptive_grasp_demo.md)：感知自适应抓握 demo 从标定到运行的流程
- [docs/robot_states.md](docs/robot_states.md)：机器人状态字段说明
- [docs/convert_dataset.md](docs/convert_dataset.md)：raw episode 转 LeRobot dataset
- [docs/recording_freq.md](docs/recording_freq.md)：录制频率和时间戳说明
- [docs/vive_calibration_notes.md](docs/vive_calibration_notes.md)：Vive 映射和标定记录
- [docs/implementation_plan.md](docs/implementation_plan.md)：阶段性实现计划
