# Camera Calibration

本文说明 Tianji head camera 与左右机械臂的 hand-eye calibration 使用流程。

## 快速流程

1. 打印标定板：A4 横向、`9x6` 黑白格、方格边长 `25 mm`。
2. 分别采集左臂和右臂数据。
3. 分别求解外参。
4. 运行验证，检查误差等级。
5. 用 Rerun 可视化外参、相机视锥和双臂 URDF。

## 标定板

默认棋盘参数是 `8x5` 内角点、`25 mm` 方格，对应打印图案为 `9x6` 个黑白格。

仓库内可直接打印的 PDF：

```text
docs/assets/calibration/calib.io_checker_297x210_6x9_25.pdf
```

也可以用 [Calib.io Pattern Generator](https://calib.io/pages/camera-calibration-pattern-generator) 生成：

| Field | Value |
| --- | --- |
| Target Type | Checkerboard |
| Board Width | `297 mm` |
| Board Height | `210 mm` |
| Rows | `6` |
| Columns | `9` |
| Checker Width | `25 mm` |

打印时关闭页面缩放，使用 100% 原始比例。打印后用尺子确认单个方格边长，并保证 `calibration.checkerboard.square_size_m` 与实物一致。

## 配置速查

| 内容 | 配置 |
| --- | --- |
| 相机设备 | `configs/camera/default.yaml` |
| 采集相机 | `configs/apps/calibration/collect.yaml` 的 `calibration.camera_names` |
| 标定模式 | `configs/apps/calibration/solve.yaml` / `verify.yaml` 的 `calibration.mode` |
| 棋盘参数 | `calibration.checkerboard` |
| 可视化 URDF | `configs/apps/calibration/visualize_camera.yaml` 的 `visualize.robot` |
| 可视化 Sharpa 手 | `configs/apps/calibration/visualize_camera.yaml` 的 `visualize.hands` |
| 可视化标定结果 | `configs/apps/calibration/visualize_camera.yaml` 的 `visualize.calibrations` |
| 可视化初始姿态 | `configs/env/dual_tianji_sharpa_initial.yaml` 的 `env.reset.targets.left/right.arm_q_deg` |

`eye-to-hand` 用于 head camera；`eye-in-hand` 用于腕部相机。采集脚本只记录样本，mode 在求解、验证和可视化阶段设置。

## 采集

左臂：

```bash
python -m unirobot_infra.apps.calibration.collect
```

右臂：

```bash
python -m unirobot_infra.apps.calibration.collect \
  robot=tianji_right \
  calibration.side=right \
  calibration.output_path=/home/ubuntu/chensixiang/Codes/Universal_Robot_Infra/data/calibration/head_camera_tianji_right/poses.json
```

按键：

| Key | Action |
| --- | --- |
| Space | 记录 TCP 位姿和相机图像 |
| Enter | 保存并退出 |

默认使用 Joint GUI 遥操作。切换为键盘遥操作：

```bash
python -m unirobot_infra.apps.calibration.collect \
  wrappers/teleop_intervention=keyboard
```

关闭相机预览：

```bash
python -m unirobot_infra.apps.calibration.collect \
  app.camera_preview.enabled=false
```

采集记录中的 `cartesian_pose` 是 `[x, y, z, qx, qy, qz, qw]`，单位为 m，frame 为：

```text
tianji_<side>_sdk_base -> tianji_<side>_tcp
```

## 求解

左臂：

```bash
python -m unirobot_infra.apps.calibration.solve
```

右臂：

```bash
python -m unirobot_infra.apps.calibration.solve \
  calibration.data_dir=/home/ubuntu/chensixiang/Codes/Universal_Robot_Infra/data/calibration/head_camera_tianji_right
```

输出：

| Mode | Output |
| --- | --- |
| `eye-to-hand` | `T_cam2base.npy`、`T_cam2base.json` |
| `eye-in-hand` | `T_cam2ee.npy`、`T_cam2ee.json` |

## 验证

```bash
python -m unirobot_infra.apps.calibration.verify
```

右臂验证：

```bash
python -m unirobot_infra.apps.calibration.verify \
  calibration.data_dir=/home/ubuntu/chensixiang/Codes/Universal_Robot_Infra/data/calibration/head_camera_tianji_right
```

参考等级：

| 等级 | 平移平均 | 平移最大 | 旋转平均 | 旋转最大 |
| --- | --- | --- | --- | --- |
| 优秀 | `<= 2 mm` | `<= 5 mm` | `<= 0.2 deg` | `<= 0.5 deg` |
| 良好 | `<= 5 mm` | `<= 15 mm` | `<= 0.5 deg` | `<= 1.5 deg` |
| 临界 | `<= 10 mm` | `<= 25 mm` | `<= 1.0 deg` | `<= 3.0 deg` |
| 需重新标定 | 超过临界 | 超过临界 | 超过临界 | 超过临界 |

该误差是 hand-eye 链路内部一致性，不等价于最终抓取点绝对误差。正式抓取前需要做实机落点验证。

## 可视化

```bash
python -m unirobot_infra.apps.calibration.visualize_camera
```

可视化会显示：

- Tianji 双臂 URDF，默认隐藏原 Robotiq 夹爪 mesh。
- 左右 Sharpa Wave 灵巧手 URDF，默认从本机 Sharpa SDK 路径加载。
- 来自 `configs/env/dual_tianji_sharpa_initial.yaml` 的左右臂初始姿态。
- 左右机械臂 base frame 坐标系：`Base_L`、`Base_R`。颜色约定为 X 红、Y 绿、Z 蓝。
- 左右机械臂 SDK TCP 坐标系：`sdk_tcp_L`、`sdk_tcp_R`。
- 左右标定结果对应的相机 frame：`head_camera_left_calib`、`head_camera_right_calib`。
- 相机视锥。

为了减少遮挡，机器人资产默认半透明，并为 `Base_L` / `Base_R` 额外绘制加粗坐标轴。

机械臂末端坐标系由 `visualize.robot.sdk_tcp_frames` 控制，默认显示与 Tianji SDK FK/IK 对齐的 `sdk_tcp_L` / `sdk_tcp_R`。该 frame 由 URDF 的 `ee_L` / `ee_R` 加固定补偿变换得到，不直接使用 URDF link frame。Sharpa 手默认挂载到 `tool_L` / `tool_R`，默认不显示手根 local frame；需要调试手安装姿态时，打开 `visualize.hands.local_frame.enabled`。若 SDK 安装位置或实机安装姿态不同，修改 `visualize.hands.left/right.urdf_path`、`package_paths` 和 `mount.xyz/rpy`。

默认配置会同时加载左右两份 `eye-to-hand` 标定结果：

```yaml
visualize:
  calibrations:
    - side: left
      data_dir: /home/ubuntu/chensixiang/Codes/Universal_Robot_Infra/data/calibration/head_camera_tianji_left
      camera_parent_link: Base_L
      camera_frame_name: head_camera_left_calib
    - side: right
      data_dir: /home/ubuntu/chensixiang/Codes/Universal_Robot_Infra/data/calibration/head_camera_tianji_right
      camera_parent_link: Base_R
      camera_frame_name: head_camera_right_calib
```

如果左右标定一致，Rerun 中 `head_camera_left_calib` 和 `head_camera_right_calib` 应该落在接近相同的位置和姿态。若两者明显分离，应优先检查左右臂采集数据、`T_cam2base.npy` 路径、SDK base 与 URDF `Base_L` / `Base_R` 的对齐关系。

默认没有 `visualize.intrinsics_path` 时，会用 `visualize.camera_frustum` 里的近似分辨率和 FOV 显示视锥。需要精确视锥时，传入相机内参：

```bash
python -m unirobot_infra.apps.calibration.visualize_camera \
  visualize.intrinsics_path=/path/to/intrinsics.json
```

只验证代码路径、不弹出 Rerun GUI：

```bash
python -m unirobot_infra.apps.calibration.visualize_camera \
  visualize.rerun.spawn=false
```

## Frame 说明

| 文件 | 含义 |
| --- | --- |
| `data/calibration/head_camera_tianji_left/T_cam2base.*` | `camera_from_left_sdk_base` |
| `data/calibration/head_camera_tianji_right/T_cam2base.*` | `camera_from_right_sdk_base` |

可视化默认把左标定结果挂到 `Base_L/head_camera_left_calib`，右标定结果挂到 `Base_R/head_camera_right_calib`。这两个相机 frame 表示同一个 head camera 从左右两套标定链路推出来的位置。

可视化默认假设 Tianji SDK base 与 URDF 的 `Base_L` / `Base_R` 对齐。如果 SDK base 与 URDF link 之间还有固定外参，需要单独加入这一层 transform。
