# Perception Adaptive Grasp Demo

本文记录从相机标定、落点 offset 调整、Sharpa 抓取锚点录制、反馈阈值设置，到最终运行感知自适应抓握 demo 的完整流程。

默认 demo 目标：

```text
感知 cup 位置 -> 加 robot_world offset -> 15 Hz 小步闭环移动到 target TCP pose -> Sharpa adaptive grasp -> robot_world +z 抬升 10 cm
```

## 1. 前置检查

进入仓库根目录并激活环境：

```bash
cd /home/ubuntu/chensixiang/Codes/Universal_Robot_Infra
source ./.venv/bin/activate
```

检查硬件网络、SDK、相机和 Sharpa 触觉配置：

- Tianji / Sharpa / ZED 设备检查见 [hardware_config.md](hardware_config.md)。
- V4L2 相机占用和 USB 路径排障见 [camera_debug.md](camera_debug.md)。
- 标定流程见 [calibration.md](calibration.md)。

模型权重放在本机目录，不提交到 git：

```text
models/perception/yoloe-26l-seg.pt
models/perception/mobileclip2_b.ts
```

当前感知配置默认读取 `models/perception/yoloe-26l-seg.pt`。如果权重不在这个路径，运行前先放回同名文件，或显式覆盖 `demo.detector.model_path`。

## 2. 相机标定

采集左臂 head camera hand-eye 数据：

```bash
python -m unirobot_infra.apps.calibration.collect
```

求解外参：

```bash
python -m unirobot_infra.apps.calibration.solve
```

验证外参：

```bash
python -m unirobot_infra.apps.calibration.verify
```

可视化相机、左右臂 base frame、SDK TCP frame 和 Sharpa 手安装姿态：

```bash
python -m unirobot_infra.apps.calibration.visualize_camera
```

左右 demo 分别使用：

```text
data/calibration/head_camera_tianji_left/T_cam2base.npy
data/calibration/head_camera_tianji_right/T_cam2base.npy
```

对应配置字段：

```text
configs/apps/demo/left_grasp_cup.yaml
configs/apps/demo/right_grasp_bottle.yaml
configs/apps/demo/dual_grasp_cup_bottle.yaml
demo.transform.camera_from_base_path
```

如果标定结果或文件路径变化，先更新这个字段。

## 3. 调整落点 Offset

先用感知 loop 检查目标识别和 3D 位置是否稳定：

```bash
python -m unirobot_infra.apps.perception.perception_loop
```

需要只看检测框时：

```bash
python -m unirobot_infra.apps.perception.detection_loop
```

用 TCP pose 工具手动验证目标姿态和落点 offset：

```bash
python -m unirobot_infra.apps.robot_ops.move_tianji_tcp_pose
```

这个工具的关键字段在 `configs/apps/robot_ops/move_tianji_tcp_pose/left.yaml`：

```yaml
tcp_pose_move:
  xyz_m: [0.593754, 0.243294, 0.260553]
  position_offset_frame: robot_world
  position_offset_m: [-0.05, 0.0, -0.10]
  orientation_frame: robot_world
  local_x_axis: [0.0, 0.0, 1.0]
  local_y_axis: [0.0, -1.0, 0.0]
```

`position_offset_m` 是 robot_world 下的平移 offset。左臂 robot_world 默认是：

```text
+x 朝前，+y 朝左，+z 朝上
```

目标抓取姿态默认是：

```text
TCP local +x 朝 world +z
TCP local +y 朝 world -y
```

如果只想临时试一个 offset，不改配置文件：

```bash
python -m unirobot_infra.apps.robot_ops.move_tianji_tcp_pose \
  tcp_pose_move.position_offset_m='[-0.04,0.02,-0.08]'
```

demo 抓取 offset 写在对应配置的 `demo.grasp_target_offset_world_m`。这个值是 robot_world 下从感知到的目标中心到 TCP 目标点的偏移。

左臂抓杯：

```text
configs/apps/demo/left_grasp_cup.yaml
```

右臂抓瓶：

```text
configs/apps/demo/right_grasp_bottle.yaml
```

## 4. 录制 Sharpa 抓取锚点

锚点录制工具会把左手先送到全零，再打开 Joint GUI。用 GUI 调整手指姿态：

```bash
python -m unirobot_infra.apps.robot_ops.sharpa_anchor_probe
```

按键：

| Key | Action |
| --- | --- |
| `s` | 保存当前锚点 |
| `q` / `Esc` / `Enter` | 退出 |

输出目录：

```text
configs/apps/adaptive_grasp/left_anchors/
```

每个锚点一个文件，文件名为整数 index：

```text
000.yaml
001.yaml
002.yaml
```

adaptive grasp 会把这些锚点按 index 顺序映射到 `alpha=0..1`，并做均匀插值。至少需要两个锚点。当前 left anchors 已有 3 个锚点。

## 5. 设置反馈阈值

单独测试自适应抓握：

```bash
python -m unirobot_infra.apps.robot_ops.sharpa_adaptive_grasp
```

空抓时可以先关闭反馈确认轨迹本身：

```bash
python -m unirobot_infra.apps.robot_ops.sharpa_adaptive_grasp \
  adaptive_grasp.feedback.enabled=false
```

阈值配置在 `configs/apps/robot_ops/sharpa_adaptive_grasp/default.yaml` 和 demo 配置的 `adaptive_grasp.feedback` 下：

```yaml
adaptive_grasp:
  feedback:
    torque_delta_threshold_by_group:
      thumb: 1.65
      index: 1.00
      middle: 0.65
      ring: 0.55
      pinky: 0.60
    position_lag_threshold_rad_by_group:
      thumb: 0.08
      index: 0.05
      middle: 0.05
      ring: 0.05
      pinky: 0.05
    min_alpha_for_stop: 0.10
```

判断原则：

- 空抓时不应过早冻结手指。
- 真抓物体时，对应手指应在接触后冻结。
- `torque_delta_threshold_by_group` 太小会空抓误停；太大会夹得更深。
- `position_lag_threshold_rad_by_group` 用于命令位置和反馈位置差异明显时停住。
- `min_alpha_for_stop` 避免刚开始运动时因为瞬态反馈误停。

## 6. 运行 Demo

入口：

```bash
python -m unirobot_infra.apps.demo.left_grasp_cup
python -m unirobot_infra.apps.demo.right_grasp_bottle
python -m unirobot_infra.apps.demo.dual_grasp_cup_bottle
```

对应配置：

```text
configs/apps/demo/left_grasp_cup.yaml
configs/apps/demo/right_grasp_bottle.yaml
configs/apps/demo/dual_grasp_cup_bottle.yaml
configs/apps/demo/grasp_common.yaml
```

当前运动参数：

```yaml
demo:
  perception_rate_hz: 2.0
  servo:
    rate_hz: 15.0
    max_translation_step_m: 0.015
    max_orientation_step_deg: 5.0
```

运动逻辑：

1. 后台感知线程按 `perception_rate_hz` 更新最新 cup target pose。
2. 主控制 loop 按 `servo.rate_hz` 固定频率读取当前 TCP。
3. 每次只朝最新 target pose 走 `max_translation_step_m` / `max_orientation_step_deg` 限幅后的一小步。
4. 使用 IK 解这一小步，并以 `blocking=false` 发送 action。
5. 到达 target 后执行 adaptive grasp。
6. 抓取完成后沿 robot_world `+z` 抬升 10 cm。

如果接近杯子时仍然太快，先减小每拍平移步长：

```bash
python -m unirobot_infra.apps.demo.left_grasp_cup \
  demo.servo.max_translation_step_m=0.008
```

如果感知丢失，demo 会继续使用最近一次有效 target；如果相机读取、检测或定位线程报错，主控制 loop 会直接报错退出。

## 7. 常见问题

| 现象 | 处理 |
| --- | --- |
| 找不到模型权重 | 确认 `models/perception/yoloe-26l-seg.pt` 存在，或覆盖 `demo.detector.model_path` |
| Rerun 中 base / TCP frame 看起来不对 | 先运行 `visualize_camera`，检查 SDK TCP frame 和 `robot.hardware.arm.arms.<side>.sdk_xyz_from_robot_world` |
| 落点偏离目标 | 用 `move_tianji_tcp_pose` 调 `position_offset_m`，再同步到 demo 配置的 `demo.grasp_target_offset_world_m` |
| 空抓就冻结手指 | 提高对应 group 的 `torque_delta_threshold_by_group` 或 `position_lag_threshold_rad_by_group` |
| Sharpa 触觉启动失败 | 按 [hardware_config.md](hardware_config.md) 检查 tactile target IP 和本机 IP |
| 相机打不开 | 按 [camera_debug.md](camera_debug.md) 检查设备占用和 USB 路径 |
