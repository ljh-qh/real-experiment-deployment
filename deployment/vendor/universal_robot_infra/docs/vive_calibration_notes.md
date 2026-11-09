# Vive 校准记录

本文只保留当前 Vive tracker -> Tianji TCP local 6D 遥操作的校准结论和复查方法。历史验证计划不再展开。

## 映射语义

Vive 不把 tracker 绝对世界坐标直接发给机械臂，而是在按住 deadman trigger 时建立锚点：

```text
T_tracker0
T_tcp0
```

同步期间使用 tracker local 相对变换控制 Tianji TCP local 相对变换：

```text
T_delta = inv(T_tracker0) @ T_tracker_now
T_tcp_target = T_tcp0 @ T_mount @ T_delta @ inv(T_mount)
```

其中 `T_mount` 是 tracker local frame 到 TCP local control frame 的固定安装修正。当前代码实现位于 `teleop/mapping/vive_arm_mapper.py`。

## 当前配置

当前入口使用：

```bash
uv run python -m unirobot_infra.apps.teleop robot=tianji_left wrappers/teleop_intervention=vive wrappers.teleop_intervention.mapping.side=left env.runtime.rate_hz=30
uv run python -m unirobot_infra.apps.teleop robot=tianji_right wrappers/teleop_intervention=vive wrappers.teleop_intervention.mapping.side=right env.runtime.rate_hz=30
uv run python -m unirobot_infra.apps.teleop robot=tianji_dual wrappers/teleop_intervention=vive wrappers.teleop_intervention.mapping.side=both env.runtime.rate_hz=30
```

按住 `I` / `O` 分别同步左/右侧；松开对应按键停止发送该侧 Tianji 动作并清空该侧锚点。下一次对应侧按下会使用新的 tracker pose 和当前 TCP pose 重新锚定。

默认 tracker 分配：

```yaml
tracker_id_by_side:
  left: 3A-A33H00153
  right: 3A-A33H00778
```

当前 mount rotation：

```yaml
left:
  - [1.0, 0.0, 0.0]
  - [0.0, -1.0, 0.0]
  - [0.0, 0.0, -1.0]
right:
  - [-1.0, 0.0, 0.0]
  - [0.0, 1.0, 0.0]
  - [0.0, 0.0, -1.0]
```

含义：

```text
left:  tracker +x -> TCP +x, tracker +y -> TCP -y, tracker +z -> TCP -z
right: tracker +x -> TCP -x, tracker +y -> TCP +y, tracker +z -> TCP -z
```

## 复查命令

如需重新确认 Tianji TCP local 轴方向，可以用小位移入口：

```bash
uv run python -m unirobot_infra.apps.robot_ops.move_tianji_tcp_local tcp_local_move.side=left tcp_local_move.axis=x tcp_local_move.distance_mm=20.0
uv run python -m unirobot_infra.apps.robot_ops.move_tianji_tcp_local robot=tianji_right tcp_local_move.side=right tcp_local_move.axis=x tcp_local_move.distance_mm=20.0
```

`axis` 可取 `x`、`y`、`z`、`-x`、`-y`、`-z`。真机运行前必须人工确认现场安全、急停、初始姿态和执行权限。
