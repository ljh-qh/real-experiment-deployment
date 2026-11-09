# Delta EE 数据集转换与回放验证

本文记录从原始 demo episode 转换为 `delta_ee_world_with_hand_alpha` LeRobot 数据集，并分别在 Rerun dry-run、MuJoCo、真机上 replay 的命令。

示例原始数据：

```text
data/episodes/tj0706_pour_drink_demo
```

示例输出数据集：

```text
data/lerobot_datasets/tj0706_pour_drink_demo_delta_ee
```

LeRobot repo id：

```text
local/tj0706_pour_drink_demo_delta_ee
```

## 动作空间
在 rerun 和 mujoco 仿真里面验证是 OK 的

转换后的 action 类型：

```text
delta_ee_world_with_hand_alpha
```

Action 总维度为 20：每侧 10 维，左右两侧拼接。

动作顺序：

```text
left:  dx dy dz r6d_c1_x r6d_c1_y r6d_c1_z r6d_c2_x r6d_c2_y r6d_c2_z hand_alpha
right: dx dy dz r6d_c1_x r6d_c1_y r6d_c1_z r6d_c2_x r6d_c2_y r6d_c2_z hand_alpha
```

含义：

- `dx dy dz`：TCP 在 `robot_world` frame 下的平移 delta，单位 `m`。
- `r6d_*`：TCP 在 `robot_world` frame 下的相对旋转矩阵前两列。第三列由 `cross(c1, c2)` 恢复。
- `hand_alpha`：Sharpa 手 `0~1` 单自由度动作，`0` 表示 open endpoint，`1` 表示 grasp endpoint。

## 状态空间

转换后的 `observation.state` 类型：

```text
tcp_pose_world_with_hand_alpha
```

State 总维度为 20：每侧 10 维，左右两侧拼接。

状态顺序：

```text
left:  x y z r6d_c1_x r6d_c1_y r6d_c1_z r6d_c2_x r6d_c2_y r6d_c2_z hand_alpha
right: x y z r6d_c1_x r6d_c1_y r6d_c1_z r6d_c2_x r6d_c2_y r6d_c2_z hand_alpha
```

含义：

- `x y z`：当前 TCP 在 `robot_world` frame 下的绝对位置，单位 `m`。
- `r6d_*`：当前 TCP 在 `robot_world` frame 下的绝对旋转矩阵前两列。
- `hand_alpha`：当前 Sharpa 手 `0~1` 状态。

左右 Tianji SDK/base frame 到 `robot_world` 的转换来自：

```text
robot.hardware.arm.arms.<side>.sdk_xyz_from_robot_world
```

`robot_world` 原点使用 Base_L/Base_R 中点，间距参数来自：

```text
robot.hardware.arm.base_l_base_r_distance_m
```

## 静止帧过滤

Delta EE 转换配置启用了静止片段压缩：

```text
convert.frame_filter.type=stationary_tcp_pose_world_with_hand_alpha
```

过滤发生在重采样之后、写入 LeRobot 之前。连续至少 `min_run_frames=5` 个 interval 同时满足下面阈值时，只保留该静止 run 的第一个 frame：

```text
max_tcp_translation_m < 0.001
max_rotation_6d < 0.005
max_hand_alpha < 0.002
```

对于 `next_state` action，过滤后的 `action[t]` 使用当前保留帧到下一保留帧的 delta，不再使用原始相邻帧 delta。

Replay 的 reset 姿态与 `apps/demo/dual_grasp_cup_bottle` 对齐：真实 replay 使用 `dual_tianji_sharpa_initial`，Rerun dry-run 使用 `kinematic_dual_tianji_sharpa_initial`，MuJoCo 使用继承了 `dual_tianji_sharpa_initial` 的 `mujoco` env。这里对齐的是 demo reset 的 TCP pose；demo reset 经过 IK 后，实际记录到的第 0 帧关节角可以和配置关节角不同。

## 离线 Delta Action 验证

离线检查脚本：

```text
scripts/validate_delta_ee_actions_offline.py
```

这个脚本不连接真机，也不跑 MuJoCo。它直接读取 LeRobot 数据集中的 `observation.state` 和 `action`，检查 delta-ee action 是否满足：

```text
state[t] + action[t] == state[t+1]
```

示例运行命令：

```bash
env HF_HOME=/tmp/hf_home HF_DATASETS_CACHE=/tmp/hf_datasets_cache \
python scripts/validate_delta_ee_actions_offline.py \
  --config-name apps/replay/demo_dual_grasp_cup_bottle_delta_ee \
  --output-dir debug/delta_ee_action_offline_validate/tj0706_pour_drink_demo_delta_ee_ep018 \
  app.dataset.repo_id=local/tj0706_pour_drink_demo_delta_ee \
  app.dataset.root=data/lerobot_datasets/tj0706_pour_drink_demo_delta_ee \
  app.dataset.episode_index=18
```

输出文件：

```text
debug/delta_ee_action_offline_validate/tj0706_pour_drink_demo_delta_ee_ep018/delta_ee_action_offline_rows.csv
debug/delta_ee_action_offline_validate/tj0706_pour_drink_demo_delta_ee_ep018/delta_ee_action_offline_summary.csv
```

重点看 `base_offset=0,target_offset=1` 这一行。如果这里的平移/旋转误差接近 0，说明数据集里的 delta action 本身和离线 state 序列是对齐的；真机 replay 的大误差就更可能来自在线执行时用 actual current 累加 target、阻塞等待未真正到位、控制频率/target 更新方式等问题。

## 转换为 Delta EE LeRobot Dataset

```bash
python -m unirobot_infra.apps.postprocess.convert_lerobot \
  --config-name apps/postprocess/convert/demo_dual_grasp_cup_bottle_delta_ee \
  app.data_dir=data/episodes/tj0706_pour_drink_demo \
  app.output_dir=data/lerobot_datasets/tj0706_pour_drink_demo_delta_ee \
  app.task="pour_drink_demo" \
  app.language_instruction="Pick up the cup with the left hand and the bottle with the right hand, pour the drink into the cup, place both objects down, and reset both arms." \
  app.overwrite=true \
  convert.repo_id=local/tj0706_pour_drink_demo_delta_ee
```

如果原始 episode 的相机流名称不是 `zed_high`、`insta_left`、`insta_right`，需要同步覆盖 `convert.cameras.*.source`。

## Rerun Dry-run Replay

这个命令不会连接 Tianji 或 Sharpa，只使用 `kinematic_dual_tianji_sharpa_initial` env 在内存里推进 `RobotState`，初始姿态与 `apps/demo/dual_grasp_cup_bottle` 使用的 `dual_tianji_sharpa_initial` 对齐，同时在 Rerun 里显示 Tianji 双臂和 Sharpa 手的 URDF mesh 动画。

```bash
python -m unirobot_infra.apps.replay_lerobot \
  --config-name apps/replay/demo_dual_grasp_cup_bottle_delta_ee_kinematic_rerun \
  app.dataset.repo_id=local/tj0706_pour_drink_demo_delta_ee \
  app.dataset.root=data/lerobot_datasets/tj0706_pour_drink_demo_delta_ee \
  app.dataset.episode_index=18
```

## MuJoCo Replay

MuJoCo backend 使用仓库内的模型资产：

```text
assets/robot/tianji_sharpa_mujoco/dual_tianji_sharpa.xml
```

该模型由 `scripts/generate_tianji_sharpa_mujoco_asset.py` 生成，包含 Tianji 双臂和左右 Sharpa 手，移除了原 Tianji URDF 里的 Robotiq 夹爪。`configs/env/mujoco.yaml` 已经配置好 model path、58 个底层 joint name 映射，并继承 `dual_tianji_sharpa_initial` 作为 reset 姿态。

运行完整 MuJoCo replay。`env=mujoco` 会打开 MuJoCo viewer 窗口，并在 replay 结束后保留 3 秒，方便确认最后姿态：

```bash
python -m unirobot_infra.apps.replay_lerobot \
  --config-name apps/replay/demo_dual_grasp_cup_bottle_delta_ee \
  env=mujoco \
  app.sdk_output.enabled=false \
  app.dataset.repo_id=local/tj0706_pour_drink_demo_delta_ee \
  app.dataset.root=data/lerobot_datasets/tj0706_pour_drink_demo_delta_ee \
  app.dataset.episode_index=18
```

当前 MuJoCo env 使用 `qpos` 模式：replay action 经过 `hand_alpha_action` 和 `tianji_delta_ee_action` wrapper 后，被写到模型对应 joint 的 `qpos`，再从 MuJoCo `qpos` 读回 `RobotState`。这适合验证 replay/action/state pipeline。真实接触、夹持力、actuator 控制还需要继续扩展 actuator、ctrl、contact 和物体模型。

## 真机 Replay

真机 replay 会使用 `dual_tianji_sharpa_initial` reset 姿态，连接 Tianji 和 Sharpa。确认 Rerun/MuJoCo 验证通过、工作空间安全、操作人准备好急停后，运行完整 episode：

```bash
python -m unirobot_infra.apps.replay_lerobot \
  --config-name apps/replay/demo_dual_grasp_cup_bottle_delta_ee \
  app.dataset.repo_id=local/tj0706_pour_drink_demo_delta_ee \
  app.dataset.root=data/lerobot_datasets/tj0706_pour_drink_demo_delta_ee \
  app.dataset.episode_index=18
```

真机 replay 前必须确认：

```text
1. Rerun dry-run 已经跑通。
2. action dim 等于 replay env action_layout size。
3. 初始姿态安全，且 replay 数据和当前 reset 姿态一致。
4. 工作空间清空。
5. 操作人准备好随时急停。
```

## 真机 Replay with TCP Error Log

每个 step 都阻塞等待当前 delta-ee target 到位，并输出 TCP error CSV 和 plot：

```bash
python -m unirobot_infra.apps.replay_lerobot \
  --config-name apps/replay/demo_dual_grasp_cup_bottle_delta_ee \
  app.dataset.repo_id=local/tj0706_pour_drink_demo_delta_ee \
  app.dataset.root=data/lerobot_datasets/tj0706_pour_drink_demo_delta_ee \
  app.dataset.episode_index=18 \
  app.replay.rate_hz=12.0 \
  env.runtime.action_blocking=true \
  app.replay.chunk.enabled=false \
  app.replay.trajectory_log.enabled=true \
  app.replay.trajectory_log.type=delta_ee_tcp \
  app.replay.trajectory_log.output_dir=debug/real_delta_ee_tcp_compare/tj0706_pour_drink_demo_delta_ee_ep018/every_step_block \
  app.replay.trajectory_log.plot=true
```

按 chunk 发送：chunk 内按 replay 频率非阻塞发送，chunk 末阻塞等待最后一个 target 到位，并输出 TCP error CSV 和 plot：

```bash
python -m unirobot_infra.apps.replay_lerobot \
  --config-name apps/replay/demo_dual_grasp_cup_bottle_delta_ee \
  app.dataset.repo_id=local/tj0706_pour_drink_demo_delta_ee \
  app.dataset.root=data/lerobot_datasets/tj0706_pour_drink_demo_delta_ee \
  app.dataset.episode_index=18 \
  app.replay.rate_hz=12.0 \
  env.runtime.action_blocking=false \
  app.replay.chunk.enabled=true \
  app.replay.chunk.size=32 \
  app.replay.trajectory_log.enabled=true \
  app.replay.trajectory_log.type=delta_ee_tcp \
  app.replay.trajectory_log.output_dir=debug/real_delta_ee_tcp_compare/tj0706_pour_drink_demo_delta_ee_ep018/chunk_level_block \
  app.replay.trajectory_log.plot=true
```
