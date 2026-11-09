# Delta EE 真机 Replay Debug 总结

本文记录 `delta_ee_world_with_hand_alpha` 数据集在真机 replay 过程中的排查、代码改动和推荐运行方式，方便复查当前实现是否符合预期。

## 当前结论

当前真机 delta ee replay 已经在 `episode_017` 和 `episode_018` 上跑通过，视觉效果上抓取定位比最初版本明显改善。

目前推荐使用：

```text
delta ee action + hand_alpha
真机 replay
chunk 内 command-target 积分
chunk 之间阻塞等待
arm tolerance = 0.1 deg
tcp position tolerance = 0.0005 m
chunk size = 16
```

这里的 `0.0005 m` 指真机实际 TCP 到当前 command target 的等待阈值。它能把 `actual vs command target` 控制到毫米级，但 `actual vs GT` 还会受到 reset 对齐、IK 分支、物体初始位置、真机跟踪残差等因素影响。

## 背景

转换后的 action 类型：

```text
delta_ee_world_with_hand_alpha
```

Action 维度为 20，每侧 10 维：

```text
left:  dx dy dz r6d_c1_x r6d_c1_y r6d_c1_z r6d_c2_x r6d_c2_y r6d_c2_z hand_alpha
right: dx dy dz r6d_c1_x r6d_c1_y r6d_c1_z r6d_c2_x r6d_c2_y r6d_c2_z hand_alpha
```

含义：

- `dx dy dz` 是 TCP 在 `robot_world` frame 下的平移 delta，单位 `m`。
- `r6d_*` 是 TCP 在 `robot_world` frame 下的相对旋转 6D 表示。
- `hand_alpha` 是 Sharpa 手的 1D 开合动作，已经由 22D 手关节映射得到。

## 排查过程

### 1. 先确认数据和 action layout

最开始确认了原始 episode 和转换后的 LeRobot dataset 是否存在，并跑了 Rerun dry-run replay。

Rerun dry-run 输出：

```text
Extracted actions shape=(324, 20)
Robot environment built with action_dim=20
Validating LeRobot action dim=20 against robot action_dim=20
Replay finished: 324 frames
```

说明数据集 action dim 和 replay env action layout 是一致的。

### 2. 初始物体位置必须和 episode 对齐

真机 replay 是开环轨迹回放，不会根据当前相机重新定位杯子/瓶子。

因此当前桌面上杯子和瓶子的初始位置必须尽量和对应 episode 录制时一致。之前 episode 初始位置没有标点，导致定位看起来像 replay 有问题；换 episode 后才更容易判断真实问题。

### 3. 第一个核心问题：下降高度不够

现象：

```text
灵巧手只到目标物体上方，没有下降到录制轨迹里的抓握高度。
```

根因：

`tianji_delta_ee_action` 在 replay 时每帧都用“当前真机反馈状态”作为 delta 的积分基准。早期 replay 下发是非阻塞的，机械臂不会在 0.1s 内完全到达上一帧 target。下一帧如果继续拿这个滞后的反馈状态加 delta，就会把轨迹压缩，累计下降量变小。

简化例子：

```text
GT:        z 每帧下降 1cm，10 帧下降 10cm
真机反馈:  每帧只追到 0.6cm
下一帧又从 0.6cm 位置继续加 delta
结果:      实际累计下降明显小于 GT
```

解决：

先改成单步阻塞 replay：

```text
env.runtime.action_blocking=true
```

每一帧发完 command 后，等待真机 arm 到达目标，再进入下一帧。这样下一帧用真机反馈状态做 delta 基准时，反馈状态已经接近上一帧 target，不会明显压缩轨迹。

这一步后，真机上的下降高度问题基本解决。

### 4. 第二个问题：横向距离目标物体仍有偏差

现象：

```text
下降高度解决后，手到达抓握高度附近，但横向距离杯子/瓶子还有偏差。
```

排查方式：

增加了真机 replay 的 TCP trajectory log。每帧记录：

- dataset GT TCP pose
- delta base pose
- delta desired pose
- IK 后 command target pose
- 真机 actual TCP pose
- 各项 position/rotation error

生成文件：

```text
debug/real_delta_ee_tcp_compare/<dataset>_episode_<idx>_<timestamp>/
  reset_pose_compare.csv
  reset_summary.csv
  actual_tcp_trajectory_compare.csv
  summary.csv
  actual_tcp_trajectory_compare.png
  chunk_end_tcp_compare.csv
  chunk_end_summary.csv
  chunk_end_tcp_compare.png
```

`actual_tcp_trajectory_compare.*` 记录的是每帧 `env.step()` 后立刻读到的 actual，chunk replay 下会包含 chunk 内机械臂滞后。`chunk_end_tcp_compare.*` 记录的是每个 chunk 末尾阻塞等待完成后的 actual，更适合判断 chunk 之间是否真正到位。

关键诊断指标：

```text
delta_base_vs_gt_current
desired_vs_gt_target
target_vs_desired
target_vs_gt_target
actual_vs_target
actual_vs_gt_target
```

含义：

- `delta_base_vs_gt_current`：当前用于积分的 base 是否已经偏离 GT 当前帧。
- `desired_vs_gt_target`：delta 积分后的笛卡尔目标是否等于 GT 下一帧。
- `target_vs_desired`：IK 后 command target 是否偏离 delta desired。
- `actual_vs_target`：真机实际 TCP 是否跟上 command target。
- `actual_vs_gt_target`：最终实际 TCP 相对 GT 目标帧的误差。

### 5. 收紧关节容差和 TCP position 容差

只用关节角容差时，即使关节误差看起来不大，TCP 末端仍可能有毫米到厘米级偏差。因此增加了 TCP position tolerance。

关键参数：

```text
env.runtime.action_arm_tolerance_deg=0.1
env.runtime.action_tcp_position_tolerance_m=0.0005
env.runtime.action_wait_timeout_s=10.0
```

`episode_018` 前 80 帧观察到的趋势：

```text
0.1deg only:
  left  actual_vs_gt_target mean ≈ 0.966cm, max ≈ 1.834cm
  right actual_vs_gt_target mean ≈ 0.581cm, max ≈ 1.022cm

0.1deg + 0.5mm TCP tolerance:
  left  actual_vs_gt_target mean ≈ 0.777cm, max ≈ 1.333cm
  right actual_vs_gt_target mean ≈ 0.380cm, max ≈ 0.579cm

actual_vs_target:
  left  max ≈ 0.050cm
  right max ≈ 0.043cm
```

这说明收紧 TCP position tolerance 后，真机实际执行到 command target 的误差已经到约 0.5mm 级；剩余 `actual_vs_gt` 误差主要来自 command target 本身和 GT 的差异、reset/IK/真实执行链路等。

### 6. 最终策略：chunk 内 command-target 积分，chunk 之间阻塞

单步阻塞能解决轨迹压缩，但未来模型通常会一次输出一个 action chunk，例如 16 帧。翔哥的要求是：

```text
不能从 reset 状态做长轨迹积分，长轨迹会漂移。
可以 chunk 内做 command-target 积分。
chunk 之间阻塞，重新用当前真机反馈状态作为下一个 chunk 的锚点。
```

当前实现：

```text
chunk 开始:
  读取当前真机反馈状态，初始化 command target state

chunk 内:
  frame 0: target = current_actual + delta_0
  frame 1: target = previous_command_target + delta_1
  frame 2: target = previous_command_target + delta_2
  ...

chunk 结束:
  等最后一帧 command target 到位
  下一 chunk 重新从当前真机反馈状态开始
```

这个方案同时避免：

- 每帧都用滞后反馈导致轨迹压缩。
- 从 reset 起点做整条长轨迹积分导致长期漂移。

## 代码改动

### `src/unirobot_infra/wrappers/tianji_delta_ee_action.py`

增加了 delta ee replay 的 debug 信息：

```text
base_pose_world
desired_pose_world
target_pose_world
base_xyz_m
desired_xyz_m
target_xyz_m
delta_xyz_m
```

增加了 chunk-level command-target 积分接口：

```text
begin_command_target_chunk()
wait_until_current_target_reached(log_label)
end_command_target_chunk()
```

chunk 内 `_expand_action()` 使用 `_command_q_rad_by_side` 作为 delta 基准，并在每帧 IK 后更新 `_command_q_rad_by_side`。

默认非 chunk 路径保持原逻辑：用最新真机反馈状态作为 delta 基准。

### `src/unirobot_infra/replay/`

新增：

- `lerobot_episode.py` 加载 `observation.state`、`frame_index`、`timestamp`，用于和 GT TCP pose 对比。
- `tcp_trajectory_log.py` 实现 `trajectory_log`，记录实际轨迹、command target、GT 轨迹并画图。
- `loop.py` 实现 `app.replay.chunk.enabled` / `app.replay.chunk.size` 和 chunk replay loop。
- `apps/replay_lerobot.py` 保持薄入口，只负责加载配置、构建 env、reset 和调用 replay 模块。

保护逻辑：

```text
app.replay.chunk.enabled=true 时，必须显式设置 env.runtime.action_blocking=false。
```

原因是 chunk 内部不应该每帧阻塞；阻塞发生在 chunk 末尾。

### `src/unirobot_infra/envs/tianji_sharpa_env.py`

新增 runtime action wait 参数：

```text
action_blocking
action_arm_tolerance_deg
action_tcp_position_tolerance_m
action_wait_timeout_s
action_wait_poll_s
action_progress_log_interval_s
```

实现了：

```text
wait_until_actions_reached(actions, log_label)
```

真机 env 中该方法会等待 arm command target 到达，并在配置了 `action_tcp_position_tolerance_m > 0` 时同时检查 TCP position error。

### `src/unirobot_infra/envs/tianji_sharpa_kinematic_env.py`

增加同名 `wait_until_actions_reached()`，用于 kinematic dry-run 的 chunk replay。

### `src/unirobot_infra/envs/tianji_sharpa_mujoco_env.py`

增加同名 `wait_until_actions_reached()`，用于 MuJoCo replay 的 chunk 接口兼容。

### `src/unirobot_infra/wrappers/base.py`

增加 wrapper 属性透传：

```text
__getattr__
```

这样最外层如果包了 Rerun wrapper，也能从 replay 层调用内层 delta wrapper 的 chunk 方法。

### 配置文件

新增默认 replay chunk 配置：

```yaml
app:
  replay:
    chunk:
      enabled: false
      size: 16
```

默认不开启 chunk，不影响旧 replay 行为。

## 关键运行命令

### 转换 episode_018

```bash
python -m unirobot_infra.apps.postprocess.convert_lerobot \
  --config-name apps/postprocess/convert/demo_dual_grasp_cup_bottle_delta_ee \
  app.data_dir=data/episodes/tj0706_pour_drink_demo/episode_018 \
  app.output_dir=data/lerobot_datasets/tj0706_pour_drink_demo_ep018_delta_ee \
  app.task="pour_drink_demo_ep018" \
  app.language_instruction="Pick up the cup with the left hand and the bottle with the right hand, pour the drink into the cup, place both objects down, and reset both arms." \
  app.overwrite=true \
  convert.repo_id=local/tj0706_pour_drink_demo_ep018_delta_ee
```

### 转换 episode_017

```bash
python -m unirobot_infra.apps.postprocess.convert_lerobot \
  --config-name apps/postprocess/convert/demo_dual_grasp_cup_bottle_delta_ee \
  app.data_dir=data/episodes/tj0706_pour_drink_demo/episode_017 \
  app.output_dir=data/lerobot_datasets/tj0706_pour_drink_demo_ep017_delta_ee \
  app.task="pour_drink_demo_ep017" \
  app.language_instruction="Pick up the cup with the left hand and the bottle with the right hand, pour the drink into the cup, place both objects down, and reset both arms." \
  app.overwrite=true \
  convert.repo_id=local/tj0706_pour_drink_demo_ep017_delta_ee
```

### Rerun dry-run

```bash
python -m unirobot_infra.apps.replay_lerobot \
  --config-name apps/replay/demo_dual_grasp_cup_bottle_delta_ee_kinematic_rerun \
  app.dataset.repo_id=local/tj0706_pour_drink_demo_ep018_delta_ee \
  app.dataset.root=data/lerobot_datasets/tj0706_pour_drink_demo_ep018_delta_ee \
  app.dataset.episode_index=0
```

### 单步阻塞真机 replay

这个版本每帧阻塞，适合和 chunk 版本做对照。

```bash
MPLCONFIGDIR=/tmp/mpl python -m unirobot_infra.apps.replay_lerobot \
  --config-name apps/replay/demo_dual_grasp_cup_bottle_delta_ee \
  app.dataset.repo_id=local/tj0706_pour_drink_demo_ep018_delta_ee \
  app.dataset.root=data/lerobot_datasets/tj0706_pour_drink_demo_ep018_delta_ee \
  app.dataset.episode_index=0 \
  app.replay.max_frames=80 \
  app.replay.trajectory_log.enabled=true \
  app.replay.trajectory_log.plot=true \
  env.runtime.action_arm_tolerance_deg=0.1 \
  env.runtime.action_tcp_position_tolerance_m=0.0005 \
  env.runtime.action_wait_timeout_s=10.0
```

### Chunk-level 真机 replay，前 80 帧

推荐先跑前 80 帧观察定位。

```bash
MPLCONFIGDIR=/tmp/mpl python -m unirobot_infra.apps.replay_lerobot \
  --config-name apps/replay/demo_dual_grasp_cup_bottle_delta_ee \
  app.dataset.repo_id=local/tj0706_pour_drink_demo_ep018_delta_ee \
  app.dataset.root=data/lerobot_datasets/tj0706_pour_drink_demo_ep018_delta_ee \
  app.dataset.episode_index=0 \
  app.replay.max_frames=80 \
  app.replay.trajectory_log.enabled=true \
  app.replay.trajectory_log.plot=true \
  app.replay.chunk.enabled=true \
  app.replay.chunk.size=16 \
  env.runtime.action_blocking=false \
  env.runtime.action_arm_tolerance_deg=0.1 \
  env.runtime.action_tcp_position_tolerance_m=0.0005 \
  env.runtime.action_wait_timeout_s=10.0
```

注意：

```text
chunk replay 必须设置 env.runtime.action_blocking=false。
```

否则 chunk 内每帧都会变成单步阻塞，和 chunk-level 语义不一致。代码里已经加了保护，开 chunk 但没关 action_blocking 会直接报错。

### Chunk-level 完整 demo

```bash
MPLCONFIGDIR=/tmp/mpl python -m unirobot_infra.apps.replay_lerobot \
  --config-name apps/replay/demo_dual_grasp_cup_bottle_delta_ee \
  app.dataset.repo_id=local/tj0706_pour_drink_demo_ep018_delta_ee \
  app.dataset.root=data/lerobot_datasets/tj0706_pour_drink_demo_ep018_delta_ee \
  app.dataset.episode_index=0 \
  app.replay.max_frames=0 \
  app.replay.trajectory_log.enabled=true \
  app.replay.trajectory_log.plot=true \
  app.replay.chunk.enabled=true \
  app.replay.chunk.size=16 \
  env.runtime.action_blocking=false \
  env.runtime.action_arm_tolerance_deg=0.1 \
  env.runtime.action_tcp_position_tolerance_m=0.0005 \
  env.runtime.action_wait_timeout_s=10.0
```

### 分析 trajectory log

运行 replay 后日志会打印输出目录，例如：

```text
debug/real_delta_ee_tcp_compare/tj0706_pour_drink_demo_ep018_delta_ee_episode_0_20260708_172646
```

查看图：

```text
debug/real_delta_ee_tcp_compare/<run_dir>/actual_tcp_trajectory_compare.png
```

命令行分析 summary：

```bash
python scripts/analyze_real_delta_ee_tcp_compare.py \
  --run-dir debug/real_delta_ee_tcp_compare/<run_dir> \
  --threshold-cm 1.0
```

## 图怎么看

`actual_tcp_trajectory_compare.png` 每行对应一只手：

- 第 1 列：XY 平面轨迹，比较 GT、delta desired、command target、actual。
- 第 2 列：`actual - GT target` 的 x/y/z 分量误差，单位 cm。
- 第 3 列：几个关键 position error 的 norm，单位 cm。
- 第 4 列：rotation error，单位 deg。

最重要看第 3 列：

```text
desired vs GT target:
  delta 积分本身是否重建了 GT 笛卡尔轨迹。

command target vs GT target:
  IK 后的 command target 是否仍然贴近 GT。

actual vs command target:
  真机是否真的追到了 command target。

actual vs GT target:
  最终真机执行相对录制 GT 的误差。
```

如果 `actual vs command target` 小，但 `command target vs GT target` 大，说明真机跟踪可以，问题更可能在 delta 基准、IK 或 GT 对齐。

如果 `command target vs GT target` 小，但 `actual vs command target` 大，说明 target 计算没问题，问题在真机执行没跟上。

`chunk_end_tcp_compare.png` 只看 chunk 末尾等待完成后的误差：

```text
actual after wait vs command target:
  每个 chunk 阻塞等待完成后，真机是否追到最后一帧 command target。

actual after wait vs GT target:
  每个 chunk 等待完成后的真实 TCP 相对 GT target 的误差。
```

这张图用于回答“chunk 之间阻塞是否到位”，比逐帧图更直接。

## 已跑验证

代码侧已跑过：

```bash
python -m py_compile \
  src/unirobot_infra/wrappers/base.py \
  src/unirobot_infra/wrappers/tianji_delta_ee_action.py \
  src/unirobot_infra/apps/replay_lerobot.py \
  src/unirobot_infra/envs/tianji_sharpa_env.py \
  src/unirobot_infra/envs/tianji_sharpa_kinematic_env.py \
  src/unirobot_infra/envs/tianji_sharpa_mujoco_env.py \
  tests/test_wrappers.py
```

```bash
pytest tests/test_envs.py -q
pytest tests/test_hydra_config.py::test_replay_lerobot_config_contract -q
pytest tests/test_wrappers.py::test_tianji_delta_ee_action_chunk_integrates_from_previous_command_target \
  tests/test_wrappers.py::test_tianji_delta_ee_action_wrapper_expands_arm_delta_and_keeps_hand_alpha \
  tests/test_wrappers.py::test_build_env_applies_tianji_delta_ee_action_wrapper_from_config -q
```

dry-run 也跑过 3 帧 chunk replay：

```bash
HF_HOME=/tmp/hf XDG_CACHE_HOME=/tmp/cache MPLCONFIGDIR=/tmp/mpl \
python -m unirobot_infra.apps.replay_lerobot \
  --config-name apps/replay/demo_dual_grasp_cup_bottle_delta_ee_kinematic_rerun \
  app.dataset.repo_id=local/tj0706_pour_drink_demo_ep018_delta_ee \
  app.dataset.root=data/lerobot_datasets/tj0706_pour_drink_demo_ep018_delta_ee \
  app.dataset.episode_index=0 \
  app.replay.max_frames=3 \
  app.replay.chunk.enabled=true \
  app.replay.chunk.size=2 \
  env.runtime.action_blocking=false \
  wrappers.rerun_visualization.enabled=false
```

真机侧：

- `episode_018` 单步阻塞 replay 视觉效果改善。
- `episode_017` 也成功 replay。
- `episode_018` chunk-level replay 已跑多次，当前观察没问题。

## 后续建议

1. 固定 `episode_018` 和 `episode_017` 的初始物体摆放标记，避免把物体初始位置误差误判成 replay 误差。
2. 用 chunk-level replay 重新保存一份完整 `trajectory_log`，比较 `actual_vs_target` 和 `actual_vs_gt_target` 曲线。
3. 如果要进一步压低 `actual_vs_gt`，优先看 `target_vs_gt_target`，确认是 delta base、IK 分支还是 reset/GT 对齐导致的剩余误差。
4. 后续接模型输出时，保持同样策略：一个 chunk 内 command-target 积分，chunk 之间等最后 target 到位并重新锚定真机反馈。
