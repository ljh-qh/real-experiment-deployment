# Quick Start

以下命令默认在 `unirobot_infra` 目录下执行：

```bash
cd /home/ubuntu/chensixiang/Codes/Universal_Robot_Infra
```

如果当前环境已经安装了本包，可以直接使用 `python -m ...`。如果使用本仓库的 `uv` 环境，把命令前缀换成 `uv run python -m ...`。

## Teleop

```bash
python -m unirobot_infra.apps.teleop
```

## Teleop Record

```bash
python -m unirobot_infra.apps.teleop_record
```

启动后按 `Space` 开始当前 episode；录制中按 `Right` 保存并进入下一条，按 `Left` 丢弃当前 episode 并重录，按 `Q` 丢弃当前 episode 并退出。默认录制 1 条，可以用 `app.record_control.num_episodes=10` 覆盖。

## Realtime Capture

```bash
python -m unirobot_infra.apps.realtime_capture
```

## Echo Robot State

```bash
python -m unirobot_infra.apps.robot_ops.echo_robot_state
```

## Vive Probe

```bash
python -m unirobot_infra.apps.vive_probe
```

## Move Tianji TCP Local

```bash
python -m unirobot_infra.apps.robot_ops.move_tianji_tcp_local
```

## Set Robot State

```bash
python -m unirobot_infra.apps.robot_ops.set_robot_state
```

## Convert Dataset

```bash
python -m unirobot_infra.apps.postprocess.convert_lerobot app.inspect_timestamps=true
```

只更新 dataset card：

```bash
python -m unirobot_infra.apps.postprocess.convert_lerobot app.update_readme_only=true
```

Rerun 可视化：

```bash
python -m unirobot_infra.apps.postprocess.visualize_lerobot --repo-id dummy --episode-index 0 --root data/lerobot_dataset
```

## Replay LeRobot Dataset

```bash
python -m unirobot_infra.apps.replay_lerobot \
  app.dataset.repo_id=local/dataset \
  app.dataset.root=data/lerobot_dataset/lerobot_dataset_test \
  app.dataset.episode_index=0
```

默认以 10Hz 固定频率发送 action；需要改用 dataset fps 时覆盖 `app.replay.rate_hz=null`。
