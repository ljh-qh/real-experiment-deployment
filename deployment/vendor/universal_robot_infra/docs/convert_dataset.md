# Dataset Conversion

本文描述 raw episode（`robot_states.h5` + MP4/MKV）到 LeRobot dataset 的后处理工具。实现参考 `FrankyFrankaGym/docs/convert_dataset.md`，但字段配置适配 unirobot_infra 当前 recording schema。

## 依赖

基础 reader、resampler 和时间戳 HTML 报告只依赖当前项目已有依赖。真正导出或可视化 LeRobot dataset 需要额外安装：

```bash
uv sync --group postprocess
```

如果需要 push 到 HuggingFace Hub，还需要登录：

```bash
huggingface-cli login
```

## 配置

默认配置在 `configs/apps/postprocess/convert/default.yaml`。该文件同时包含 app 运行参数和转换 schema：

- `app` 控制脚本运行行为，例如 raw 数据目录、输出目录、任务短名、语言指令、是否覆盖、是否 push。
- `convert` 控制 LeRobot dataset schema 和转换规则，例如 state/action 字段、相机、触觉力、触觉图像保存格式。

```yaml
app:
  data_dir: data/episodes
  output_dir: data/lerobot_dataset
  task: pour_drink
  language_instruction: Pour the drink into the shaker and place the shaker on the tray
  push: false
  overwrite: false
  inspect_timestamps: false
  update_readme_only: false

convert:
  tactile_image_format: png
  target_fps: 10
  repo_id: "local/tj0623"
  robot_type: "tianji_sharpa"

  state_keys:
    - left_arm.joint_pos
    - left_end_effector.joint_pos
    - right_arm.joint_pos
    - right_end_effector.joint_pos

  action:
    type: joint_position
    source: sent_action
    keys: []

  cameras:
    head_left:
      source: zed_high
      crop:
        top: 0
        bottom: 0
        left: 0
        right: 0
      resize: [640, 360]
    left_wrist:
      source: insta_left
      crop:
        top: 0
        bottom: 0
        left: 0
        right: 0
      resize: [640, 360]
    right_wrist:
      source: insta_right
      crop:
        top: 0
        bottom: 0
        left: 0
        right: 0
      resize: [640, 360]

  tactile_force:
    field: f6
    sensors:
      - side: left
        finger: thumb

  tactile_images:
    tactile_left_raw_thumb:
      side: left
      finger: thumb
      modality: raw
      resize: null
```

默认配置面向当前 tj0623 数据：58 维关节位置状态、`sent_action`、60 维触觉力、`head_left/left_wrist/right_wrist` 三路相机，以及 20 路触觉图像。上面的 YAML 只展示了触觉配置的一部分，完整列表以 `configs/apps/postprocess/convert/default.yaml` 为准。raw episode 目录和输出目录可以按实际磁盘路径写入配置，或用 Hydra override 覆盖。

字段 token 使用 `<stream>.<field>`：

- `stream` 对应 HDF5 里的 `robot_state/<stream>`，例如 `left_arm`、`right_end_effector`。
- `field` 对应该 stream 下的数据集名，例如 `joint_pos`、`joint_vel`、`joint_torque`、`est_cart_force`。
- `state_keys` 按列表顺序拼接为 `observation.state`。
- `convert.state.type: tcp_pose_world_with_hand_alpha` 会把 state 派生为每侧 `tcp_pose_world[9] + hand_alpha[1]`，左右双臂共 20 维；源数据仍由 `arm_keys` / `hand_keys` 指定。
- `frame_filter` 控制是否在重采样后、写 LeRobot 前压缩静止片段；启用 `stationary_tcp_pose_world_with_hand_alpha` 时，会把连续满足阈值的静止 run 只保留第一个 frame，并让 `next_state` action 指向下一条保留帧。
- `cameras` 会导出为 LeRobot `video` feature，配置 key 是输出 feature 名，`source` 是 raw episode 里的 MP4/HDF5 camera stream 名；`resize: null` 表示保持 crop 后原始相机分辨率。
- `tactile_force` 会把 `tactile/<side>/<finger>/<field>` 按配置顺序拼接为 `observation.tactile_force`；`null` 表示不导出触觉力向量。
- `tactile_image_format` 控制触觉图像导出格式：`mp4` 导出为 LeRobot `video` feature，写到 `videos/observation.images.<name>/...mp4`；`png` 导出为 LeRobot `image` feature，按 PNG 无损编码嵌入 parquet。默认配置使用 `png`。
- `tactile_images` 转换时读取 raw episode 的触觉 FFV1/MKV 帧；`resize: null` 表示保持原始触觉图像尺寸。

`action.source` 支持：

| source | 含义 |
| --- | --- |
| `next_state` | `action[t]` 使用 `action.keys` 在 `t+1` 的状态值，因此每个 episode 最后一帧会被丢弃 |
| `sent_action` | 使用 HDF5 `action/sent_action` 按时间戳最近邻对齐 |
| `policy_action` | 使用 HDF5 `action/policy_action` 按时间戳最近邻对齐 |

`sent_action` / `policy_action` 只适用于 `teleop_record` 写出了 action stream 的 episode。字段缺失会直接报错。

## 降采样和导出

使用默认转换 schema：

```bash
python -m unirobot_infra.apps.postprocess.convert_lerobot
```

覆盖输入输出目录：

```bash
python -m unirobot_infra.apps.postprocess.convert_lerobot \
  app.data_dir=data/episodes \
  app.output_dir=data/lerobot_dataset
```

如果输出目录已经存在，并且确认可以覆盖，加上：

```bash
app.overwrite=true
```

默认不会生成时间戳对齐 HTML。需要调试重采样时再打开：

```bash
python -m unirobot_infra.apps.postprocess.convert_lerobot \
  app.inspect_timestamps=true
```

触觉图像默认按 LeRobot `image` feature 存为 parquet 内嵌 PNG 无损图像。如果要改成 LeRobot `video` feature/MP4：

```bash
python -m unirobot_infra.apps.postprocess.convert_lerobot \
  app.data_dir=data/episodes \
  app.output_dir=data/lerobot_dataset_mp4 \
  convert.tactile_image_format=mp4
```

如果不需要导出任何触觉图像/视频 feature，使用无触觉图像配置：

```bash
python -m unirobot_infra.apps.postprocess.convert_lerobot \
  --config-name apps/postprocess/convert/no_tactile_images \
  app.data_dir=data/episodes \
  app.output_dir=data/lerobot_dataset_no_tactile_images
```

常用 Hydra 配置项：

| 配置项 | 含义 |
| --- | --- |
| `app.data_dir` | 包含 `episode_001/`、`episode_002/` 的 raw 数据目录 |
| `app.output_dir` | LeRobot dataset 输出目录 |
| `app.task` | 任务短名；写入 `info.json` 的 `task_name` 扩展字段和 dataset card |
| `app.language_instruction` | 语言指令；按 LeRobot 设计写入每帧 `task`，最终保存到 `meta/tasks.parquet` 并映射为 `task_index` |
| `app.push` | 转换完成后 push 到 HuggingFace Hub |
| `app.overwrite` | 输出目录已存在时直接覆盖，不再交互询问 |
| `convert.tactile_image_format` | 触觉图像导出格式；`mp4` 为 LeRobot video feature，`png` 为 LeRobot image feature 的无损 PNG |
| `app.inspect_timestamps` | 每条 raw episode 旁边写 `resampling_timeline_<fps>hz.html` |
| `app.update_readme_only` | 只更新已有 dataset 的 `README.md`，不重新转换数据；README 会从现有 dataset metadata 推断触觉图像格式 |

如果 `app.output_dir` 已存在且 `app.overwrite=false`，脚本会询问是否覆盖。

## Dataset Card Only

两种方式都可以只更新 dataset card：

```bash
python -m unirobot_infra.apps.postprocess.convert_lerobot \
  app.update_readme_only=true
```

或使用独立脚本：

```bash
python -m unirobot_infra.apps.postprocess.update_lerobot_readme \
  --config configs/apps/postprocess/convert/default.yaml \
  --output-dir data/lerobot_dataset \
  --task "pour_drink" \
  --language-instruction "Pour the drink into the shaker and place the shaker on the tray" \
  --data-dir data/episodes
```

该模式只写 `README.md`，不改 parquet、视频或 metadata。

## Rerun 可视化

LeRobot 自带 viewer：

```bash
python -m lerobot.scripts.lerobot_dataset_viz \
  --repo-id dummy \
  --episode-index 0 \
  --root data/lerobot_dataset
```

unirobot_infra 提供 grouped state/action plot：

```bash
python -m unirobot_infra.apps.postprocess.visualize_lerobot \
  --repo-id dummy \
  --episode-index 0 \
  --root data/lerobot_dataset
```

保存 `.rrd` 而不打开 viewer：

```bash
python -m unirobot_infra.apps.postprocess.visualize_lerobot \
  --repo-id dummy \
  --episode-index 0 \
  --root data/lerobot_dataset \
  --save \
  --output-dir data/rerun
```

可选：

- `--state-labels ...` 自定义 `observation.state` 图例。
- `--action-labels ...` 自定义 `action` 图例。
- `--display-compressed-images` 降低 Rerun 内存占用。

## 时间戳对齐可视化

按 convert config 检查一条 episode：

```bash
python -m unirobot_infra.apps.postprocess.inspect_resampling_timeline \
  --episode-dir data/episodes/episode_001 \
  --target-fps 10 \
  --config configs/apps/postprocess/convert/default.yaml
```

不提供 config 时，脚本会显示该 episode 里所有 `robot_state` stream 和所有 camera stream：

```bash
python -m unirobot_infra.apps.postprocess.inspect_resampling_timeline \
  --episode-dir data/episodes/episode_001 \
  --target-fps 10
```

输出 HTML 中：

- `grid` 是目标均匀时间网格。
- `state:<stream>` 是对应 robot state stream 的原始时间戳和匹配时间戳。
- `camera:<name>` 是相机帧时间戳和匹配帧。
- `action` 只在配置使用 `sent_action` / `policy_action` 或手动加 `--include-action` 时显示。
- 红色 tick 表示该帧与最近原始样本的时间差超过 `2 / target_fps` 秒。
- 鼠标滚轮或触控板滚动可以缩放，横向拖拽可以平移，悬停在点或 tick 上会显示时间戳和 gap。

## Push

转换时直接 push：

```bash
python -m unirobot_infra.apps.postprocess.convert_lerobot \
  app.push=true
```

push 已有本地 dataset：

```bash
python -m unirobot_infra.apps.postprocess.push_lerobot \
  --repo-id user/tianji_sharpa_dataset \
  --root data/lerobot_dataset \
  --tags tianji sharpa
```

## 输出格式

输出目录遵循 LeRobot dataset 布局，典型结构为：

```text
data/lerobot_dataset/
+-- README.md
+-- meta/
|   +-- info.json
|   +-- stats.json
|   +-- tasks.parquet
|   \-- episodes/
+-- data/
\-- videos/
```

相机始终位于 `videos/observation.images.<name>/...mp4`。默认配置中，触觉图像作为 LeRobot `image` feature 以 PNG 无损编码嵌入 `data/...parquet`；如果使用 `convert.tactile_image_format=mp4`，触觉图像会作为 LeRobot `video` feature 写到 `videos/observation.images.<name>/...mp4`。LeRobot dataloader 读取后，它们都会表现为 `observation.images.*` 下的 `C,H,W` 图像 tensor。触觉力 `observation.tactile_force` 和状态/action 位于 `data/...parquet`。触觉 metadata 仍位于 raw episode 的 `tactile/<side>/<finger>` 组下，例如 `tactile/left/thumb/f6` 和 `tactile/left/thumb/contact_point`。
