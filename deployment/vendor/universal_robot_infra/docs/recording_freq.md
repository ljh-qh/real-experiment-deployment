# Recording Frequency

本页记录 `teleop_record.py` 在当前真实硬件上的 20 秒录制消融测试结果。

## 测试方式

- 测试日期：2026-06-17。
- 测试入口：`src/unirobot_infra/apps/teleop_record.py`。
- Python 环境：项目 `.venv`，Python 3.12；ZED 使用 `pyzed==5.2` cp312 wheel。
- 通用参数：`wrappers/teleop_intervention=joint_gui`，`env.runtime.rate_hz=30`，`app.record_control.auto_start=true`，`app.record_control.episode_duration_s=20.0`，`app.status_gui.enabled=false`。
- 统计口径：从 HDF5 的 `sys_time_ns` 首尾差计算 `(samples - 1) / duration_s`，不使用日志 wall-clock。日志 wall-clock 包含视频编码、HDF5 close 和硬件断开。
- 输出目录：`data/teleop_record_freq_ablation_current/<group>/episode_001`。
- 当前默认 recording 配置录制 3 相机、双臂 Tianji 高频实时字段、双手 Sharpa joint packet 字段和双手触觉。状态字段在配置中显式列出，不使用 `*`。
- 注：本页表格里的实测数据对应当时默认 state stream 仍为 `1000/500 Hz` 的版本；当前默认值已统一下调到 `360/360 Hz`。

## 频率定义

- Tianji state Hz：`robot_state/left_arm/sys_time_ns` 和 `robot_state/right_arm/sys_time_ns` 的写入频率，默认记录 Tianji SDK 高频实时包里的状态字段，不记录 `tool_kine/tool_dyn` 等工具配置。
- Sharpa state Hz：`robot_state/left_end_effector/sys_time_ns` 和 `robot_state/right_end_effector/sys_time_ns` 的写入频率，默认只记录 Sharpa joint packet 的 `sdk_timestamp`、`sequence`、`joint_pos`、`joint_vel`、`joint_torque`。
- Sharpa tactile Hz/指：`tactile/<side>/<finger>/sys_time_ns` 的逐指写入频率。它是独立触觉帧采集线程，不等于 Sharpa state stream 频率。
- Camera Hz：`camera/<camera_name>/sys_time_ns` 的写入频率。
- Action Hz：`teleop_record.py` 主循环每次 `env.step()` 后写入 `action/*` 的频率。它表示 action dispatch / teleop loop 频率，不是硬件状态读取频率，因此不放在主频率表里。

## 频率结果

| 组 | 配置增量 | Tianji state Hz L/R | Sharpa state Hz L/R | Sharpa tactile Hz/指 L/R | insta_left Hz | insta_right Hz | zed_high Hz | episode 大小 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 01 | 单左臂高频实时状态，无相机/触觉 | 997.55 / - | - | - | - | - | - | 29.9 MiB |
| 02 | 双臂高频实时状态，无相机/触觉 | 993.37 / 992.99 | - | - | - | - | - | 59.8 MiB |
| 03 | 组 02 + 双手 Sharpa joint packet，无相机/触觉 | 990.13 / 991.34 | 497.47 / 497.48 | - | - | - | - | 70.9 MiB |
| 04 | 组 03 + 3 相机 | 989.46 / 989.52 | 497.16 / 497.21 | - | 29.90 | 30.01 | 30.03 | 99.4 MiB |
| 05 | 组 04 + 双手触觉 raw/deform 视频与 metadata | 988.06 / 987.99 | 496.92 / 496.89 | 30.00 / 30.00 | 29.92 | 29.78 | 30.02 | 264.6 MiB |

触觉频率是 HDF5 `tactile/<side>/<finger>/sys_time_ns` 的逐指平均值。第 05 组左手每指 602 帧，右手 601-602 帧。触觉 `RAW`/`DEFORM` 图像写为 FFV1/MKV，无损；触觉 metadata、`F6` 和 `CONTACT_POINT` 写入 HDF5。

## Action 主循环频率

| 组 | action Hz | action samples |
| --- | ---: | ---: |
| 01 | 19.64 | 393 |
| 02 | 9.81 | 197 |
| 03 | 9.79 | 196 |
| 04 | 9.78 | 196 |
| 05 | 9.77 | 196 |

## 结论

- 当前代码口径下，双臂 Tianji 高频实时状态在全量配置下仍约 `988Hz`。
- 双手 Sharpa joint packet 状态在全量配置下约 `497Hz`，接近当时默认 `500Hz` 目标。
- 三相机都达到约 `30Hz`；第 05 组 `insta_right` 为 `29.78Hz`，仍接近标称 30Hz。
- 双手触觉逐指达到约 `30Hz`；触觉全量加入后没有明显拉低 Sharpa joint state 频率。
- `device_info`、`runtime.*`、`tactile.summary` 和 `tool_kine/tool_dyn` 不作为 robot state 采集。
- `teleop_record.py` 的 action 主循环在真实双臂双手配置下约 `9.8Hz`。这不是 robot state 采样线程频率；state、camera、tactile 都由独立采集线程写入。

## 当前默认配置

`configs/recording/default.yaml` 当前默认值：

- `camera_names`: `insta_left`、`insta_right`、`zed_high`
- `recording.tactile.enabled: true`
- `recording.tactile.target_hz: 30.0`
- `recording.tactile.modalities`: `raw`、`deform`
- `left_arm` / `right_arm`: `target_hz: 360.0`，`states` 显式列出 Tianji 高频实时字段，不含 `tool_kine/tool_dyn`
- `left_end_effector` / `right_end_effector`: `target_hz: 360.0`，`states: [sdk_timestamp, sequence, joint_pos, joint_vel, joint_torque]`

设备身份、连接配置、工具配置、runtime 诊断和 tactile summary 不在默认 recording 中采集。触觉图像和触觉低维量由 `recording.tactile` 独立采集。

## 实验输出

| 组 | 目录 |
| --- | --- |
| 01 | `data/teleop_record_freq_ablation_current/01_left_arm_state/episode_001` |
| 02 | `data/teleop_record_freq_ablation_current/02_dual_arm_state/episode_001` |
| 03 | `data/teleop_record_freq_ablation_current/03_dual_arm_hand_joint_state/episode_001` |
| 04 | `data/teleop_record_freq_ablation_current/04_dual_arm_hand_joint_3cam/episode_001` |
| 05 | `data/teleop_record_freq_ablation_current/05_dual_arm_hand_joint_3cam_tactile/episode_001` |

## 运行备注

- 第 04 组初次启动时因为当前 `.venv` 缺少 Python 3.12 的 ZED binding 失败；随后从 Stereolabs 官方 URL 下载并安装 `pyzed-5.2-cp312-cp312-linux_x86_64.whl` 到项目 `.venv`。
- 三相机实验前，两台 Insta360 GO 3S 出现过 V4L warmup 偶发超时。手动重启相机后，两台 GO 3S 均能稳定读帧，本页结果来自重启后的有效录制。
