# Hardware Config

本文记录 unirobot_infra 当前硬件的初次配置方式，以及日常录制前应该检查的网络、设备路径和 SDK 依赖。

配置来源：

- `configs/robot_hardware/arm/tianji.yaml`
- `configs/robot_hardware/end_effector/sharpa.yaml`
- `configs/camera/default.yaml`
- `configs/device/vive/default.yaml`
- `configs/wrappers/teleop_intervention/*.yaml`

## 为什么会需要重新配置 IP

这次 Sharpa 触觉启动失败不是因为代码要求每次重新配置 IP，而是 SDK 在启动触觉流时检查到：

- 手内部 heartbeat 里配置的 tactile 目标地址是 `192.168.10.240`
- 本机当前 `eno1` 上没有 `192.168.10.240`
- SDK 因此报 `host_ip != target_ip (target not local)`，拒绝启动 tactile stream

断线重连通常不会改掉手内部保存的 tactile 目标地址；但如果之前只是临时给本机加过 `192.168.10.240/24`，这个地址会在重启、NetworkManager 重启、网卡禁用/启用、网线重新接入后消失。因此“之前配置过”仍然可能在当前会话里失效。

先检查本机当前地址：

```bash
ip -br addr
```

如果 Sharpa 触觉目标仍然是 `192.168.10.240`，可以临时给 `eno1` 加地址：

```bash
sudo ip addr add 192.168.10.240/24 dev eno1
```

录制结束后可删除临时地址：

```bash
sudo ip addr del 192.168.10.240/24 dev eno1
```

如果要持久化，使用当前系统的网络管理工具把 `192.168.10.240/24` 追加到 `eno1` 对应连接上。NetworkManager 示例：

```bash
nmcli con show --active
sudo nmcli con mod <connection-name> +ipv4.addresses 192.168.10.240/24
sudo nmcli con up <connection-name>
```

注意用 `+ipv4.addresses` 追加地址，不要覆盖已有的 `192.168.1.180/24` 和 `192.168.10.180/24`。

另一种方案是使用 Sharpa 厂商工具把手内部的 `tactile_data_config.target_ip` 改成当前主机地址 `192.168.10.180`。当前仓库的 `network.host_ip: 192.168.10.180` 记录的是期望主机地址，现有连接流程不会用它自动改写手内部 heartbeat 的 tactile 目标 IP。

## 主机网络

当前默认假设有线网卡为 `eno1`，同一块网卡挂多个 IPv4 地址：

| 用途 | 本机地址 | 远端地址 |
| --- | --- | --- |
| Tianji 控制柜 | `192.168.1.180/24` | `192.168.1.190` |
| Sharpa 控制/发现 | `192.168.10.180/24` | `192.168.10.10`, `192.168.10.20` |
| Sharpa 触觉接收 | `192.168.10.240/24` | 由手内部 tactile target 指向本机 |

基础检查：

```bash
ip -br addr
ping -c 2 192.168.1.190
ping -c 2 192.168.10.10
ping -c 2 192.168.10.20
```

如果使用 Vive tracker，默认从无线网卡 `wlp0s20f3` 接收 `tcp://192.168.1.225:5555`：

```bash
ip -br addr show wlp0s20f3
```

## Tianji 双臂

配置文件：`configs/robot_hardware/arm/tianji.yaml`

当前配置：

| 项 | 值 |
| --- | --- |
| SDK 目录 | `/home/ubuntu/rl/PDM_test/PDMode_Test/SDK_PYTHON` |
| 运动学配置 | `../TEST_SCRIPT/ccs_m6_40.MvKDCfg` |
| 控制柜 IP | `192.168.1.190` |
| 左臂 arm id | `A` |
| 右臂 arm id | `B` |

初次配置：

1. 确认主机 `eno1` 有 `192.168.1.180/24`。
2. 确认 Tianji 控制柜 IP 为 `192.168.1.190`，并且与主机处于同一网段。
3. 确认 Tianji SDK 目录存在，且 Python 3.12 能 import `fx_robot`。
4. 在真正 teleop 或 record 前，先用只读状态读取确认左右臂状态正常。

检查命令：

```bash
test -d /home/ubuntu/rl/PDM_test/PDMode_Test/SDK_PYTHON
PYTHONPATH=/home/ubuntu/rl/PDM_test/PDMode_Test/SDK_PYTHON python -c "import fx_robot; print('fx_robot ok')"
cd /home/ubuntu/chensixiang/Codes/Universal_Robot_Infra
python -m unirobot_infra.apps.robot_ops.echo_robot_state
```

`echo_robot_state` 当前只调用 `TianjiRobot.connect()` 和状态读取，不会进入 position mode。`teleop` 和 `teleop_record` 会进入控制流程，应该在网络和状态检查通过后再运行。

### Tianji SDK `port bind failure`

如果 Tianji SDK 日志里出现：

```text
Map Master success
port bind failure, possibly occupied by another program
RuntimeError: Failed to connect Tianji robot at 192.168.1.190
```

优先按本机端口占用处理。`/var/tmp/AAA_shm already exists` 通常只是 SDK 复用共享内存文件的提示，不是本次失败的直接原因；真正关键是 `port bind failure`。

先查是否有厂商 MarvinPlatform UI 或其它 Python 程序占用了 Tianji SDK UDP 端口：

```bash
ps -ef | grep -E 'MarvinPlatform|ui.py|unirobot|move_tianji|set_robot_state' | grep -v grep
ss -lunp | grep -E ':4730\b'
```

如果看到类似：

```text
python ui.py
UDP *:4730 users:(("python",pid=2296421,fd=6))
```

说明厂商 UI 仍占着 SDK 端口。确认进程目录后结束它：

```bash
pwdx 2296421
kill 2296421
sleep 1
ss -lunp | grep -E ':4730\b'
```

如果进程不退出，再手动确认不是正在控制机械臂的程序，然后使用 `kill -9 <pid>`。端口释放后，先用只读状态读取验证连接，暂时不要直接发送运动命令：

```bash
python -m unirobot_infra.apps.robot_ops.echo_robot_state robot=tianji_right robot_state.sides='[right]'
```

确认能读到 `cur_state`、`cmd_state`、`err_code` 后，再根据现场安全状态决定是否运行 `set_robot_state` 或其它会进入 position mode 的程序。

## Sharpa 双手和触觉

配置文件：`configs/robot_hardware/end_effector/sharpa.yaml`

当前配置：

| 项 | 值 |
| --- | --- |
| SDK 根目录 | `/home/ubuntu/chensixiang/Codes/sharpa-manus-sdk` |
| 期望主机控制地址 | `192.168.10.180` |
| 左手 SN | `C0509131C050` |
| 左手 IP | `192.168.10.10` |
| 右手 SN | `C8519039C851` |
| 右手 IP | `192.168.10.20` |
| 默认触觉 | `skip_tactile: false` |
| 默认连接使能 | `enable_on_connect: true` |

初次配置：

1. 确认主机 `eno1` 有 `192.168.10.180/24`。
2. 确认左手和右手分别为 `192.168.10.10`、`192.168.10.20`。
3. 确认两只手的 SN 与配置文件一致，避免左右手或设备替换后误连。
4. 确认 Sharpa SDK 根目录存在，里面包含 SharpaWave Python binding、动态库和 Manus client。
5. 如果要采集稠密触觉，保持 `skip_tactile: false`。
6. 检查手内部 `tactile_data_config.target_ip`。如果目标是 `192.168.10.240`，主机必须持有 `192.168.10.240/24`，或者把手内部目标 IP 改成 `192.168.10.180`。

基础检查：

```bash
test -d /home/ubuntu/chensixiang/Codes/sharpa-manus-sdk
ping -c 2 192.168.10.10
ping -c 2 192.168.10.20
ip -br addr show eno1
```

2026-06-17 实机检查时，左手启动触觉失败的原因是手内部 tactile 目标为 `192.168.10.240:50011`，而本机没有 `192.168.10.240`。右手通常对应 `192.168.10.240:50001`。如果两只手重新配置过，应该以实际 `DeviceInfo.tactile_data_config` 为准。

常见错误：

| 报错 | 原因 | 处理 |
| --- | --- | --- |
| `No Sharpa ... discovered` | 主机网段、手 IP、SN 不匹配 | 检查 `192.168.10.180/24`、手 IP 和 SN |
| `host_ip != target_ip` | 手的 tactile target 不是本机当前地址 | 给本机添加对应 IP，或改手内部 tactile target |
| `failed to start` | 触觉端口或 SDK 状态异常 | 先检查 target IP，再检查端口占用和 SDK 日志 |

端口占用检查：

```bash
ss -uapn | grep -E '50001|50011'
```

## 相机

配置文件：`configs/camera/default.yaml`

当前默认录制三路相机：

| 名称 | 类型 | 标识 | 分辨率 | FPS |
| --- | --- | --- | --- | --- |
| `insta_left` | `insta360_v4l2` | `/dev/v4l/by-path/pci-0000:0a:00.0-usb-0:2:1.0-video-index0`, USB `9-2` | `1280x720` | `30` |
| `insta_right` | `insta360_v4l2` | `/dev/v4l/by-path/pci-0000:08:00.0-usb-0:2:1.0-video-index0`, USB `5-2` | `1280x720` | `30` |
| `zed_high` | `zed_sdk` | serial `33712565` | `1280x720` | `30` |

初次配置：

1. 固定两台 Insta360 GO 3S 到当前 USB 拓扑，确认 `/dev/v4l/by-path/...` 与配置一致。
2. 确认 V4L2 设备名包含 `GO 3S`，避免系统把其他 UVC 设备匹配进来。
3. 确认 ZED 2i 的 serial number 是 `33712565`。
4. 确认 `pyzed` SDK 与 Python 3.12 环境匹配。

检查命令：

```bash
ls -l /dev/v4l/by-path/
for dev in /sys/class/video4linux/video*/name; do echo "$dev: $(cat "$dev")"; done
python -c "import pyzed.sl as sl; print('pyzed ok')"
cd /home/ubuntu/chensixiang/Codes/Universal_Robot_Infra
python -m unirobot_infra.apps.realtime_capture realtime_capture.max_frames=30 realtime_capture.display=false
```

如果 USB 拓扑变化导致 `/dev/v4l/by-path/...` 不存在，应先确认相机实际枚举路径，再更新 `configs/camera/default.yaml`。不要只改成 `/dev/video0` 这类不稳定路径。

## Manus

配置文件：

- `configs/wrappers/teleop_intervention/manus.yaml`
- `configs/wrappers/teleop_intervention/manus_vive.yaml`

当前配置：

| 项 | 值 |
| --- | --- |
| mocap endpoint | `tcp://127.0.0.1:2044` |
| expected keypoints | `25` |
| frame side | `left` |
| frame name | `manus` |
| client auto start | `true` |
| client log | `/tmp/unirobot_infra/manus_client.log` |

初次配置：

1. 确认 Sharpa-Manus SDK 根目录存在。
2. 确认 `SharpaManusClient.out` 和 Manus SDK 动态库存在。
3. 默认由程序自动启动本机 Manus client，并监听 `127.0.0.1:2044`。
4. 如果已有外部 Manus client 在运行，保持 endpoint 一致即可。

检查命令：

```bash
ss -ltpn | grep 2044
tail -n 100 /tmp/unirobot_infra/manus_client.log
```

如果 `ss` 没有看到 `2044`，但 `client.auto_start: true`，程序启动 teleop 时会尝试自动拉起 client。

## Vive Tracker

配置文件：

- `configs/device/vive/default.yaml`
- `configs/wrappers/teleop_intervention/vive.yaml`
- `configs/wrappers/teleop_intervention/manus_vive.yaml`

当前配置：

| 项 | 值 |
| --- | --- |
| endpoint | `tcp://192.168.1.225:5555` |
| network interface | `wlp0s20f3` |
| topic | `vive_trackers` |
| 左臂 tracker id | `3A-A33H00153` |
| 右臂 tracker id | `3A-A33H00778` |

初次配置：

1. 确认 Vive tracker publisher 在 `192.168.1.225:5555` 发布 ZMQ 消息。
2. 确认本机 `wlp0s20f3` 与 publisher 在同一网络。
3. 确认 tracker id 与实际左右手安装位置一致。
4. 用 `vive_probe` 先看 tracker pose 是否有效，再进入 teleop。

检查命令：

```bash
ip -br addr show wlp0s20f3
cd /home/ubuntu/chensixiang/Codes/Universal_Robot_Infra
python -m unirobot_infra.apps.vive_probe
```

## 录制前检查清单

录制真实全状态、三相机和触觉前，按顺序检查：

1. `eno1` 有 `192.168.1.180/24`、`192.168.10.180/24`，以及当前 Sharpa tactile target 需要的本机地址。
2. `ping 192.168.1.190`、`ping 192.168.10.10`、`ping 192.168.10.20` 成功。
3. Tianji SDK、Sharpa-Manus SDK、`pyzed` 都能 import 或找到。
4. `/dev/v4l/by-path/` 下两台 Insta360 的路径与配置一致。
5. `realtime_capture.max_frames=30 display=false` 能读到三路相机。
6. `echo_robot_state` 能读取左右臂状态。
7. Sharpa 触觉不再报 `host_ip != target_ip`。

通过后再运行：

```bash
cd /home/ubuntu/chensixiang/Codes/Universal_Robot_Infra
python -m unirobot_infra.apps.teleop_record
```
