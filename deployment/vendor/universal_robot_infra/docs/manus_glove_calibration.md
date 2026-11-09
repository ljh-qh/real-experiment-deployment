# Manus 手套标定文件

本文记录 Manus 手套 `.mcal` 标定文件的生成、部署和验证流程。

## 1. 运行关系

当前仓库的 Manus 链路在 Ubuntu 上使用 MANUS SDK Integrated mode，不依赖 Windows MANUS Core：

```text
Ubuntu MANUS dongle
  -> sharpa-manus-sdk/client/SharpaManusClient.out
  -> tcp://127.0.0.1:2044
  -> ManusDevice
  -> SharpaRetargeter
```

`SharpaManusClient.out` 启动时会从当前工作目录加载 `Calibration_left.mcal` 和 `Calibration_right.mcal`。当前 MANUS Pro 的标定文件目标路径是：

```text
/home/ubuntu/chensixiang/Codes/sharpa-manus-sdk/client/calibration_profiles/manus_pro/Calibration_left.mcal
/home/ubuntu/chensixiang/Codes/sharpa-manus-sdk/client/calibration_profiles/manus_pro/Calibration_right.mcal
```

如果切回 MANUS Pro Haptic，则目标目录换成 `client/calibration_profiles/manus_pro_haptic/`。如果换手套、换使用者，或 retarget 后手指姿态明显不对，需要重新生成并替换对应 profile 下的两个文件。

## 2. Ubuntu GUI 生成 `.mcal`

标定工具在：

```text
/home/ubuntu/chensixiang/Codes/sharpa-manus-sdk/client/CalibrationGUI
```

如果还没有 `CalibrationGUI.out`，先构建：

```bash
cd /home/ubuntu/chensixiang/Codes/sharpa-manus-sdk/client/CalibrationGUI
./build.sh
```

运行标定前，确保没有旧的 Manus client 占用设备：

```bash
pgrep -af 'MANUS|Manus|SharpaManusClient|CalibrationGUI|teleop'
ss -lntp | rg ':2044|SharpaManusClient'
```

如果有旧的 `SharpaManusClient.out`，先确认 teleop 没在使用 Manus 输入，再停止它。

启动 GUI：

```bash
cd /home/ubuntu/chensixiang/Codes/sharpa-manus-sdk/client/CalibrationGUI
./CalibrationGUI.out
```

GUI 操作：

1. 选择 `Left Glove` 或 `Right Glove`。
2. 点击 `Start Calibration` 或按 `F5`。
3. 按界面提示做手势。
4. 点击 `Next Step` 或按 `F9`。
5. 标定完成后，GUI 会生成 `Calibration_left.mcal` 或 `Calibration_right.mcal`。

按上面的路径运行时，生成文件在：

```text
/home/ubuntu/chensixiang/Codes/sharpa-manus-sdk/client/CalibrationGUI/Calibration_left.mcal
/home/ubuntu/chensixiang/Codes/sharpa-manus-sdk/client/CalibrationGUI/Calibration_right.mcal
```

## 3. Windows MANUS Core 生成 `.mcal`

如果 Ubuntu GUI 标定效果不稳定，可以先在 Windows MANUS Core 中完成官方 Dashboard 标定和可视化检查，再把标定结果导出成 `.mcal`，最后拷回 Ubuntu 使用。

流程：

1. 在 Windows 机器上连接 MANUS dongle 和手套。
2. 打开 MANUS Core Dashboard，分别完成左手和右手标定。
3. 在 MANUS Core 里检查手部预览，确认手指弯曲、拇指对掌和左右手方向都正常。
4. 用 Windows 上的 MANUS SDK 导出当前 Core 中的左右手 calibration bytes，保存为：

```text
Calibration_left.mcal
Calibration_right.mcal
```

导出工具应连接 MANUS Core，获取对应 glove/user 的 calibration size，再调用：

```cpp
CoreSdk_GetGloveCalibrationSize(...)
CoreSdk_GetGloveCalibration(...)
```

或按 user/side 导出时使用：

```cpp
CoreSdk_GetGloveCalibrationSizeForUser(...)
CoreSdk_GetGloveCalibration(...)
```

当前仓库还没有提交 Windows 导出工具；如果新增这个工具，建议直接复用 MANUS SDK Client example 的保存 calibration 逻辑。

注意当前这副 Haptic 手套在 Ubuntu 日志里被 SDK 识别为 `Metaglove Glove`。导出时优先使用 `GloveProfileType_Metaglove`；如果导出的文件在 Ubuntu 加载时报：

```text
glove family does not match
```

说明 `.mcal` 的 profile family 和当前 SDK 识别到的手套 family 不一致，需要重新导出对应 family 的文件。

把 Windows 导出的文件拷回 Ubuntu 后，放到当前手套型号对应的 profile 目录。当前 MANUS Pro 目标路径是：

```text
/home/ubuntu/chensixiang/Codes/sharpa-manus-sdk/client/calibration_profiles/manus_pro/Calibration_left.mcal
/home/ubuntu/chensixiang/Codes/sharpa-manus-sdk/client/calibration_profiles/manus_pro/Calibration_right.mcal
```

然后按本文后面的部署和验证步骤确认加载成功。

## 4. GUI 显示 `Not connected`

如果手套灯正常，但 GUI 显示 `Not connected`，优先检查是否有旧进程占用了 Manus 设备或 2044 端口：

```bash
lsusb | rg -i 'manus|3325'
pgrep -af 'MANUS|Manus|SharpaManusClient|CalibrationGUI|teleop'
ss -lntp | rg ':2044|SharpaManusClient'
```

如果看到 `SharpaManusClient.out` 正在运行，先确认当前没有 teleop 或录制程序在使用 Manus 输入，再停止旧进程：

```bash
kill <PID>
```

然后从 GUI 目录重新启动标定工具：

```bash
cd /home/ubuntu/chensixiang/Codes/sharpa-manus-sdk/client/CalibrationGUI
./CalibrationGUI.out
```

如果仍然是 `Not connected`，先关闭 GUI，再单独运行 Manus client 验证硬件是否能被 SDK Integrated mode 识别：

```bash
cd /home/ubuntu/chensixiang/Codes/sharpa-manus-sdk/client/calibration_profiles/manus_pro
../../SharpaManusClient.out
```

能看到 `is connected as Metaglove Glove` 或左右手 `Frame[...] glove: ... is published`，说明硬件链路是通的。验证完成后停止 `SharpaManusClient.out`，再回到 GUI 标定。

## 5. 部署 `.mcal`

先备份当前运行文件：

```bash
cd /home/ubuntu/chensixiang/Codes/sharpa-manus-sdk/client
target_dir=calibration_profiles/manus_pro
backup_dir=calibration_backup/$(date +%Y%m%d_%H%M%S)
mkdir -p "$backup_dir"
cp "$target_dir"/Calibration_left.mcal "$target_dir"/Calibration_right.mcal "$backup_dir"/
```

再把新文件复制到目标 profile 目录。Ubuntu GUI 生成的文件在 `client/CalibrationGUI/`：

```bash
cp /home/ubuntu/chensixiang/Codes/sharpa-manus-sdk/client/CalibrationGUI/Calibration_left.mcal \
  /home/ubuntu/chensixiang/Codes/sharpa-manus-sdk/client/calibration_profiles/manus_pro/Calibration_left.mcal

cp /home/ubuntu/chensixiang/Codes/sharpa-manus-sdk/client/CalibrationGUI/Calibration_right.mcal \
  /home/ubuntu/chensixiang/Codes/sharpa-manus-sdk/client/calibration_profiles/manus_pro/Calibration_right.mcal
```

如果只重新标定了一只手，只替换对应侧的 `.mcal`。

如果使用 Windows MANUS Core 路径，把 Windows 导出的 `.mcal` 复制到同一个 profile 目录并重命名。本机当前导出的 Haptic 文件对应命令为：

```bash
cp /home/ubuntu/chensixiang/Codes/sharpa-manus-sdk/srobotLeftMetagloveProHaptics.mcal \
  /home/ubuntu/chensixiang/Codes/sharpa-manus-sdk/client/calibration_profiles/manus_pro_haptic/Calibration_left.mcal

cp /home/ubuntu/chensixiang/Codes/sharpa-manus-sdk/srobotRightMetagloveProHaptics.mcal \
  /home/ubuntu/chensixiang/Codes/sharpa-manus-sdk/client/calibration_profiles/manus_pro_haptic/Calibration_right.mcal
```

## 6. 验证

可以先直接运行 Manus client 验证加载和数据发布：

```bash
cd /home/ubuntu/chensixiang/Codes/sharpa-manus-sdk/client/calibration_profiles/manus_pro
../../SharpaManusClient.out
```

看到下面日志表示 `.mcal` 已加载：

```text
Successfully loaded glove calibration file: Calibration_left.mcal
Successfully loaded glove calibration file: Calibration_right.mcal
```

看到下面日志表示左右手套数据正在发布：

```text
Frame[...] glove: Left is published
Frame[...] glove: Right is published
```

验证完成后停止 `SharpaManusClient.out`，再运行 `unirobot_infra` 的 teleop 或录制入口。否则 `127.0.0.1:2044` 会被手动启动的 client 占用。
