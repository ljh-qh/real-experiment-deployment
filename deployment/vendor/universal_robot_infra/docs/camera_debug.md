# 相机排障速查

本文只放几个短命令，目标是快速判断：

1. `video` 设备有没有被占用
2. 怎么把占用杀干净
3. 怎么确认当前相机对应的设备路径和 USB 口

## 1. 判断 `video` 是否被占用

先看常用设备：

```bash
fuser -v /dev/video0 /dev/video1 /dev/video2 /dev/video4
```

如果某个设备被占用，会看到类似输出：

```text
/dev/video2: ubuntu 123456 F...m python
```

然后看这个 PID 到底是什么：

```bash
ps -fp 123456
```

如果想看所有相关进程：

```bash
ps -eo pid,ppid,stat,cmd | rg 'teleop_record|realtime_capture|ffmpeg|video'
```

## 2. 杀干净占用

优先杀应用主进程：

```bash
pkill -f '^python -m unirobot_infra.apps.teleop_record '
pkill -f '^python -m unirobot_infra.apps.realtime_capture '
```

如果还有残留的相机采集子进程，再杀 `ffmpeg`：

```bash
pkill -f '^ffmpeg -hide_banner -loglevel error -fflags nobuffer -f v4l2'
```

如果还有残留，就直接按 PID 杀：

```bash
kill -9 123456
```

杀完立刻复查：

```bash
fuser -v /dev/video0 /dev/video1 /dev/video2 /dev/video4
```

没有任何输出，说明设备已经释放。

## 3. 诊断相机端口和设备路径

先列出 V4L2 设备：

```bash
v4l2-ctl --list-devices
```

再看稳定路径：

```bash
ls -l /dev/v4l/by-path
```

看某个稳定路径最终指向哪个 `/dev/videoN`：

```bash
readlink -f /dev/v4l/by-path/pci-0000:0b:00.0-usb-0:2:1.0-video-index0
```

看某个 `/dev/videoN` 的设备名：

```bash
cat /sys/class/video4linux/video2/name
```

看某个 `/dev/videoN` 对应的 USB 路径：

```bash
readlink -f /sys/class/video4linux/video2/device
```

本仓库里 `usb_port` 配置要和这个路径里的 USB 段一致。例如：

```text
.../usb11/11-2/11-2:1.0/...
```

这里对应：

```yaml
usb_port: 11-2
```

## 4. 不阻塞的快速冒烟测试

不要一上来就跑完整程序。先用 `timeout` 做 1 帧抓图：

```bash
timeout 5s ffmpeg -hide_banner -loglevel error \
  -f v4l2 \
  -input_format mjpeg \
  -video_size 1280x720 \
  -framerate 30 \
  -i /dev/video2 \
  -frames:v 1 \
  /tmp/insta_test.jpg -y
```

如果这条命令能快速结束并生成 `/tmp/insta_test.jpg`，说明：

1. 设备存在
2. 设备没被占用
3. 相机本身能正常出流

## 5. 推荐排障顺序

建议固定按这个顺序查：

1. `fuser -v /dev/video0 /dev/video1 /dev/video2 /dev/video4`
2. 如果被占用，先 `pkill` / `kill -9`
3. `v4l2-ctl --list-devices`
4. `ls -l /dev/v4l/by-path`
5. `timeout 5s ffmpeg ... -frames:v 1 ...`
6. 最后再跑 `realtime_capture` 或 `teleop_record`

## 6. 当前这台机器最近一次确认结果

2026-06-22 这次排查时，设备对应关系是：

```yaml
insta_left:
  device_path: /dev/v4l/by-path/pci-0000:0b:00.0-usb-0:2:1.0-video-index0
  usb_port: 11-2

insta_right:
  device_path: /dev/v4l/by-path/pci-0000:08:00.0-usb-0:2:1.0-video-index0
  usb_port: 5-2
```

如果你重新插拔 USB，这个映射可能会变，需要重新确认。
