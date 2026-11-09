# 真机交接文档

这份文档给真机电脑上的新 Codex 会话使用。目标是让新的会话快速继承当前上下文：项目背景、已经理解的代码主线、真机上已经发生的事情、以及下一步应该怎么继续。

## 项目背景

当前线程/项目名：

```text
自进化项目：真机
```

这是同一个大项目的一部分：

```text
QWan 一体化模型
+ 自进化 pipeline
+ action following benchmark
+ 真机执行
```

当前大目标：

```text
打通真机这一关
```

当前短期目标：

```text
亲手跑一遍已有真机 pipeline，理解每个入口在验证什么，看到真实输出/效果，
并且知道失败时应该沿哪条链路排查。
```

仓库：

```text
PKU-HMI-Lab/Universal_Robot_Infra
```

真机电脑上的仓库路径：

```text
/home/ubuntu/chensixiang/Codes/Universal_Robot_Infra
```

工作分支：

```text
dev-wjx
```

不要在 `main` 分支上修改。后续所有修改都应该在 `dev-wjx` 上进行。

真机仓库里可能有一个本地未跟踪文件：

```text
auth.json
```

不要删除、提交或修改它，除非用户明确要求。

## Codex 设置记录

真机电脑一开始的 Codex 可以启动，但发消息时报错：

```text
401 Unauthorized: Invalid token
```

原因是这台机器的 Codex 配置曾经把 OpenAI provider 指到了一个中转站：

```text
https://newapi.150226.xyz/v1
```

当时 `/status` 里显示：

```text
Model provider: OpenAI - https://newapi.150226.xyz/v1
Account: 2430085618@qq.com (Pro)
```

也就是说：用户已经登录了自己的 Pro 账号，但是本机配置还在走别人留下的 NewAPI 中转站，所以 token 不匹配。

修复方式：

1. 备份并编辑：

```text
~/.codex/config.toml
```

2. 删除类似下面的自定义 provider block：

```toml
[model_providers.OpenAI]
base_url = "https://newapi.150226.xyz/v1"
requires_openai_auth = true
```

3. 将：

```toml
model_provider = "OpenAI"
```

改成：

```toml
model_provider = "openai"
```

修复后，Codex 已经能正常回复消息，并且 `/status` 不再显示 `newapi.150226.xyz`。

目前仍然可能有这个警告：

```text
MCP client for `node_repl` failed to start
```

这个暂时不影响当前真机实操，可以先忽略。

## 代码仓库理解地图

我们已经带用户过了一轮代码主线。不要重新从零讲起，可以从下面的地图继续。

### 阶段 1：项目地图和入口

顶层目录含义：

```text
src       正式源代码
configs   运行配置
docs      作者写的说明文档
tests     测试代码
scripts   辅助脚本
models    模型权重目录
assets    机器人资源/描述文件
```

重要入口目录：

```text
src/unirobot_infra/apps
```

`apps` 可以理解成命令行入口，也就是“按钮面板”：

```text
teleop.py                         实时遥操作
teleop_record.py                  遥操作 + 数据录制
replay_lerobot.py                 将 LeRobot action 回放到真机
robot_ops/echo_robot_state.py     读取 Tianji 机器人状态
robot_ops/move_tianji_tcp_pose.py 移动 Tianji TCP pose
robot_ops/sharpa_adaptive_grasp.py 单独测试 Sharpa 自适应抓握
perception/perception_loop.py     感知调试 loop
demo/left_grasp_cup.py            左臂抓杯 demo
demo/right_grasp_bottle.py        右臂抓瓶 demo
demo/dual_grasp_cup_bottle.py     双臂 demo
postprocess/convert_lerobot.py    raw episode 转 LeRobot
```

`apps` 入口层应该保持比较薄，主要负责：

```text
读取配置
初始化日志
构建 env/runtime
启动主循环
退出时关闭资源
```

真正复杂逻辑在这些模块里：

```text
envs
wrappers
controllers
robots
perception
grasp
recorder
dataset
```

### 阶段 2：真机控制主链路

核心链路：

```text
flat action
-> RobotActionLayout
-> RobotAction
-> env.step
-> RobotGroupController
-> TianjiRobot / SharpaHand
-> 厂商 SDK
-> 真机动作
```

关键概念：

```text
action                  发给机器人的动作命令
flat action             把所有机器人部件的动作拼成一个长向量
arm_q_rad               Tianji 机械臂关节角，单位 rad
end_effector_q_rad      Sharpa 灵巧手关节角，单位 rad
RobotActionLayout       负责 flat action 和 RobotAction 的互相转换
env.step(action)        一帧控制
controller              动作分发员
robots/                 对厂商 SDK 的项目内封装
SDK                     Software Development Kit，厂商软件开发工具包
```

默认双臂双手配置下：

```text
left arm        7
left hand       22
right arm       7
right hand      22
total           58
```

58 维 flat action 的默认切分：

```text
0-6      left arm
7-28     left end_effector / left hand
29-35    right arm
36-57    right end_effector / right hand
```

重要文件：

```text
src/unirobot_infra/envs/layout.py
src/unirobot_infra/envs/tianji_sharpa_env.py
src/unirobot_infra/controllers/robot_group.py
src/unirobot_infra/robots/arm/tianji.py
src/unirobot_infra/robots/end_effector/sharpa.py
```

### 阶段 3：遥操作 pipeline

遥操作 teleop 的意思：

```text
人示范，机器人跟随
```

完整 pipeline：

```text
人
-> Manus / Vive / Keyboard / Joint GUI
-> teleop source
-> mapping / retarget / IK
-> RobotAction
-> TeleopInterventionWrapper
-> env.step
-> controller
-> robots/SDK
-> 真机
```

设备含义：

```text
Manus       手套/手部动捕，通常控制 Sharpa 手
Vive        3D tracker，通常控制 Tianji 机械臂 TCP
Keyboard    调试用的小步 jog 控制
Joint GUI   用网页滑条直接调关节，适合调试
```

Manus 到 Sharpa：

```text
Manus keypoints
-> SharpaRetargeter
-> end_effector_q_rad(22)
-> RobotAction
```

Vive 到 Tianji：

```text
Vive tracker pose
-> TCP target pose
-> IK
-> arm_q_rad(7)
-> RobotAction
```

相关概念：

```text
TCP = Tool Center Point，工具中心点/末端工具控制点
IK  = Inverse Kinematics，逆运动学，目标 TCP pose -> 关节角
deadman trigger = 安全门，按住才让 Vive 控机械臂，松开就停
```

`TeleopInterventionWrapper` 是 teleop 进入主链路的地方。它读取 Manus/Vive/Keyboard/Joint GUI 产生的 action，并只覆盖 flat action 里对应的 segment。

### 阶段 4：感知 + 抓取 demo

感知抓取链路：

```text
RGB 图像 + Depth 深度
-> detector 找 cup/bottle 的 2D 区域
-> depth projection / localization 得到 3D 点
-> frame transform 从 camera frame 转到 robot base/world frame
-> 加 offset + 设置 TCP orientation 得到抓取目标 TCP pose
-> servo loop 小步闭环靠近目标
-> adaptive grasp 自适应抓握
-> lift 抬升
```

关键概念：

```text
perception        从传感器理解环境
camera frame      相机坐标系
robot/world frame 机器人/世界坐标系
calibration       标定，求相机坐标系和机器人坐标系的关系
T_cam2base.npy    保存 camera 到 base 的转换矩阵
offset            检测到的物体中心不一定是 TCP 抓取目标点
TCP pose          TCP 的位置 + 朝向
servo             高频、小步、闭环地跟踪目标
```

Servo 的逻辑：

```text
读当前 TCP pose
看目标 TCP pose
朝目标走一小步
IK 解这一小步
发送 non-blocking command
下一帧继续读当前状态并更新
直到误差足够小
```

Adaptive grasp 的逻辑：

```text
准备 anchor 手势轨迹
alpha 从 0 到 1 插值闭合
每帧读取 Sharpa feedback
某个手指 group 的 torque_delta 或 position_lag 超阈值就冻结
其他手指 group 继续闭合
```

关键概念：

```text
anchor            预先录好的手部姿态样本
stop group        通常按 thumb/index/middle/ring/pinky 分组
torque_delta      相对初始力矩的变化量
position_lag      命令让手指动，但实际关节跟不上
```

重要文档：

```text
docs/perception_adaptive_grasp_demo.md
docs/calibration.md
docs/camera_debug.md
```

### 阶段 5：数据记录、LeRobot 转换、Replay

数据闭环：

```text
teleop_record
-> raw episode
-> convert_lerobot
-> LeRobot dataset
-> replay_lerobot
-> 真机
```

`teleop_record` 主要录：

```text
robot state      机器人真实反馈，比如关节位置/速度/力矩
action           policy_action 和 sent_action
camera           head/wrist 图像
tactile          Sharpa 触觉图像/力信息/metadata
```

两个 action 概念：

```text
policy_action    上层原本传进来的 action
sent_action      经过 teleop wrapper 覆盖后，真正发给机器人的 action
```

做 imitation data 时，通常更关心 `sent_action`。

raw episode 通常类似：

```text
data/episodes/episode_001/
  robot_states.h5
  camera videos
  tactile videos / metadata
```

为什么要转 LeRobot：

```text
raw streams 频率不同
训练/回放需要统一 fps 的整齐帧
convert_lerobot 会把多频率 raw streams 重采样到 target_fps
```

Replay 的关键约束：

```text
LeRobot action 维度必须等于当前 RobotActionLayout size
replay 频率决定每帧 action 的间隔
delta ee action 对频率尤其敏感
```

重要文档：

```text
docs/convert_dataset.md
docs/recording_freq.md
```

## 当前真机进度

已经完成：

```text
1. 确认真机电脑 repo 路径。
2. 从 main 切到了 dev-wjx。
3. 确认 Python 3.12.13。
4. 确认 import unirobot_infra 正常。
5. 修复 Codex CLI provider/token 问题，移除了 NewAPI 中转站配置。
```

真机上曾看到的 git 状态：

```text
## dev-wjx
?? auth.json
```

不要动 `auth.json`。

尝试过的命令：

```bash
python -m unirobot_infra.apps.robot_ops.echo_robot_state
```

失败信息：

```text
port bind failure, possibly occupied by another program
RuntimeError: Failed to connect Tianji robot at 192.168.1.190
```

网络检查：

```text
ping 192.168.1.190 成功
0% packet loss
```

本机 IP：

```text
eno1 有 192.168.1.180/24
Tianji robot 是 192.168.1.190
```

端口检查发现：

```text
ToDesk_Se PID 1765 root TCP *:20000 (LISTEN)
```

结论：

```text
ToDesk 很可能占用了 Tianji/Marvin SDK 需要的 20000 端口。
只要 ToDesk 还在占用 20000，Tianji SDK 连接可能失败。
```

重要操作提醒：

```text
如果 ToDesk 是用户唯一的远程控制方式，不要直接停止 ToDesk，
否则用户会失去对真机电脑的控制。
应该改成本地物理操作、SSH、VS Code Remote，或者其他不占 20000 端口的连接方式。
```

## 下一步真机实操顺序

### Step 0：不用 ToDesk 或释放 ToDesk 占用的端口

目标：

```text
释放 20000 端口，但不要丢失对真机电脑的控制。
```

如果已经在真机电脑本地操作，或者通过 SSH / VS Code Remote 连接，可以停 ToDesk：

```bash
sudo systemctl stop todeskd
```

如果服务名不确定：

```bash
systemctl list-units | grep -i todesk
```

确认 20000 是否释放：

```bash
ss -lntp | grep 20000
```

期望结果：

```text
没有输出
```

如果 20000 仍被占用：

```bash
sudo lsof -iTCP:20000 -sTCP:LISTEN -n -P
```

不要随便 kill 进程。先判断是不是远程控制软件或重要服务。

### Step 1：环境确认

目标：

```text
确认 repo、分支、Python 环境正常。
```

命令：

```bash
cd /home/ubuntu/chensixiang/Codes/Universal_Robot_Infra
git status --short --branch
python --version
python -c "import unirobot_infra; print('unirobot_infra import ok')"
```

期望：

```text
## dev-wjx
Python 3.12.x
unirobot_infra import ok
```

### Step 2：Tianji 只读连通性

目标：

```text
只读取 Tianji 状态，不主动发运动命令。
```

运行前确认：

```text
急停可用
机器人周围没人
机器人电源/控制柜/网络正常
```

命令：

```bash
python -m unirobot_infra.apps.robot_ops.echo_robot_state
```

期望看到：

```text
left/right arm joint positions
cur_state/cmd_state/err_code
```

如果失败，排查：

```bash
ping -c 3 192.168.1.190
ss -lntp | grep 20000
sudo lsof -iTCP:20000 -sTCP:LISTEN -n -P
tail -n 80 logs/<latest_echo_robot_state_dir>/tianji_marvin.log
```

### Step 3：传感器/输入设备连通性

目标：

```text
在移动机器人之前，先确认相机和 Vive。
```

命令：

```bash
python -m unirobot_infra.apps.realtime_capture
python -m unirobot_infra.apps.vive_probe
```

期望：

```text
相机能出画面或 frame log
Vive tracker 有数据且延迟正常
```

相机问题看：

```text
docs/camera_debug.md
```

### Step 4：小动作 dry-run

目标：

```text
先验证配置、IK、轨迹逻辑，不让机器人真实动。
```

命令：

```bash
python -m unirobot_infra.apps.robot_ops.sharpa_adaptive_grasp adaptive_grasp.dry_run=true
python -m unirobot_infra.apps.robot_ops.move_tianji_tcp_pose tcp_pose_move.dry_run=true
```

期望：

```text
输出 dry-run 的目标 q / max joint 等信息，不发生真实运动。
```

### Step 5：小动作真机探针

目标：

```text
在只读和 dry-run 都通过后，做极小范围真机动作。
```

安全要求：

```text
操作人在机器旁
急停可用
工作空间清空
从最小动作开始
不要一上来跑完整 demo
```

可能命令：

```bash
python -m unirobot_infra.apps.robot_ops.sharpa_adaptive_grasp adaptive_grasp.feedback.enabled=false
python -m unirobot_infra.apps.robot_ops.move_tianji_tcp_pose
```

优先用 Hydra override 把动作调小、调慢。

### Step 6：perception loop

目标：

```text
确认检测和 3D 定位稳定，再跑抓取 demo。
```

命令：

```bash
python -m unirobot_infra.apps.perception.perception_loop
```

检查：

```text
target label 是否正确
2D detection 是否稳定
3D point 是否跳动
camera/depth 是否正常
```

相关文档：

```text
docs/perception_adaptive_grasp_demo.md
```

### Step 7：单侧 demo

目标：

```text
只跑一个最小单侧抓取 demo。
```

命令：

```bash
python -m unirobot_infra.apps.demo.left_grasp_cup
```

或者：

```bash
python -m unirobot_infra.apps.demo.right_grasp_bottle
```

期望阶段：

```text
live perception target update
servo to target
adaptive grasp
lift
```

如果靠近太快，可以减小每步平移：

```bash
python -m unirobot_infra.apps.demo.left_grasp_cup demo.servo.max_translation_step_m=0.008
```

如果 offset 不对，用这个工具调：

```bash
python -m unirobot_infra.apps.robot_ops.move_tianji_tcp_pose \
  tcp_pose_move.position_offset_m='[-0.04,0.02,-0.08]'
```

调出好值以后，再同步到对应 demo 配置。

### Step 8：数据闭环，如果时间允许

目标：

```text
录一条很小的 teleop episode，并转成 LeRobot。
```

命令：

```bash
python -m unirobot_infra.apps.teleop_record app.record_control.num_episodes=1
python -m unirobot_infra.apps.postprocess.convert_lerobot app.inspect_timestamps=true
```

Replay 放最后。Replay 前必须确认：

```text
action dim 等于当前 RobotActionLayout size
初始姿态安全
replay 频率是有意设置的
操作人准备好随时急停
```

命令形态：

```bash
python -m unirobot_infra.apps.replay_lerobot \
  app.dataset.repo_id=local/dataset \
  app.dataset.root=data/lerobot_dataset/lerobot_dataset_test \
  app.dataset.episode_index=0
```

## 安全规则

始终遵守：

```text
1. 先读状态，再移动。
2. 能 dry-run 就先 dry-run。
3. 先单侧，不要一上来双臂。
4. 急停必须可用。
5. 工作空间必须清空。
6. 未知 action 维度的数据不要 replay。
7. perception 和 offset 没确认前不要跑完整 demo。
8. 如果 ToDesk 是唯一控制方式，不要直接停 ToDesk。
```

## 常见问题排查地图

### Python/import 失败

检查：

```bash
python --version
python -c "import unirobot_infra; print('ok')"
```

期望 Python：

```text
3.12.x
```

### Tianji 连接失败

检查：

```bash
ping -c 3 192.168.1.190
ss -lntp | grep 20000
sudo lsof -iTCP:20000 -sTCP:LISTEN -n -P
tail -n 80 logs/<latest>/tianji_marvin.log
```

已知问题：

```text
ToDesk 可能占用 TCP 20000。
```

### action 维度不匹配

常见原因：

```text
LeRobot dataset 的 action 维度和当前 robot config 不一致。
```

检查：

```text
configs/apps/replay/default.yaml
configs/robot/*.yaml
src/unirobot_infra/envs/layout.py
```

### 感知目标不对或不稳定

先跑：

```bash
python -m unirobot_infra.apps.perception.perception_loop
```

再检查：

```text
camera image
target label/classes
depth range
calibration path
3D point stability
```

相关文档：

```text
docs/perception_adaptive_grasp_demo.md
docs/calibration.md
docs/camera_debug.md
```

### adaptive grasp 过早停止

大概率阈值太低。

相关配置：

```text
adaptive_grasp.feedback.torque_delta_threshold_by_group
adaptive_grasp.feedback.position_lag_threshold_rad_by_group
adaptive_grasp.feedback.min_alpha_for_stop
```

可先关反馈验证轨迹本身：

```bash
python -m unirobot_infra.apps.robot_ops.sharpa_adaptive_grasp \
  adaptive_grasp.feedback.enabled=false
```

### 松手/释放不好

相关配置：

```text
adaptive_grasp.release.stage_duration_s
adaptive_grasp.release.rate_hz
adaptive_grasp.release.stages
```

当前 release 逻辑是分 group 分阶段打开，不是所有手指一次性全松。

## 给真机 Codex 的开场 prompt

可以在真机电脑的新 Codex 会话里直接发：

```text
我们现在在 Universal_Robot_Infra 真机电脑上，分支 dev-wjx。
请先阅读 REAL_MACHINE_HANDOFF.md，然后继续带我做真机实操。
当前目标是先解决 ToDesk 占用 20000 端口导致 Tianji SDK 连接失败的问题，
然后按文档顺序跑 echo_robot_state、相机/Vive探针、dry-run、小动作、
perception_loop、单侧抓取 demo。每一步先说明目标，再给命令，
我贴输出后你帮我判断下一步。
```

## 当前最高优先级

下一件具体事：

```text
停止或绕开 ToDesk，让 20000 端口释放，然后重跑 echo_robot_state。
```

在 `echo_robot_state` 成功之前，不要进入任何真机运动步骤。
