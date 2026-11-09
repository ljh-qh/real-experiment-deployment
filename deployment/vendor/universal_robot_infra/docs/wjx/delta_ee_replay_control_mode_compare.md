# Delta-EE Replay 与 tj_fx Teleop 控制方式对比

本文只做只读分析，不涉及代码修改。目标是对比当前 `Universal_Robot_Infra` 的 delta-ee replay / policy action chunk 控制方式，和 `/home/ubuntu/chensixiang/Codes/tj_fx_teleop_middleware_20260522` 里的实时 teleop middleware 控制方式，找出可能导致 delta-ee replay 真机效果不稳定的实现差异。

注意：`tj_fx_teleop_middleware_20260522` 当前工作树里很多 `src/teleop/...` 文件是 0 字节，分析使用的是该仓库 git HEAD `7907d8c` 的内容。

## 关注的两个 replay 命令

`docs/wjx/delta_ee_validate.md` 最后两个命令分别是：

1. 每个 step 阻塞等待到位：

```text
env.runtime.action_blocking=true
app.replay.chunk.enabled=false
```

2. 按 chunk 发送：

```text
env.runtime.action_blocking=false
app.replay.chunk.enabled=true
app.replay.chunk.size=16
```

这两个命令不是底层 SDK 控制方式的简单开关，而是两种 replay 层控制策略。

## 当前 delta-ee replay 的实际控制链路

入口：

```text
src/unirobot_infra/apps/replay_lerobot.py
```

核心 loop：

```text
src/unirobot_infra/replay/loop.py
```

delta-ee action 展开：

```text
src/unirobot_infra/controllers/delta_ee.py
src/unirobot_infra/wrappers/tianji_delta_ee_action.py
```

真机 env 发送：

```text
src/unirobot_infra/envs/tianji_sharpa_env.py
src/unirobot_infra/robots/arm/tianji.py
```

### 每个 step 阻塞模式

`app.replay.chunk.enabled=false` 时，`replay_actions()` 对 action 逐帧循环：

```text
for frame_index in range(start_frame, stop_frame):
    action = actions[frame_index]
    env.step(action) 或 delta_ee_controller.send_action(action)
    sleep_remaining(period_s)
```

如果启用了 trajectory log，会走 `DeltaEeCommandController.send_action()`；否则由 `TianjiDeltaEeActionWrapper.step()` 展开。两者的核心语义相同：把单帧 delta-ee action 展开成底层关节 action，然后调用真机 env。

关键点在 `TianjiSharpaEnv.step()`：

```text
self.controller.send_actions(actions, blocking=False)
if action_blocking:
    states = self._wait_until_arm_actions_reached(actions, ...)
else:
    states = self.controller.read_state()
```

也就是说，即使配置里 `env.runtime.action_blocking=true`，底层 SDK command 仍然是 `blocking=False` 发送。所谓 blocking 是 env 层发完命令后轮询反馈，直到 arm target 到位，才返回本 step。

因此这个模式可以称为：

```text
每帧非阻塞发送 SDK command + 每帧反馈轮询等待到位
```

而不是：

```text
SDK 内部同步阻塞轨迹执行
```

### chunk 模式

`app.replay.chunk.enabled=true` 时，`loop.py` 要求：

```text
env.runtime.action_blocking=false
```

否则直接报错。chunk 内部逻辑是：

```text
begin_chunk()
for frame in chunk:
    send_chunk_action(action)
    sleep_remaining(period_s)
wait_until_current_target_reached(last_target)
end_chunk()
```

这说明 chunk 不是一次性发送给底层机器人。chunk 内仍然是逐帧处理 action，只是每一帧不等待到位，按 replay 频率非阻塞发送。chunk 末尾才等待最后一帧 target 到位。

`DeltaEeCommandController.send_chunk_action()` 还有一个重要语义：

```text
current_q_rad_by_side = previous command target q
target = current_q_rad_by_side + delta-ee action
update current_q_rad_by_side = target
```

也就是说，chunk 内 delta 积分基准不是滞后的真机反馈，而是上一帧 command target。chunk 结束后，再用真机反馈状态作为下一 chunk 起点。

这个设计是为了避免之前已经记录过的问题：非阻塞 replay 时，如果每帧都拿真机当前反馈当 delta 基准，机械臂尚未追到上一帧 target，下一帧 delta 会从滞后位置开始积分，轨迹会被压缩，尤其表现为下降高度不够。这个问题在 `docs/wjx/delta_ee_real_replay_debug_summary.md` 里已经记录过。

## 当前底层 Tianji SDK 发送方式

`TianjiRobot.send_joint_positions_deg()` 在 `blocking=False` 时做的是：

```text
clear_set()
set_joint_cmd_pose(...)
send_cmd()
return
```

没有等待 command 被真机执行到位。当前 `TianjiSharpaEnv.step()` 总是以 `blocking=False` 调它。

另外，当前 `RobotGroupController.send_actions()` 对同一台 Tianji robot 的双臂 command 会合并成一次 `send_joint_positions_rad(commands_by_arm)`，再由底层一次 `send_cmd()` 发出。这一点和 `tj_fx` middleware 不同，见后文。

## 当前 policy action chunk 控制方式

入口：

```text
src/unirobot_infra/apps/openpi_policy_control_loop.py
```

policy server 返回的 action 是 `[H, D]` chunk。当前逻辑不是把整个 chunk 一次性交给机器人，而是：

```text
action_buffer.replace(policy_action_chunk)
每个 control tick:
    action = action_buffer.pop_next()
    env.step(action)
```

`ActionChunkBuffer` 只保留最多 `policy.action.execute_steps` 个 action，然后逐个 pop。

因此模型 action chunk 控制 robot 的方式是：

```text
模型一次预测 chunk，本地逐 step 执行，每 step 一次 env.step()
```

不是：

```text
整段 chunk 作为轨迹一次发给底层控制器
```

还有一个配置层面的重要点：当前 `configs/apps/openpi_policy_control_loop/default.yaml` 默认没有加载 `wrappers/tianji_delta_ee_action`，默认 action layout 是 `dual_tianji_sharpa` 的底层关节/手 action，不是 delta-ee action。如果实际要让模型输出 delta-ee action，需要额外 config/override 启用 `tianji_delta_ee_action`，否则“模型 delta-ee action chunk”这个假设在默认配置下不成立。

## tj_fx middleware 的控制方式

对比对象：

```text
/home/ubuntu/chensixiang/Codes/tj_fx_teleop_middleware_20260522
```

由于当前工作树核心文件为空，以下基于 git HEAD `7907d8c`。

核心链路：

```text
PICO / TeleopFrame
-> coordinate transform
-> safety gate
-> TargetBuffer
-> FixedRateCommandScheduler
-> RobotCommandAdapter
-> IK
-> send_joint_command()
```

### 固定频率 scheduler

`FullTeleopApp.run()` 使用 scheduler 的 period 跑固定周期循环：

```text
period_s = self.scheduler.period_s()
while running:
    self.step_once(now_ns)
    sleep until next tick
```

`FixedRateCommandScheduler.step()` 每个 tick 从 target buffer 取最新 target：

```text
safe_target = target_buffer.get_latest(now)
if target stale:
    no command
else:
    command_target = latest target
    limiter.limit(command_target, dt_s)
```

这和 replay 的区别很大：middleware 不是按 dataset frame 一个一个阻塞回放，也没有 chunk 边界。它是固定频率实时 servo loop。

### 每 tick 立即发一个 command

`FullTeleopApp.step_once()` 每 tick 做：

```text
command_target, diagnostics = scheduler.step(curr_ns)
command_adapter.send_command(command_target, now_ns=curr_ns)
```

`RobotCommandAdapter.send_command()` 对左右臂分别：

1. 根据当前 command target 做 IK。
2. 根据上一条成功发送的 q 和发送时间做 joint step / joint velocity 限制。
3. 调 `send_joint_command()`。

`send_joint_command()` 只做：

```text
clear_set()
set_joint_cmd_pose(arm, joints)
send_cmd()
```

它不等待真机到位，也不在每 tick 后轮询到达目标。

所以 `tj_fx` middleware 的真实控制模式是：

```text
固定频率逐 tick 非阻塞发送 command
```

### tj_fx 的限幅比当前 replay 更保守

`tj_fx` 的默认命令限制：

```text
joint_limit_mode: reject
max_joint_step_deg: 5.0
max_joint_velocity_deg_s: 90.0
ctrl_hz: 100
```

实际每 tick 的允许关节步长：

```text
allowed_step_deg = min(max_joint_step_deg, max_joint_velocity_deg_s * dt_s)
```

如果 100 Hz，则 `dt_s=0.01`，速度限制给出的每 tick 上限约 `0.9 deg`。也就是说它默认非常保守，除非切到 ramp 模式，否则大跳变直接 reject。

当前 delta-ee replay 的 `tianji_delta_ee_action` 默认是：

```text
ik.max_joint_step_deg: 25.0
ik.joint_limit_mode: ramp
```

`TianjiTcpPoseMapper` 只按 `max_joint_step_deg` 做单步限制，没有像 `tj_fx` 一样叠加 `max_joint_velocity_deg_s * dt_s`。这意味着 replay 每帧允许的 joint-space 变化可能远大于 middleware 的实时控制 loop。

这不是一定错误，但它是一个很明确的差异：middleware 通过高频小步持续追踪目标，当前 replay 可能以较低 replay Hz 发送更大的关节目标，依赖机器人自己追。

### 左右臂发送同步方式也不同

当前 `Universal_Robot_Infra` 会把同一台 Tianji robot 的左右臂 command 合并后一次 `send_cmd()` 发出。

`tj_fx` middleware 的 `RobotCommandAdapter` 是左右臂分别 `_process_side()`，每个 side 内部各自 `send_joint_command()`，因此是两次 `send_cmd()` 顺序发送。

这个差异不一定是问题。当前合并发送反而更接近双臂同步。但如果真机 SDK 对双臂同包发送和单臂连续发送的内部处理不同，值得后续单独确认。

## 对比后得到的怀疑点

### 怀疑点 1：当前 replay 不是 middleware 那种高频 servo loop

`tj_fx` 是固定频率 command scheduler，默认 100 Hz 或 50 Hz。每 tick 都只发当前一个小步 command，不等待到位。

当前 delta-ee replay 的两个模式都不是完全同一种行为：

- 每 step 阻塞模式：每帧都等到位，破坏了原始时间节奏。
- chunk 模式：chunk 内按 replay Hz 逐帧发，chunk 末等最后 target，到 chunk 边界会重新用真机反馈作为基准。

如果 replay dataset fps 是 10 Hz，chunk 内实际发送频率远低于 `tj_fx` 的 100 Hz。这样每个 action 的 delta 更像“大步目标”，不是高频小步 servo。这可能导致实际路径、接触时机、横向偏差和 middleware 的表现不一致。

### 怀疑点 2：every-step blocking 能修高度，但不代表 replay 方式正确

每 step 阻塞解决了“反馈滞后导致 delta 积分压缩”的问题，因为下一帧开始前真机已经接近上一帧 target。

但它也带来另一个问题：真实执行时间不再等于 dataset 时间。只要某帧等待时间超过 replay period，后续动作就整体变慢。对于抓取、接触、物体交互，这可能改变力学效果。

所以 every-step blocking 更像诊断工具，适合验证几何目标是否能到，但不一定适合作为最终 policy/replay 控制方式。

### 怀疑点 3：chunk replay 语义接近实时控制，但仍缺少 tj_fx 的高频限速层

chunk replay 避免了每帧反馈滞后积分问题：chunk 内使用上一帧 command target 作为 delta 基准。

但它仍然按 replay action 频率发送 action，而不是像 `tj_fx` 一样在 100 Hz scheduler 中持续生成小步 target。当前 IK step limit 是 `25 deg + ramp`，而 `tj_fx` 默认是 `5 deg + 90 deg/s`，实际 100 Hz 下约 `0.9 deg/tick`。

因此，如果某些 delta-ee action 在 joint space 映射后跳变较大，当前 replay 可能会产生更激进的中间 command。机器人跟不上时，chunk 内 actual TCP 会滞后；chunk 末虽然等最后 target，但中间接触过程已经不一样。

### 怀疑点 4：默认 OpenPI policy loop 不是 delta-ee

如果目前认为“模型 delta-ee action chunk 控制 robot”是通过 `openpi_policy_control_loop` 完成，需要确认实际运行命令是否启用了：

```text
wrappers.tianji_delta_ee_action.enabled=true
```

当前默认 config 没有加载该 wrapper，所以默认模型 action dim 是底层关节/手 layout，而不是每臂 9 维 delta-ee + hand alpha。

如果模型 checkpoint 输出的是 delta-ee action，但控制 loop 没启用 delta-ee wrapper，会直接出现 action dim 或语义不匹配。

### 怀疑点 5：tj_fx 没有“整 chunk 发给机器人”的实现

`tj_fx` middleware 没有找到把 action horizon / chunk 一次性交给 robot 底层执行的实现。它是实时单 tick 发 command。

所以如果我们想“参考 tj_fx”，重点不应该是实现整 chunk 下发，而应该是参考它的：

```text
固定频率 scheduler
target buffer / zero-order hold
每 tick 小步 IK
per-arm joint velocity limit
非阻塞发送
```

当前 `Universal_Robot_Infra` 的 replay chunk 更像“按 dataset 帧率做 command-target 积分 + chunk 末等待”，不是同一个控制模型。

## 明天建议翔哥重点看什么

1. 先确认目标控制模式：
   - 是要做诊断 replay，到位优先，可以继续用 every-step blocking。
   - 还是要复现真实执行时序，应更接近 `tj_fx` 的 fixed-rate non-blocking servo loop。

2. 如果要接近 `tj_fx`，考虑把 delta-ee replay action 转成更高频的小步 command：
   - dataset delta-ee action 仍逐帧读。
   - replay 控制层在帧间插值或高频 hold target。
   - 以固定控制频率发送小步 IK target。
   - 使用类似 `max_joint_velocity_deg_s * dt_s` 的限制，而不是只靠 `max_joint_step_deg=25`。

3. 对 chunk replay，重点检查：
   - chunk 内 actual TCP 滞后是否过大。
   - `chunk_end_tcp_compare.csv` 是否到位。
   - 如果 chunk end 到位但过程偏差大，问题更可能是控制频率/限速/中间路径，而不是最终 IK target。

4. 对 OpenPI 模型控制，先确认实际命令是否启用了 delta-ee wrapper。
   - 如果没有，先不要把它当作“模型 delta-ee chunk 控制”分析。
   - 如果启用了，再看它是否每 tick `env.step()`，以及是否需要引入类似 `tj_fx` 的高频小步限速。

5. 对比实验建议：
   - 同一段 episode，分别跑 every-step blocking 和 chunk replay。
   - 对比 `actual_tcp_trajectory_compare.csv` 与 `chunk_end_tcp_compare.csv`。
   - 如果 every-step 几何上能到，chunk end 也能到，但 chunk 内轨迹明显滞后，说明 replay 主要问题是控制时序/频率/限速，不是 delta-ee 数据本身。

