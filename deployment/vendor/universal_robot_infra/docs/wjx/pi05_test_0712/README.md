# pi05 真机测试命令 - 2026-07-12

本目录准备 joint absolute 与 delta-ee 两套 pi05 checkpoint 的真机侧启动命令。今晚先只跑 Joint Absolute 真机测试。本文主命令默认就是正式真机下发：`control_joint.sh`、`control_delta_ee_step.sh`、`control_delta_ee_chunk.sh` 都会设置 `policy.execution.dry_run=false` 和 `policy.execution.confirm_real_execution=true`。

## 路径填写位置

- Joint checkpoint: 填到 `serve_joint.sh` 的环境变量 `JOINT_CKPT_DIR`。
- Delta-ee checkpoint: 填到 `serve_delta_ee.sh` 的环境变量 `DELTA_EE_CKPT_DIR`。
- OpenPI 仓库: 默认 `OPENPI_ROOT=/mnt/dataset/csx_workspace/openpi`，机器人 PC 路径不同则覆盖。
- 控制仓库: 默认 `URI_ROOT=/mnt/dataset/csx_workspace/Ideas/AF3/code/openpi/third_party/Universal_Robot_Infra`，机器人 PC 路径不同则覆盖。

今晚 Joint Absolute 本机路径：

```bash
OPENPI_ROOT=/home/ubuntu/chensixiang/Codes/openpi_cscsx
URI_ROOT=/home/ubuntu/chensixiang/Codes/Universal_Robot_Infra
JOINT_CKPT_DIR=/home/ubuntu/chensixiang/Checkpoints/20260712_tianji_pour_drink_test_without_trainstate/joint_absolute/29999
```

已检查 `JOINT_CKPT_DIR`：目录约 12G，包含 `params/`、`_CHECKPOINT_METADATA` 和 `assets/tianji_sharpa_pour_drink_joint_absolute/norm_stats.json`。

已 clone `/home/ubuntu/chensixiang/Codes/openpi_cscsx`，仓库 `CSCSX/openpi` 默认分支 `dev-csx` 包含 `pi05_tianji_sharpa_pour_drink_joint_absolute` policy config。已验证 policy server 可恢复 checkpoint、加载 norm stats，并监听 `0.0.0.0:8010`。

Joint Absolute 本机 initial action：

```bash
JOINT_INITIAL_ACTION=/home/ubuntu/chensixiang/Codes/Universal_Robot_Infra/docs/wjx/pi05_test_0712/tj0710_pour_drink_joint_initial_action_ep000_frame000.json
```

该文件来自 `data/lerobot_datasets/tj0710_pour_drink_demo_joint` 的 episode 0 frame 0 action。不要使用旧的 `ckpt/pi05_csx/dataset_initial_action_ep000_frame000.json`，它对应 `tj0627_csx_no_stactile`，和当前 pour-drink joint checkpoint 不匹配。

训练侧参考 ckpt step 路径如下，拷到机器人 PC 后用机器人 PC 上的本地路径替换：

```bash
JOINT_CKPT_DIR=/mnt/dataset/public_data/cscsx_projects/AF3/openpi/checkpoints/pi05_tianji_sharpa_pour_drink_joint_absolute/ACWM_openpi_pi05_pour_jointabs58_h50_full_8gpu_30k_retry1_20260710/29000
DELTA_EE_CKPT_DIR=/mnt/dataset/public_data/cscsx_projects/AF3/openpi/checkpoints/pi05_tianji_sharpa_pour_drink_delta_ee/ACWM_openpi_pi05_pour_deltaee20_h50_full_8gpu_30k_retry1_20260710/29000
```

## Joint Absolute 模型

Joint 模型使用 58D state / 58D absolute action，不走 delta wrapper。控制配置是 `apps/openpi_policy_control_loop/joint_absolute`，默认 `policy.action.execute_steps=32`。

终端 A 启动 policy server：

```bash
cd /home/ubuntu/chensixiang/Codes/Universal_Robot_Infra/docs/wjx/pi05_test_0712
OPENPI_ROOT=/home/ubuntu/chensixiang/Codes/openpi_cscsx \
JOINT_CKPT_DIR=/home/ubuntu/chensixiang/Checkpoints/20260712_tianji_pour_drink_test_without_trainstate/joint_absolute/29999 \
PORT=8010 \
./serve_joint.sh
```

终端 B 启动正式真机控制：

```bash
cd /home/ubuntu/chensixiang/Codes/Universal_Robot_Infra/docs/wjx/pi05_test_0712
URI_ROOT=/home/ubuntu/chensixiang/Codes/Universal_Robot_Infra \
PORT=8010 \
FRAMES=20 \
./control_joint.sh \
  env.reset.move_to_initial_state=false \
  policy.execution.initial_action.enabled=true \
  policy.execution.initial_action.path=/home/ubuntu/chensixiang/Codes/Universal_Robot_Infra/docs/wjx/pi05_test_0712/tj0710_pour_drink_joint_initial_action_ep000_frame000.json \
  policy.execution.initial_action.move_steps=100 \
  policy.execution.initial_action.max_delta=0.02
```

## Delta-ee 模型

Delta-ee 模型使用 20D policy state / 20D delta-ee action。控制侧必须开启：

- `wrappers.hand_alpha_action.enabled=true`
- `wrappers.adaptive_hand_action.enabled=true`
- `wrappers.tianji_delta_ee_action.enabled=true`
- `policy.observation.server.state.source=policy_state`
- `policy.action.max_delta=0.0`
- `policy.action.max_state_delta=0.0`
- `policy.execution.initial_action.enabled=false`

代码里现在支持两种 blocking 模式：

- Step-level blocking: 每执行 1 个 delta-ee action 后等待当前 target 到位，再进入下一帧；配置 `apps/openpi_policy_control_loop/delta_ee_step_blocking`，默认 `execute_steps=16`。
- Chunk-level blocking: 对一个 action chunk 连续积分并发送，chunk 结束时等待最终 target 到位；配置 `apps/openpi_policy_control_loop/delta_ee_chunk_blocking`，默认 `rate_hz=10.0`、`execute_steps=50`，与 10Hz 数据集和 50-step action horizon 对齐。

### 模型预测 chunk 与 execute_steps

OpenPI policy server 每次会返回一个 action chunk，形状类似 `[action_horizon, action_dim]`。控制环不会一定执行完整 chunk，而是由 `policy.action.execute_steps` 决定“每次拿到模型预测 chunk 后实际执行前多少步”：

- Joint 配置默认位置：`configs/apps/openpi_policy_control_loop/joint_absolute.yaml` 里的 `policy.action.execute_steps: 32`。
- Delta-ee step 配置默认位置：`configs/apps/openpi_policy_control_loop/delta_ee_step_blocking.yaml` 里的 `policy.action.execute_steps: 16`。
- Delta-ee chunk 配置默认位置：`configs/apps/openpi_policy_control_loop/delta_ee_chunk_blocking.yaml` 里的 `env.runtime.rate_hz: 10.0`、`policy.action.execute_steps: 50`。
- 命令行临时覆盖：在控制脚本后追加 `policy.action.execute_steps=<N>`。

例如连续运行 10Hz / 50-step chunk 真机测试：

```bash
PORT=8010 FRAMES=0 ./control_delta_ee_chunk.sh
```

终端 A 启动 delta-ee policy server：

```bash
cd /mnt/dataset/csx_workspace/Ideas/AF3/code/openpi/third_party/Universal_Robot_Infra/docs/wjx/pi05_test_0712
DELTA_EE_CKPT_DIR=/path/on/robot_pc/to/delta_ee_ckpt/29000 ./serve_delta_ee.sh
```

终端 B 跑 step-level blocking 正式真机控制：

```bash
cd /mnt/dataset/csx_workspace/Ideas/AF3/code/openpi/third_party/Universal_Robot_Infra/docs/wjx/pi05_test_0712
PORT=8010 FRAMES=20 ./control_delta_ee_step.sh
```

或跑 chunk-level blocking 正式真机控制：

```bash
cd /mnt/dataset/csx_workspace/Ideas/AF3/code/openpi/third_party/Universal_Robot_Infra/docs/wjx/pi05_test_0712
PORT=8010 FRAMES=0 ./control_delta_ee_chunk.sh
```

## Adaptive Grasp 状态

Delta-ee 控制现在默认启用轻量版 `adaptive_hand_action` adapter。模型仍输出每只手 1D `hand_alpha`，但真实下发前会由 adapter 复用 demo record 的 `AdaptiveGraspPlanner` / `AdaptiveGraspStateMachine` 核心逻辑修正手部 22D q。

当前对齐范围：

- Grasp: `alpha` 从低到高时建立 Sharpa torque baseline，按 finger group 检测 `torque_delta` / `position_lag`，触发后冻结该 group。
- Hold: `alpha` 高位或所有 group stopped 后进入 holding，按 demo 同款 hold 参数继续微闭合/微打开，并检测 object missing。
- Release: `alpha <= 0.05` 时进入轻量 release，沿 anchor trajectory 按当前 alpha 打开；`alpha <= 0.02` 后 reset 为 released。

执行链路：

```text
policy 20D delta-ee+alpha
  -> tianji_delta_ee_action: 20D 展开为 16D delta-ee target q + hand alpha
  -> adaptive_hand_action: 用 adaptive grasp 状态机修正 hand q
  -> hand_alpha_action 内层 env: 绕过线性 hand 展开后下发 58D robot action
```

`hand_alpha_action` 仍保留，因为它定义了训练/推理一致的 1D alpha action layout，以及 state/action 中 alpha 的压缩/反推逻辑；`adaptive_hand_action` 只改变真机执行语义，不改变 pi05 action schema。

## 安全检查顺序

1. 先启动 policy server，确认 checkpoint config 与模型类型匹配。
2. 确认 action dim：joint=58，delta-ee=20。
3. 确认三路相机 key 与训练一致：`observation/image_head_left`、`observation/image_left_wrist`、`observation/image_right_wrist`。
4. 确认急停、工作空间、初始位姿和夹爪状态。
5. 小步数正式真机测试：建议先 `FRAMES=20`，确认无异常后再放开。
