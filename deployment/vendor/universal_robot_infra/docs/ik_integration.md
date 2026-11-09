# SDK 闭式 IK 与 MuJoCo 差分 IK 互补集成

日期：2026-10-27。分支：`dev/tianji`。承 `docs/tianji_ik_diff.md` 的定性判断与
`docs/ik_example_comparison.md` 的量化实验结论，本文记录落到遥操主链的三个互补功能。
遥操步进仍走 MuJoCo 差分 IK（连续流平滑，E6）；SDK 闭式 IK 只做"低频、绝对、需要
显式失败语义"的三件事。

## 路径与配置解析

- SDK 仓库（开源 [TJ_FX_ROBOT_CONTRL_SDK](https://github.com/cynthia-you/TJ_FX_ROBOT_CONTRL_SDK) master）：
  `TJ_FX_SDK_DIR` > `MARVIN_SDK_DIR` > `./vendor`，仓库内须存在 `SDK_PYTHON/fx_kine.py`
  （ctypes 加载同目录 `libKine.so`，x86-64 纯计算库）。
- 机型配置 `.MvKDCfg`：`--kine-config` > `MARVIN_KINE_CFG` > `$SDK_DIR/CommonConfig/ccs_m6_40.MvKDCfg`。
- 懒加载：`import unirobot_infra.kinematics.sdk_backend` 不需要 SDK；构造
  `SdkKinematics`（即按下 G/B 键）时才解析路径，找不到会 raise RuntimeError 并给出克隆命令。
- **跨 solver 约定**：SDK 与 URDF 关节同序同号同零位，但连杆平移差 57–108 mm，且
  URDF 的固定 JointTCP 关节使 tcp site 系与 SDK 法兰系差一个常量旋转（见
  `tests/test_sdk_complement.py::test_joint_rotation_mapping` 的帧无关判据）。因此
  SDK IK 目标一律由 SDK 自己对参考构型 FK 生成，或直接用 CLI 给的臂基座系 xyzabc，
  绝不把 MuJoCo 的 TCP 平移喂给 SDK。

## A：`--pose` 绝对位姿命令（G 键）

```bash
unirobot-infra-teleop-pico sim --pose "576,180,316,97,83,105" --arm left --pose-speed 60
```

- `--pose "x,y,z,a,b,c"`：SDK 臂基座系（mm/deg）；`--arm {left,right}`（默认 left）；
  `--pose-speed` 关节速率限幅（deg/s，默认 60）。
- 运行时按一次 `G`：暂停跟踪（命令保持）→ 以当前关节为参考构型做 SDK 闭式 IK →
  失败（out_of_range / joint_exceeded+tags / degenerate）时打印原因、不移动；
  成功后该臂以 smoothstep 关节插值到位（时长按最大关节差与 `--pose-speed` 预算，
  与 R 回 home 同一先例），另一臂保持；到位后清状态，`reference is None` 使后续帧
  自动重新锚定。HUD 显示 `POSE left -> [q] ...%`。
- 动机：实验 E1——SDK 单点求解数值级精确且 ~30 µs，适合"命令臂去一个绝对位姿"；
  差分 IK 从固定 home 出发不跨冗余分支（E2），不适合绝对定位。

## B：臂角规划注入 home bias（B 键）

```bash
unirobot-infra-teleop-pico sim --elbow-angle 15
```

- 按一次 `B`：对 `--arm` 臂，以当前关节为参考、当前 wrist 目标存在为前提，SDK 侧
  `FK(ref)` 为目标做 `ik → ik_nsp`（zsp_type=1 参考平面取 `fk_nsp(ref)` 第一列，
  臂角取 `--elbow-angle`，默认 15°），解出的规划构型通过
  `MarvinTeleop.set_arm_plan_bias(side, q7_rad)` 替换 `solve()` 零空间 home 项的目标；
  再按一次清除。`reset_home` 的 `_reset_target` 仍用真实 home，不受 bias 影响。
- 效果：腕部主任务精度不变，冗余自由度被拉到"肘部抬 15°"的规划构型，
  HUD 显示 `PLAN left +15deg`。
- 动机：实验 E3——SDK 臂角参数化是精确的冗余控制（TCP 零误差下转 ±15°），而
  MuJoCo 肘部通道是弱次级任务（40 mm 目标只吸收 ~4 mm）；把闭式规划结果注入
  差分 IK 的零空间 home 项，是两种求解器最自然的分工点。
- 注意：左/右臂共用 `--elbow-angle` 数值（厂商 demo 中右臂肘上抬需取负，按需在
  右臂上按 B 前调 `--elbow-angle`）。

## C：遥操停滞安全门（拦截录制，C 键）

```bash
unirobot-infra-teleop-pico sim --stall-threshold 30 --stall-ticks 135
```

- `solve()` 每 tick 记录各臂 wrist 位置残差的最优值；残差持续高于
  `--stall-threshold`（mm，默认 30）且连续 `--stall-ticks`（默认 135 ≈ 1.5s@90Hz）
  无改善即置 `stalled=True`（HUD/headless 状态行显示 `STALLED (.. mm)`）。
- 目标消失或重置（`reset_reference` / 新目标）时清零；纯软件机制，不依赖 SDK。
- `stalled` 时按 `C` 只打印 `tracking stalled (residual XX mm); recording blocked`，
  不 toggle 相机录制。
- 动机：实验 E2/E5——差分 IK 没有内置不可达判定，不可达目标表现为阻尼"泄力"停滞
  （731 mm 残差静止）；录制停滞段会把无效跟踪数据写进数据集，需要显式拦截。

## 测试

`tests/test_sdk_complement.py`（全部离线）：关节旋转映射（帧无关判据，左右臂）、
后端 roundtrip/不可达严格 flag、停滞门升起与清零、plan bias 零空间拉动与清除、
pose-move 插值端点/中点。SDK 依赖项用 `requires_sdk_repo` 标记（探测
`TJ_FX_SDK_DIR` 或 `/tmp/tj_fx_sdk`，区别于真机 `requires_tianji_sdk`）。
