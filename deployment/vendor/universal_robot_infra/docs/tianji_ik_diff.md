# 天机 IK 两种方式对比（tianji-sharpa-infra main vs Universal_Robot_Infra main）

日期：2026-10-26。分支：`dev/tianji`。

## 背景

`tianji-sharpa-infra` main 当前主链路（PICO → MuJoCo 双臂 IK → Tianji SDK 真机）使用自研的
**MuJoCo 数值差分 IK**；`Universal_Robot_Infra` main 使用天机厂商 SDK 自带的
**`fx_kine` 闭式 IK** 封装。两种方式不是替代关系，而是分别服务于"仿真内高频步进"与"绝对位姿求解"
两个不同场景。合并时两种都保留，统一进 `unirobot_infra` 包。

## 两边实现

### 1. SDK 闭式 IK（Universal main 现有）

- 封装：`src/unirobot_infra/teleop/mapping/tianji_kinematics.py` 的 `TianjiKinematics`
- 底层：天机 SDK 的 `fx_kine.Marvin_Kine`，从 `sdk_dir` 动态 import，
  初始化需加载 SDK 的 kine 配置（DH/PNVA/BD/TYPE），按臂号区分
- 能力：
  - FK：`fk` → 4x4 矩阵 → `xyzabc`（mm + deg）
  - 互转：`xyzabc_to_mat4x4` / `mat4x4_to_xyzabc`
  - IK：`ik(FX_InvKineSolvePara)`，输入目标 `xyzabc` + 参考关节种子 +
    零空间参数 `zsp_type/zsp_para`；输出 7 关节绝对解（deg）
- 失败行为：目标不可达 / 越限直接 `RuntimeError`
- 外层步进保护：`robots/arm/tianji_tcp_pose.py` 的 `TianjiIkSolver` 在闭式解之上加
  `max_joint_step_deg` 关节步长限制（`reject` 报错或 `ramp` 截断），
  `TianjiTcpPoseMapper` 负责世界系 ↔ SDK base 系变换与 delta（xyz + 旋转 6d）应用

### 2. MuJoCo 数值差分 IK（tianji-sharpa-infra 现有）

- 实现：`MuJoCo-Marvin/mujoco_marvin/ik.py` 的 `wrist_and_elbow_steps`（单步求解核心）
- 底层：不依赖厂商 SDK；雅可比由 MuJoCo `mj_jacSite` 基于仿真模型给出，
  线性代数用 numpy + scipy（`lsq_linear` bvls）
- 单步算法（wrist 优先的阻尼最小二乘）：
  1. 主任务：wrist 6D 位姿误差（位置 + 0.5 加权旋转），damped lstsq
     （阻尼 0.01），单步限界 ±0.08 rad，触界用 bvls 求有界解
  2. 次任务：肘部位置误差投影到主任务雅可比的**精确 SVD 零空间**
     （`elbow_gain` 可调），附带微弱 home 关节偏好；整步均匀缩放保零空间性质，
     并按限界回缩
  3. 返回 `(primary, secondary)` 关节增量（rad），不返回绝对解
- 调用侧（`mujoco_marvin/teleop.py` 的 `solve`）：每控制周期最多 8 轮迭代，
  每轮 `mj_forward` 刷新雅可比 → 求 primary/secondary → secondary 经
  fraction ∈ {1, .5, .25, .125} 回溯线搜索，FK 校验 wrist 不偏离 primary 结果
  0.5 mm / 0.1 deg 带内，全部失败则丢弃 secondary；周期末对 command 做
  关节速度限幅（`max_speed_deg_s * dt`）

## 对比表

| 维度 | SDK 闭式 IK（main） | MuJoCo 差分 IK（tianji） |
| --- | --- | --- |
| 求解形态 | 一次性绝对解（SDK 内部闭式/数值） | 单步关节增量，调用方迭代积分 |
| 依赖 | 天机 `fx_kine` SDK + kine 配置 | MuJoCo + scipy + 机器人 XML 模型 |
| 输入契约 | `xyzabc`(mm,deg) 目标 + 参考 7 关节种子 | wrist/elbow 雅可比、6D 误差、当前关节、限界、home |
| 输出契约 | 7 关节绝对角度（deg） | `(primary, secondary)` 关节增量（rad） |
| 关节限界 | 解算后检查（越界/超限报错）+ 外层 `TianjiIkSolver` 步长截断 | 解算内硬限界（bvls，±0.08 rad/步）+ clip |
| 零空间处理 | SDK 黑盒参数 `zsp_type/zsp_para` | 显式 SVD 零空间：肘部任务 + home 偏好，带回溯 FK 校验 |
| 失败行为 | 直接抛错（reject 语义） | 返回可用增量；secondary 校验失败仅丢弃 secondary |
| 高频步进 | 不适合（每解一次是完整 IK，需外层限步） | 适合（天然增量、速度限幅、限界内生） |
| 真机可达性 | 直接给出 SDK 语义的可执行关节解 | 解基于仿真模型，真机执行需经 command 下发 |

## 结论与落位

- 两者互补：SDK IK 回答"目标位姿对应哪组关节角"（绝对定位、真机命令来源）；
  MuJoCo IK 回答"这一帧遥操输入下仿真臂如何平滑逼近目标"（高频步进、限界内生、
  显式肘部零空间）。
- 合并策略（main 主导，两种都进包）：
  - SDK 侧现有代码不动：`teleop/mapping/tianji_kinematics.py`（封装）、
    `robots/arm/tianji_tcp_pose.py`（步长保护 + TCP 位姿映射）
  - MuJoCo 差分 IK 吸收为 `src/unirobot_infra/kinematics/differential.py`
    （`wrist_and_elbow_steps` 原样迁入，`kinematics/` 目录在 main 上已存在但无源文件，
    属填空而非新架构）；包 `__init__.py` 注明两种 IK 的位置与适用场景
  - 调用编排（雅可比计算、迭代、线搜索、速度限幅，原 `teleop.py` 的 `solve`）
    随遥操主链路一并吸收，见 `docs/tianji_merge_manifest.md`
