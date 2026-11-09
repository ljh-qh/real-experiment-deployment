# MuJoCo 差分 IK vs 天机 SDK 闭式 IK：示例对比实验

日期：2026-10-27。分支：`dev/tianji`。承 `docs/tianji_ik_diff.md` 的定性对比，本文用
curobo 示例风格的任务集（单点 / 批量 / 冗余构型 / 限位 / 不可达 / 轨迹连续）做量化对比。

##  setup 与复现

- **SDK 侧**：开源仓库 [TJ_FX_ROBOT_CONTRL_SDK](https://github.com/cynthia-you/TJ_FX_ROBOT_CONTRL_SDK)
  `master` 分支（注意默认 main 分支只有 LICENSE），Apache-2.0，`SDK_PYTHON/fx_kine.py`
  ctypes 加载 `libKine.so`（x86-64，纯计算库，无控制器依赖）。机型配置
  `CommonConfig/ccs_m6_40.MvKDCfg`，`arm_type=0`（左臂）。
- **MuJoCo 侧**：本仓库遥操编排 `unirobot_infra.teleop.marvin_teleop.MarvinTeleop.solve()`
  （90 Hz tick、120°/s 关节速率限幅、每 tick 最多 8 轮 `wrist_and_elbow_steps`），
  模型 `assets/robot/marvin_dual_arm`。
- **实验脚本**：`scripts/ik_compare_mujoco_sdk.py`（种子固定 `SEED=20261027`，
  全量约 40 s）。SDK 仓库路径用 `TJ_FX_SDK_DIR` 覆盖，默认 `/tmp/tj_fx_sdk`：

  ```bash
  git clone --depth 1 -b master https://gh-proxy.com/https://github.com/cynthia-you/TJ_FX_ROBOT_CONTRL_SDK /tmp/tj_fx_sdk
  uv sync --group episode --group plot   # 确保 mujoco/numpy/scipy 在
  .venv/bin/python scripts/ik_compare_mujoco_sdk.py
  ```

- **公平性声明**：目标位姿由**各求解器自己的 FK** 生成（自洽），坐标系与单位各自原生
  （SDK：DH/mm；MuJoCo：URDF/m），结果统一换算 mm/deg。经验证两套模型**不存在固定
  SE(3) 基座映射**（旋转部分一致到 0.003°，平移差 57–108 mm，连杆参数有真实差异），
  因此"同一世界目标"的交叉对比不可做，已跳过。

## E1 单点基本求解（home 附近，参考 = 生成构型）

| solver | 位置误差 (mm) | 姿态误差 (deg) | 耗时 (s) | 关节跳变 (deg) |
| --- | --- | --- | --- | --- |
| SDK | 1.5e-5 | 0 | **3.0e-5** | 0.0000（精确复现） |
| MuJoCo | 2.3e-13 | 2.4e-14 | 0.256（57 tick × 4.5 ms） | 0.0025 |

两者都到数值精度极限；MuJoCo 单次"解到收敛"慢约 8500×，但每 tick 4.5 ms 是 90 Hz
预算的一半，实时可用。

## E2 批量随机可达目标（60 个，限位中段 ±40%）

| solver | 成功率 | 位置误差 mean/p95/max (mm) | 姿态误差 max (deg) | 耗时 mean/p95 (s) | 关节偏差 mean/max (deg) |
| --- | --- | --- | --- | --- | --- |
| SDK | **100%** | 2e-5 / 9e-5 / 1.4e-4 | 2e-5 | **3.1e-5 / 3.3e-5** | 0.000 / 0.000 |
| MuJoCo | 98.3% | 1.7 / 2.8e-4 / 100.4 | 0.38 | 0.475 / 0.701 | 98.7 / 244.5 |

- MuJoCo 的 6D 腕部主任务精度极高（p95 = 0.0003 mm），但 43/60 收敛到了 **TCP 相同、
  冗余分支不同**的构型（肘部残差 mean 40 mm）——从固定 home 出发的局部迭代不跨分支，
  这是差分法的本质行为，不是 bug。
- 1 个目标（idx 41）停滞在 100 mm 残差的局部极小。注意 MuJoCo 的"收敛"标志表示
  迭代 stationary，**不等于可达**，调用方需用残差自行判失败。

## E3 冗余 / 肘部构型（同一 TCP，3 组构型）

| solver | 参数 | TCP 误差 (mm/deg) | 肘部变化 | 最大关节偏差 (deg) | 耗时 (s) |
| --- | --- | --- | --- | --- | --- |
| SDK | zsp_angle 0/+15/−15 | 3e-6 / 0（全部） | 无肘点接口 | 0.0 / 12.3 / 13.6 | 3.0e-5 |
| MuJoCo | 肘目标 +0/+40/−40 mm | 0 / 0（全部） | 实际移动 0.2/4.2/3.0 mm | 0.04 / 0.74 / 0.53 | 0.245 |

SDK 的臂角参数化是**精确的冗余控制**：`ik_nsp` 在 TCP 零误差下把构型绕肩-腕轴转
±15°（J4 保持不动，其余关节联动）。MuJoCo 的肘部通道是腕优先设计下的**弱次级任务**
（增益 0.35 + 回溯接受准则）：40 mm 的肘目标偏移只被吸收约 4 mm。**角度参数 vs
位置目标的效力差异是结构性的**——要达 SDK 等效的冗余控制，需要把肘部任务升级为主
任务级或提高增益并放宽腕部精度。

## E4 关节限位边界（参考 = 贴限位构型，目标再推过限位）

| case | SDK 返回 | IsJntExd / 超限值 | 限位内分支数 | MuJoCo TCP 误差 | MuJoCo 行为 |
| --- | --- | --- | --- | --- | --- |
| J6=63°（越过 +60°） | ✓ 解 J6=63° | **True / 3.00°** | **0** | 1.6e-13 mm | J6 截断在 60° |
| J1=174° | ✓ 解 J1=174° | **True / 4.71°** | 0 | 1.2e-13 mm | 无（URDF 允许 178°） |
| J1=165°（对照） | ✓ | False / 0 | 1 | 2.8e-13 mm | 无 |

- ⚠️ **意外发现：同一台机器人的两套限位不一致**——URDF `jnt_range` J1/J3/J5 = ±178°，
  SDK PNVA = ±170°（其余关节一致）。J1=174° 的目标 SDK 报超限、MuJoCo 精确解出；
  真机部署若以 URDF 为准规划，会以超限构型到达 SDK 限位 8° 差的外缘。
- 失败语义对比：SDK **报 tag 但仍返回越限解**（TCP 精确，把取舍交给调用方，限位内
  分支数为 0 时别无选择）；MuJoCo 把关节**截断在限位**并用冗余自由度在边界上把 TCP
  补回数值精度。

## E5 不可达目标（+1000 mm）

| solver | 行为 | 标志 | 残差 | 耗时 |
| --- | --- | --- | --- | --- |
| SDK | 返回 False | **IsOutRange=True** | – | 1.2e-5 s |
| MuJoCo | 71 tick 后停滞 | 无内置不可达判定 | **731 mm** / 0.26°，无关节顶限位 | 0.484 s |

SDK 有显式可达性判定（与官方 `showcase_ik_failed_conclusion.py` 语义一致）；
MuJoCo 表现为阻尼最小二乘的整体"泄力"停滞，需要调用方用残差判失败。

## E6 轨迹连续性（20 个插值 TCP 目标，两种参考策略）

| solver | 策略 | 相邻解最大跳变 (deg) | 平均跳变 (deg) | 总变差 (deg) | 平均耗时 (s)/目标 |
| --- | --- | --- | --- | --- | --- |
| SDK | (a) ref 固定 = 起点 | **139.1**（i=18 分支翻转） | 23.2 | 440 | 2e-5 |
| SDK | (b) ref = 上一解 | **49.2**（i=3） | 17.8 | 337 | 2e-5 |
| MuJoCo | (a) 每点重置 + 固定参考 | 6.2 | 5.5 | 105 | 0.319（67 tick） |
| MuJoCo | (b) 连续流（指令延续） | **6.2** | 5.2 | 99 | **0.033**（8.6 tick） |

闭式 IK 即使"参考 = 上一解"也会**分支翻转**（最近分支选择逐点独立，七轴 CCS 构型
最多 4 组解）；MuJoCo 连续流天然平滑（最大 6.2° 主要来自速率限幅），这正是遥操场景
选差分 IK 的核心原因。"ref=上一解"策略把 SDK 总变差降 25%，但无法根除翻转。

## SDK 使用注意（实验中发现）

1. `ik_nsp` **不能单独调用**：必须先对同一结构体跑 `ik()` 再 `ik_nsp()`，否则返回
   False/垃圾解；`ik()` 忽略 `m_Input_ZSP_Angle`，臂角参数只在 `ik_nsp` 路径生效。
2. `ik_nsp` 返回时 `m_OutPut_Result_Num=0`，选解信息只看 `m_Output_RetJoint`。
3. `ik()` 返回 True 不等于可直接驱动：从 `m_OutPut_AllJoint` 自选解时须自查 J2 奇异区
   （±0.05°）、PNVA 限位、六七关节按 BD 多项式做干涉校验（`fx_kine.py:673-686`）。

## 结论

| 维度 | SDK 闭式 IK | MuJoCo 差分 IK |
| --- | --- | --- |
| 精度 | 数值级（~1e-5 mm） | 数值级（~1e-4 mm p95） |
| 单解速度 | ~30 µs | 4.5 ms/tick（实时预算内） |
| 轨迹连续性 | 会分支翻转（最大 139°） | 连续流平滑（≤6.2°） |
| 冗余控制 | 臂角参数精确（ik_nsp） | 肘部弱次级任务 |
| 失败语义 | 显式 flag（IsOutRange/IsJntExd/IsDeg），仍返回越限解 | 截断限位/停滞，需残差判失败 |
| 适用 | 绝对定位、选解、可达性判定、臂角规划 | 高频遥操步进、限界内生平滑 |

两者互补关系与 `docs/tianji_ik_diff.md` 的定性判断一致；本实验新增的可执行结论是：
**遥操主链继续用差分 IK，SDK IK 适合做绝对目标预解、可达性/限位校验与臂角规划**；
同时应统一 URDF 与 PNVA 的限位定义（当前 J1/J3/J5 差 8°）。

## 局限

目标为各 solver 自洽 FK（无共享世界帧）；无碰撞/干涉建模（SDK 的 `libInterfCheck`
未涉及）；MuJoCo 耗时含速率限幅收敛路径，与 SDK 单次调用不对等，故同时给出每 tick
耗时；SDK 限位语义以 PNVA 为准，`m_Output_RunLmtP/N` 运行限位未验证。
