# 运动学求解包。两种 IK 方式的分工：
# - SDK 闭式 IK：本模块的 sdk_backend（懒加载开源 fx_kine，绝对位姿/臂角规划/可达性校验）；
# - MuJoCo 差分 IK：本模块的 wrist_and_elbow_steps（wrist 优先 + elbow 零空间任务，仿真/遥操侧）。

from .differential import wrist_and_elbow_steps

__all__ = ["wrist_and_elbow_steps"]
