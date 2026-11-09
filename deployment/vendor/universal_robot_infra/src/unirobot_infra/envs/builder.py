from pathlib import Path

from omegaconf import DictConfig

from unirobot_infra.controllers.robot_group import RobotGroupController
from unirobot_infra.envs.layout import RobotActionLayout
from unirobot_infra.envs.tianji_sharpa_kinematic_env import TianjiSharpaKinematicEnv
from unirobot_infra.envs.tianji_sharpa_env import TianjiSharpaEnv
from unirobot_infra.envs.tianji_sharpa_mujoco_env import TianjiSharpaMujocoEnv
from unirobot_infra.robots.arm.tianji import TianjiArm, TianjiRobot
from unirobot_infra.robots.compositional.group import RobotGroup
from unirobot_infra.robots.compositional.unit import RobotUnit
from unirobot_infra.robots.end_effector.sharpa import SharpaHand
from unirobot_infra.utils.sdk_paths import sharpa_manus_sdk_paths


def build_base_env(cfg: DictConfig, robot_group: RobotGroup | None = None):
    action_layout = RobotActionLayout.from_robot_config(cfg.robot)
    if cfg.env.type == "real":
        group = build_robot_group(cfg) if robot_group is None else robot_group
        return TianjiSharpaEnv(
            controller=RobotGroupController.from_group(group),
            action_layout=action_layout,
            cfg=cfg.env,
            arm_cfg=cfg.robot.hardware.arm if _robot_has_part(cfg, "arm") else None,
        )
    if cfg.env.type == "kinematic":
        return TianjiSharpaKinematicEnv(
            action_layout=action_layout,
            cfg=cfg.env,
        )
    if cfg.env.type == "mujoco":
        return TianjiSharpaMujocoEnv(
            action_layout=action_layout,
            cfg=cfg.env,
        )
    raise ValueError("unsupported env.type")


def build_robot_group(cfg: DictConfig) -> RobotGroup:
    arm_robot = TianjiRobot(cfg.robot.hardware.arm) if _robot_has_part(cfg, "arm") else None
    units = tuple(
        _build_unit(cfg, side, cfg.robot.sides[side], arm_robot)
        for side in cfg.robot.sides
    )
    return RobotGroup(name=cfg.robot.name, units=units)


def _build_unit(
    cfg: DictConfig,
    side: str,
    side_cfg: DictConfig,
    arm_robot: TianjiRobot | None,
) -> RobotUnit:
    arm = None
    end_effector = None
    if "arm" in side_cfg.parts:
        arm = TianjiArm(cfg.robot.hardware.arm, side=side, robot=arm_robot)
    if "end_effector" in side_cfg.parts:
        sdk_paths = sharpa_manus_sdk_paths(Path(cfg.robot.hardware.end_effector.sdk.vendor_root))
        end_effector = SharpaHand(cfg.robot.hardware.end_effector, side=side, sdk_paths=sdk_paths)
    return RobotUnit(side=side, arm=arm, end_effector=end_effector)


def _robot_has_part(cfg: DictConfig, role: str) -> bool:
    return any(role in cfg.robot.sides[side].parts for side in cfg.robot.sides)
