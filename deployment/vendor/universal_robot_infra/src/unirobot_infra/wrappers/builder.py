from omegaconf import DictConfig
from rich.console import Console

from unirobot_infra.devices.joint_gui import JointGui, build_joint_gui_parts
from unirobot_infra.devices.keyboard import KeyboardDevice
from unirobot_infra.devices.manus import ManusDevice
from unirobot_infra.devices.vive import ViveTrackerDevice
from unirobot_infra.envs.builder import build_base_env
from unirobot_infra.robots.compositional.group import RobotGroup
from unirobot_infra.teleop.mapping.manus_retarget import SharpaRetargeter
from unirobot_infra.utils.logging import get_logger
from unirobot_infra.wrappers.adaptive_hand_action import AdaptiveHandActionWrapper
from unirobot_infra.wrappers.camera_observation import CameraObservationWrapper
from unirobot_infra.wrappers.hand_alpha_action import HandAlphaActionWrapper
from unirobot_infra.wrappers.hand_action_projection import HandActionProjectionWrapper
from unirobot_infra.wrappers.image_display import ImageDisplayWrapper
from unirobot_infra.wrappers.rerun_visualization import RerunVisualizationWrapper
from unirobot_infra.wrappers.state_obs import StateObservationWrapper
from unirobot_infra.wrappers.teleop_intervention import (
    CompositeTeleopSource,
    JointGuiTeleopSource,
    KeyboardTeleopSource,
    ManusTeleopSource,
    TeleopInterventionWrapper,
    ViveTeleopSource,
)
from unirobot_infra.wrappers.tianji_delta_ee_action import TianjiDeltaEeActionWrapper


def build_env(cfg: DictConfig, robot_group: RobotGroup | None = None, console: Console | None = None):
    env = build_base_env(cfg, robot_group=robot_group)
    wrapper_specs = []
    for wrapper_name in cfg.wrappers:
        wrapper_cfg = cfg.wrappers[wrapper_name]
        wrapper_specs.append((int(wrapper_cfg.priority), wrapper_name, wrapper_cfg))
    for _, wrapper_name, wrapper_cfg in sorted(wrapper_specs, key=lambda item: item[0]):
        env = _apply_wrapper(cfg, env, wrapper_name, wrapper_cfg, console)
    return env


def _apply_wrapper(cfg: DictConfig, env, wrapper_name: str, wrapper_cfg: DictConfig, console: Console | None):
    if wrapper_name == "state_obs":
        return StateObservationWrapper(env, wrapper_cfg)
    if wrapper_name == "camera_observation":
        return CameraObservationWrapper(env, wrapper_cfg, cfg.camera.cameras)
    if wrapper_name == "image_display":
        return ImageDisplayWrapper(env, wrapper_cfg)
    if wrapper_name == "rerun_visualization":
        return RerunVisualizationWrapper(env, wrapper_cfg)
    if wrapper_name == "hand_action_projection":
        return HandActionProjectionWrapper(env, wrapper_cfg)
    if wrapper_name == "hand_alpha_action":
        return HandAlphaActionWrapper(env, wrapper_cfg)
    if wrapper_name == "adaptive_hand_action":
        return AdaptiveHandActionWrapper(env, wrapper_cfg)
    if wrapper_name == "tianji_delta_ee_action":
        return TianjiDeltaEeActionWrapper(env, wrapper_cfg, cfg.robot.hardware.arm)
    if wrapper_name == "teleop_intervention":
        return TeleopInterventionWrapper(env, _build_teleop_source(cfg, wrapper_cfg, console))
    raise ValueError("unsupported env wrapper")


def _build_teleop_source(cfg: DictConfig, wrapper_cfg: DictConfig, console: Console | None):
    if wrapper_cfg.source == "manus":
        return _build_manus_source(cfg, wrapper_cfg.device.manus, wrapper_cfg.mapping)
    if wrapper_cfg.source == "vive":
        return _build_vive_source(cfg, wrapper_cfg.device.vive, wrapper_cfg.device.deadman_keyboard, wrapper_cfg.mapping)
    if wrapper_cfg.source == "manus_vive":
        if wrapper_cfg.mapping.type != "manus_vive":
            raise ValueError("manus_vive source requires mapping.type=manus_vive")
        return CompositeTeleopSource(
            (
                _build_manus_source(cfg, wrapper_cfg.device.manus, wrapper_cfg.mapping.manus),
                _build_vive_source(
                    cfg,
                    wrapper_cfg.device.vive,
                    wrapper_cfg.device.deadman_keyboard,
                    wrapper_cfg.mapping.vive,
                ),
            )
        )
    if wrapper_cfg.source == "keyboard":
        if wrapper_cfg.mapping.type != "tianji_cartesian_jog":
            raise ValueError("keyboard source requires mapping.type=tianji_cartesian_jog")
        side = _single_arm_side(cfg)
        return KeyboardTeleopSource(
            keyboard=KeyboardDevice(wrapper_cfg.device.keyboard),
            mapping_cfg=wrapper_cfg.mapping,
            tianji_cfg=cfg.robot.hardware.arm,
            side=side,
            logger=get_logger(__name__),
        )
    if wrapper_cfg.source == "joint_gui":
        if wrapper_cfg.mapping.type != "direct_joint_action":
            raise ValueError("joint_gui source requires mapping.type=direct_joint_action")
        return JointGuiTeleopSource(
            JointGui(
                cfg=wrapper_cfg.device.joint_gui,
                parts=build_joint_gui_parts(cfg.robot, wrapper_cfg.device.joint_gui),
                console=console,
            )
        )
    raise ValueError("unsupported teleop source")


def _build_manus_source(cfg: DictConfig, device_cfg: DictConfig, mapping_cfg: DictConfig) -> ManusTeleopSource:
    if mapping_cfg.type != "sharpa_retarget":
        raise ValueError("manus source requires mapping.type=sharpa_retarget")
    hand_sides = _end_effector_sides(cfg)
    if not hand_sides:
        raise ValueError("manus source requires end_effector sides")
    return ManusTeleopSource(
        manus=ManusDevice(
            cfg=device_cfg,
            sharpa_cfg=cfg.robot.hardware.end_effector,
        ),
        retargeter=SharpaRetargeter(
            retarget_cfg=mapping_cfg.retarget,
            sharpa_cfg=cfg.robot.hardware.end_effector,
        ),
        hand_sides=hand_sides,
        recv_timeout_s=float(device_cfg.recv_timeout_s),
    )


def _build_vive_source(
    cfg: DictConfig,
    vive_device_cfg: DictConfig,
    deadman_keyboard_cfg: DictConfig,
    mapping_cfg: DictConfig,
) -> ViveTeleopSource:
    _require_vive_arm_sides(cfg, str(mapping_cfg.side))
    if mapping_cfg.type != "tianji_vive_local_se3_ik":
        raise ValueError("vive source requires mapping.type=tianji_vive_local_se3_ik")
    return ViveTeleopSource(
        vive=ViveTrackerDevice(vive_device_cfg),
        deadman_keyboard=KeyboardDevice(deadman_keyboard_cfg),
        mapping_cfg=mapping_cfg,
        tianji_cfg=cfg.robot.hardware.arm,
        recv_timeout_s=float(vive_device_cfg.timeout_s),
        logger=get_logger(__name__),
    )


def _end_effector_sides(cfg: DictConfig) -> tuple[str, ...]:
    return tuple(side for side in cfg.robot.sides if "end_effector" in cfg.robot.sides[side].parts)


def _single_arm_side(cfg: DictConfig) -> str:
    arm_sides = tuple(side for side in cfg.robot.sides if "arm" in cfg.robot.sides[side].parts)
    if len(arm_sides) != 1:
        raise ValueError("keyboard source requires exactly one arm side in robot config")
    return arm_sides[0]


def _require_vive_arm_sides(cfg: DictConfig, mapping_side: str) -> None:
    for side in _vive_controlled_sides(cfg, mapping_side):
        _require_arm_side(cfg, side, "vive")


def _vive_controlled_sides(cfg: DictConfig, mapping_side: str) -> tuple[str, ...]:
    if mapping_side == "both":
        return ("left", "right")
    return (mapping_side,)


def _require_arm_side(cfg: DictConfig, side: str, source: str) -> None:
    if side not in cfg.robot.sides:
        raise ValueError(f"{source} source requires robot side {side}")
    if "arm" not in cfg.robot.sides[side].parts:
        raise ValueError(f"{source} source requires robot side {side} with arm part")
