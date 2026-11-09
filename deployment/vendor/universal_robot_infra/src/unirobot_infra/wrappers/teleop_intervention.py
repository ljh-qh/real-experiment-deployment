from collections.abc import Sequence
from dataclasses import dataclass
import re
import threading
import time
from typing import Protocol

from omegaconf import ListConfig

from unirobot_infra.core.types import DeviceFrame, RobotAction
from unirobot_infra.devices.joint_gui import JointGui
from unirobot_infra.devices.keyboard import KeyboardDevice
from unirobot_infra.devices.manus import ManusDevice, ManusKeypoints
from unirobot_infra.devices.vive import ViveTrackerDevice
from unirobot_infra.robots.arm.tianji_tcp_pose import TianjiIkSolver
from unirobot_infra.teleop.mapping.tianji_cartesian_jog import TianjiCartesianJogMapper
from unirobot_infra.teleop.mapping.tianji_kinematics import TianjiKinematics
from unirobot_infra.teleop.mapping.manus_retarget import SharpaRetargeter, retargeted_joints_to_actions
from unirobot_infra.teleop.mapping.vive_arm_mapper import (
    ViveTianjiLocalSe3Mapper,
)
from unirobot_infra.wrappers.base import EnvLike, EnvWrapper


@dataclass(frozen=True)
class TeleopAction:
    device_frames: tuple[DeviceFrame, ...]
    actions: tuple[RobotAction, ...]


class TeleopActionSource(Protocol):
    last_read_timing: dict[str, float]

    def connect(self, info: dict) -> None: ...

    def read(self) -> TeleopAction | None: ...

    def close(self) -> None: ...


class DeviceFrameMapper(Protocol):
    def map_frame(self, frame: DeviceFrame) -> tuple[RobotAction, ...]: ...


class DeviceMappingTeleopSource:
    def __init__(self, device, mapper: DeviceFrameMapper) -> None:
        self.device = device
        self.mapper = mapper
        self.last_read_timing: dict[str, float] = {}

    def connect(self, info: dict) -> None:
        self.device.connect()

    def read(self) -> TeleopAction | None:
        started = time.monotonic()
        frame = self.device.read_frame()
        frame_ready = time.monotonic()
        actions = self.mapper.map_frame(frame)
        finished = time.monotonic()
        self.last_read_timing = {
            "read_frame_s": frame_ready - started,
            "map_frame_s": finished - frame_ready,
            "total_s": finished - started,
        }
        return TeleopAction(device_frames=(frame,), actions=actions)

    def close(self) -> None:
        self.device.close()


class CompositeTeleopSource:
    def __init__(self, sources: tuple[TeleopActionSource, ...]) -> None:
        self.sources = sources
        self.last_read_timing: dict[str, float] = {}

    def connect(self, info: dict) -> None:
        for source in self.sources:
            source.connect(info)

    def read(self) -> TeleopAction | None:
        started = time.monotonic()
        device_frames = []
        actions = []
        timings: dict[str, float] = {}
        for source in self.sources:
            source_started = time.monotonic()
            teleop_action = source.read()
            source_finished = time.monotonic()
            source_name = _teleop_source_name(source)
            timings[f"{source_name}_source_read_s"] = source_finished - source_started
            for key, value in source.last_read_timing.items():
                timings[f"{source_name}.{key}"] = value
            if teleop_action is not None:
                device_frames.extend(teleop_action.device_frames)
                actions.extend(teleop_action.actions)
        finished = time.monotonic()
        timings["composite_source_read_s"] = finished - started
        self.last_read_timing = timings
        if not actions:
            return None
        return TeleopAction(
            device_frames=tuple(device_frames),
            actions=_merge_robot_actions_by_side(tuple(actions)),
        )

    def close(self) -> None:
        for source in self.sources:
            source.close()


class ManusTeleopSource:
    def __init__(
        self,
        manus: ManusDevice,
        retargeter: SharpaRetargeter,
        hand_sides: tuple[str, ...],
        recv_timeout_s: float,
    ) -> None:
        self.manus = manus
        self.retargeter = retargeter
        self.hand_sides = hand_sides
        self.recv_timeout_s = recv_timeout_s
        self.last_read_timing: dict[str, float] = {}

    def connect(self, info: dict) -> None:
        self.manus.connect()
        self.retargeter.connect()
        self.retargeter.start()

    def read(self) -> TeleopAction | None:
        started = time.monotonic()
        keypoints = _recv_manus_keypoints_for_sides(
            manus=self.manus,
            hand_sides=self.hand_sides,
            timeout_s=self.recv_timeout_s,
        )
        keypoints_ready = time.monotonic()
        if keypoints is None:
            self.last_read_timing = {
                "recv_keypoints_s": keypoints_ready - started,
                "total_s": keypoints_ready - started,
            }
            return None
        frame = keypoints.to_device_frame(side=self.hand_sides[0], frame="manus")
        joints = self.retargeter.process(keypoints)
        finished = time.monotonic()
        self.last_read_timing = {
            "recv_keypoints_s": keypoints_ready - started,
            "retarget_s": finished - keypoints_ready,
            "total_s": finished - started,
        }
        actions = retargeted_joints_to_actions(joints, self.hand_sides)
        return TeleopAction(device_frames=(frame,), actions=actions)

    def close(self) -> None:
        self.retargeter.stop()
        self.manus.close()


class ViveTeleopSource:
    def __init__(
        self,
        vive: ViveTrackerDevice,
        deadman_keyboard: KeyboardDevice,
        mapping_cfg,
        tianji_cfg,
        recv_timeout_s: float,
        logger,
    ) -> None:
        self.vive = vive
        self.deadman_keyboard = deadman_keyboard
        self.mapping_cfg = mapping_cfg
        self.tianji_cfg = tianji_cfg
        self.recv_timeout_s = float(recv_timeout_s)
        self.logger = logger
        self.mappers: dict[str, ViveTianjiLocalSe3Mapper] = {}
        self.deadman_enabled_by_side: dict[str, bool] = {}
        self.last_read_timing: dict[str, float] = {}

    def connect(self, info: dict) -> None:
        self.mappers = {}
        for side in _vive_controlled_sides(str(self.mapping_cfg.side)):
            robot_state = _robot_state_by_side(info["robot_states"], side)
            _validate_tianji_initial_state(self.mapping_cfg, robot_state)
            arm_id = str(self.tianji_cfg.arms[side].arm_id)
            kinematics = TianjiKinematics(self.tianji_cfg, arm_id=arm_id)
            kinematics.initialize()
            solver = TianjiIkSolver(
                kinematics=kinematics,
                max_joint_step_deg=float(self.mapping_cfg.max_joint_step_deg),
                joint_limit_mode=str(self.mapping_cfg.joint_limit_mode),
            )
            self.mappers[side] = ViveTianjiLocalSe3Mapper(
                side=side,
                pose_name=str(self.mapping_cfg.tracker_id_by_side[side]),
                mount_rotation_from_tracker_local=self.mapping_cfg.mount_rotation_from_tracker_local_by_side[side],
                position_scale=float(self.mapping_cfg.position_scale),
                orientation_filter_enabled=bool(self.mapping_cfg.orientation_filter.enabled),
                orientation_filter_tau_s=float(self.mapping_cfg.orientation_filter.tau_s),
                orientation_filter_fallback_dt_s=float(self.mapping_cfg.orientation_filter.fallback_dt_s),
                solver=solver,
                robot_state=robot_state,
            )
        self.vive.connect()
        self.deadman_keyboard.connect()
        self.deadman_enabled_by_side = {side: False for side in self.mappers}
        self.logger.info("Vive/Tianji 已就绪 %s", _vive_mapper_summary(self.mappers))

    def read(self) -> TeleopAction | None:
        started = time.monotonic()
        if not self.mappers:
            raise RuntimeError("ViveTeleopSource is not connected")
        keyboard_frame = self.deadman_keyboard.read_frame()
        keyboard_ready = time.monotonic()
        active_sides = tuple(
            side
            for side in self.mappers
            if _vive_deadman_enabled_for_side(self.mapping_cfg, side, keyboard_frame.active_commands)
        )
        inactive_sides = tuple(side for side in self.mappers if side not in active_sides)
        for side in inactive_sides:
            if self.deadman_enabled_by_side[side]:
                self.mappers[side].clear_anchor()
                self.deadman_enabled_by_side[side] = False
                self.logger.info("Vive trigger released side=%s，停止同步", side)
        if not active_sides:
            self.last_read_timing = {
                "keyboard_s": keyboard_ready - started,
                "total_s": keyboard_ready - started,
            }
            return None

        tracker_frame = self.vive.recv_latest(timeout_s=self.recv_timeout_s)
        tracker_ready = time.monotonic()
        if tracker_frame is None:
            self.last_read_timing = {
                "keyboard_s": keyboard_ready - started,
                "recv_tracker_s": tracker_ready - keyboard_ready,
                "total_s": tracker_ready - started,
            }
            return None
        frames = tuple(tracker_frame.to_device_frame(side=side) for side in active_sides)
        for frame in frames:
            mapper = self.mappers[frame.side]
            if not self.deadman_enabled_by_side[frame.side]:
                mapper.anchor_from_pose(frame.poses[mapper.pose_name])
                self.deadman_enabled_by_side[frame.side] = True
                self.logger.info("Vive 已锚定 side=%s tracker=%s", frame.side, mapper.pose_name)
        actions = []
        for frame in frames:
            mapper = self.mappers[frame.side]
            try:
                actions.extend(mapper.map_frame(frame))
            except RuntimeError as exc:
                if _handle_tianji_ik_runtime_error(
                    exc=exc,
                    on_ik_failure=str(self.mapping_cfg.on_ik_failure),
                    logger=self.logger,
                    action_name="Vive",
                ):
                    finished = time.monotonic()
                    self.last_read_timing = {
                        "keyboard_s": keyboard_ready - started,
                        "recv_tracker_s": tracker_ready - keyboard_ready,
                        "map_actions_s": finished - tracker_ready,
                        "total_s": finished - started,
                    }
                    return None
        finished = time.monotonic()
        self.last_read_timing = {
            "keyboard_s": keyboard_ready - started,
            "recv_tracker_s": tracker_ready - keyboard_ready,
            "map_actions_s": finished - tracker_ready,
            "total_s": finished - started,
        }
        return TeleopAction(
            device_frames=frames + tuple(keyboard_frame.to_device_frame(side=side) for side in active_sides),
            actions=tuple(actions),
        )

    def close(self) -> None:
        self.deadman_keyboard.close()
        self.vive.close()


class KeyboardTeleopSource:
    def __init__(
        self,
        keyboard: KeyboardDevice,
        mapping_cfg,
        tianji_cfg,
        side: str,
        logger,
    ) -> None:
        self.keyboard = keyboard
        self.mapping_cfg = mapping_cfg
        self.tianji_cfg = tianji_cfg
        self.side = side
        self.logger = logger
        self.mapper: TianjiCartesianJogMapper | None = None
        self.last_read_timing: dict[str, float] = {}

    def connect(self, info: dict) -> None:
        side = self.side
        robot_state = _robot_state_by_side(info["robot_states"], side)
        _validate_tianji_initial_state(self.mapping_cfg, robot_state)
        arm_id = str(self.tianji_cfg.arms[side].arm_id)
        kinematics = TianjiKinematics(self.tianji_cfg, arm_id=arm_id)
        kinematics.initialize()
        solver = TianjiIkSolver(
            kinematics=kinematics,
            max_joint_step_deg=float(self.mapping_cfg.max_joint_step_deg),
            joint_limit_mode=str(self.mapping_cfg.joint_limit_mode),
        )
        self.mapper = TianjiCartesianJogMapper(
            side=side,
            solver=solver,
            robot_state=robot_state,
            sdk_xyz_from_robot_world=self.tianji_cfg.arms[side].sdk_xyz_from_robot_world,
            translation_speed_mm_s=float(self.mapping_cfg.translation_speed_mm_s),
            rotation_speed_deg_s=float(self.mapping_cfg.rotation_speed_deg_s),
            max_dt_s=float(self.mapping_cfg.max_dt_s),
        )
        self.keyboard.connect()
        self.logger.info("Keyboard/Tianji 已锚定 side=%s arm=%s", side, arm_id)

    def read(self) -> TeleopAction | None:
        started = time.monotonic()
        if self.mapper is None:
            raise RuntimeError("KeyboardTeleopSource is not connected")
        frame = self.keyboard.read_frame()
        frame_ready = time.monotonic()
        try:
            actions = self.mapper.map_frame(frame)
        except RuntimeError as exc:
            if _handle_tianji_ik_runtime_error(
                exc=exc,
                on_ik_failure=str(self.mapping_cfg.on_ik_failure),
                logger=self.logger,
                action_name="keyboard",
            ):
                finished = time.monotonic()
                self.last_read_timing = {
                    "keyboard_s": frame_ready - started,
                    "map_actions_s": finished - frame_ready,
                    "total_s": finished - started,
                }
                return None
        finished = time.monotonic()
        self.last_read_timing = {
            "keyboard_s": frame_ready - started,
            "map_actions_s": finished - frame_ready,
            "total_s": finished - started,
        }
        if actions is None:
            return None
        return TeleopAction(device_frames=(frame.to_device_frame(side=self.mapper.side),), actions=actions)

    def close(self) -> None:
        self.keyboard.close()


class JointGuiTeleopSource:
    def __init__(self, gui: JointGui) -> None:
        self.gui = gui
        self.pending_action: TeleopAction | None = None
        self.lock = threading.Lock()
        self.last_read_timing: dict[str, float] = {}

    def connect(self, info: dict) -> None:
        self.gui.connect()
        self.gui.set_initial_actions(tuple(_state_to_action(state) for state in info["robot_states"]))
        self.gui.set_action_callback(self._on_action)
        self.gui.start()

    def read(self) -> TeleopAction | None:
        started = time.monotonic()
        with self.lock:
            action = self.pending_action
            self.pending_action = None
        finished = time.monotonic()
        self.last_read_timing = {"total_s": finished - started}
        return action

    def close(self) -> None:
        self.gui.close()

    def _on_action(self, action: RobotAction) -> None:
        frame = self.gui.read_side_frame(action.side)
        with self.lock:
            self.pending_action = TeleopAction(device_frames=(frame,), actions=(action,))


class TeleopInterventionWrapper(EnvWrapper):
    def __init__(self, env: EnvLike, teleop_source: TeleopActionSource) -> None:
        super().__init__(env)
        self.teleop_source = teleop_source
        self.teleop_source_connected = False

    def reset(self) -> tuple[tuple[float, ...], dict]:
        if self.teleop_source_connected:
            self.teleop_source.close()
            self.teleop_source_connected = False
        observation, info = self.env.reset()
        self.teleop_source.connect(info)
        self.teleop_source_connected = True
        return observation, info

    def step(self, action: Sequence[float]) -> tuple[tuple[float, ...], float, bool, bool, dict]:
        policy_action = tuple(float(value) for value in action)
        teleop_read_started = time.monotonic()
        teleop_action = self.teleop_source.read()
        teleop_read_finished = time.monotonic()
        if teleop_action is None:
            env_step_started = time.monotonic()
            observation, reward, terminated, truncated, info = self.env.step(policy_action)
            env_step_finished = time.monotonic()
            info["policy_action"] = policy_action
            info["teleop_active"] = False
            info["timing"] = _merged_timing(
                {
                    "teleop_source_read_s": teleop_read_finished - teleop_read_started,
                    "inner_env_step_s": env_step_finished - env_step_started,
                },
                self.teleop_source.last_read_timing,
                info["timing"] if "timing" in info else {},
            )
            return observation, reward, terminated, truncated, info

        candidate_action = self.action_layout.overlay_actions(policy_action, teleop_action.actions)
        env_step_started = time.monotonic()
        observation, reward, terminated, truncated, info = self.env.step(candidate_action)
        env_step_finished = time.monotonic()
        info["policy_action"] = policy_action
        info["teleop_active"] = True
        info["device_frames"] = teleop_action.device_frames
        info["target_actions"] = teleop_action.actions
        info["timing"] = _merged_timing(
            {
                "teleop_source_read_s": teleop_read_finished - teleop_read_started,
                "inner_env_step_s": env_step_finished - env_step_started,
            },
            self.teleop_source.last_read_timing,
            info["timing"] if "timing" in info else {},
        )
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        if self.teleop_source_connected:
            self.teleop_source.close()
            self.teleop_source_connected = False
        self.env.close()


def _robot_state_by_side(states, side: str):
    state_by_side = {state.side: state for state in states}
    return state_by_side[side]


def _merged_timing(*timings: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for timing in timings:
        for key, value in timing.items():
            merged[str(key)] = float(value)
    return merged


def _teleop_source_name(source: TeleopActionSource) -> str:
    class_name = source.__class__.__name__
    if class_name.endswith("TeleopSource"):
        class_name = class_name[: -len("TeleopSource")]
    return _camel_to_snake(class_name)


def _camel_to_snake(value: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()


def _merge_robot_actions_by_side(actions: tuple[RobotAction, ...]) -> tuple[RobotAction, ...]:
    merged_by_side: dict[str, RobotAction] = {}
    for action in actions:
        if action.side not in merged_by_side:
            merged_by_side[action.side] = action
        else:
            merged_by_side[action.side] = _merge_robot_action(merged_by_side[action.side], action)
    return tuple(merged_by_side[side] for side in merged_by_side)


def _merge_robot_action(left: RobotAction, right: RobotAction) -> RobotAction:
    if left.arm_q_rad is not None and right.arm_q_rad is not None:
        raise ValueError("duplicate arm action for side")
    if left.end_effector_q_rad is not None and right.end_effector_q_rad is not None:
        raise ValueError("duplicate end_effector action for side")
    return RobotAction(
        side=left.side,
        timestamp_ns=max(left.timestamp_ns, right.timestamp_ns),
        arm_q_rad=left.arm_q_rad if left.arm_q_rad is not None else right.arm_q_rad,
        end_effector_q_rad=left.end_effector_q_rad
        if left.end_effector_q_rad is not None
        else right.end_effector_q_rad,
    )


def _vive_controlled_sides(mapping_side: str) -> tuple[str, ...]:
    if mapping_side == "both":
        return ("left", "right")
    return (mapping_side,)


def _vive_mapper_summary(mappers: dict[str, ViveTianjiLocalSe3Mapper]) -> str:
    return ", ".join(f"side={side} tracker={mappers[side].pose_name}" for side in mappers)


def _vive_deadman_enabled_for_side(mapping_cfg, side: str, active_commands: tuple[str, ...]) -> bool:
    deadman_commands = _vive_deadman_commands_by_side(mapping_cfg, side)
    return any(command in active_commands for command in deadman_commands)


def _vive_deadman_commands_by_side(mapping_cfg, side: str) -> tuple[str, ...]:
    command_cfg = mapping_cfg.deadman.command_by_side[side]
    if isinstance(command_cfg, str):
        if not command_cfg:
            raise ValueError("vive deadman command must not be empty")
        return (command_cfg,)
    if isinstance(command_cfg, (list, tuple, ListConfig)):
        commands = tuple(str(command) for command in command_cfg)
        if not commands:
            raise ValueError("vive deadman command list must not be empty")
        return commands
    raise TypeError(f"unsupported vive deadman command config type: {type(command_cfg).__name__}")


def _state_to_action(state) -> RobotAction:
    return RobotAction(
        side=state.side,
        timestamp_ns=state.timestamp_ns,
        arm_q_rad=state.arm_joint.q_rad if state.arm_joint is not None else None,
        end_effector_q_rad=state.end_effector_joint.q_rad if state.end_effector_joint is not None else None,
    )


def _recv_manus_keypoints_for_sides(
    manus: ManusDevice,
    hand_sides: tuple[str, ...],
    timeout_s: float,
) -> ManusKeypoints | None:
    import time

    deadline = time.monotonic() + float(timeout_s)
    latest_points_by_side: dict[str, tuple[tuple[float, ...], ...]] = {}
    while True:
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0.0:
            return None
        keypoints = manus.recv_latest(timeout_s=remaining_s)
        if keypoints is None:
            return None
        for side in hand_sides:
            if keypoints.side_has_data(side):
                latest_points_by_side[side] = keypoints.side_points(side)
        if all(side in latest_points_by_side for side in hand_sides):
            return _controlled_manus_keypoints(
                keypoints=keypoints,
                hand_sides=hand_sides,
                latest_points_by_side=latest_points_by_side,
            )


def _controlled_manus_keypoints(
    keypoints: ManusKeypoints,
    hand_sides: tuple[str, ...],
    latest_points_by_side: dict[str, tuple[tuple[float, ...], ...]],
) -> ManusKeypoints:
    return ManusKeypoints(
        timestamp_ns=keypoints.timestamp_ns,
        left=latest_points_by_side["left"] if "left" in hand_sides else _zero_points_like(keypoints.left),
        right=latest_points_by_side["right"] if "right" in hand_sides else _zero_points_like(keypoints.right),
    )


def _zero_points_like(points: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(0.0 for _ in point) for point in points)


def _handle_tianji_ik_runtime_error(
    exc: RuntimeError,
    on_ik_failure: str,
    logger,
    action_name: str,
) -> bool:
    if str(exc) not in _RECOVERABLE_TIANJI_IK_ERRORS:
        raise exc
    if on_ik_failure == "warn_hold":
        logger.warning("%s，跳过本帧 %s 动作", str(exc), action_name)
        return True
    if on_ik_failure == "raise":
        raise exc
    raise ValueError("unsupported on_ik_failure")


_RECOVERABLE_TIANJI_IK_ERRORS = {
    "Tianji IK failed",
    "Tianji IK target is out of range",
    "Tianji IK joint result exceeds limits",
}


def _validate_tianji_initial_state(mapping_cfg, robot_state) -> None:
    from unirobot_infra.core.units import rad_to_deg

    if robot_state.arm_joint is None:
        raise RuntimeError("Tianji teleop requires arm_joint state")
    q_deg = tuple(rad_to_deg(value) for value in robot_state.arm_joint.q_rad)
    if len(q_deg) != 7:
        raise RuntimeError("Tianji 初始关节反馈不是 7 维")
    all_zero_tol_deg = float(mapping_cfg.initial_q_all_zero_tolerance_deg)
    if all(abs(value) <= all_zero_tol_deg for value in q_deg):
        raise RuntimeError(
            "Tianji 初始关节反馈全零，不能作为遥操作 IK 锚点；"
            "请先确认机器人反馈流有效并处在非奇异 ready 姿态"
        )
    min_abs_joint4_deg = float(mapping_cfg.min_abs_joint4_deg)
    if abs(q_deg[3]) < min_abs_joint4_deg:
        raise RuntimeError(
            f"Tianji 初始第 4 关节过接近 0 deg，当前={q_deg[3]:.3f} deg，"
            "IK 会处于奇异附近；请先移动到非奇异 ready 姿态"
        )
