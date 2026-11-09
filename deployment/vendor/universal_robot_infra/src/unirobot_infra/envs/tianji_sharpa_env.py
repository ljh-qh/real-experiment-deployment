from collections.abc import Sequence
from dataclasses import dataclass, field
import logging
import math
import time

from omegaconf import DictConfig

from unirobot_infra.controllers.robot_group import RobotGroupController
from unirobot_infra.core.time import now_ns
from unirobot_infra.core.types import DeviceFrame, RobotAction, RobotState
from unirobot_infra.core.units import deg_to_rad
from unirobot_infra.envs.layout import RobotActionLayout
from unirobot_infra.envs.spaces import BoxSpace
from unirobot_infra.envs.target_action import target_flat_action_from_targets
from unirobot_infra.robots.arm.tianji_tcp_pose import TianjiTcpPoseMapper


_LOGGER = logging.getLogger(__name__)


@dataclass
class TianjiSharpaEnv:
    controller: RobotGroupController
    action_layout: RobotActionLayout
    cfg: DictConfig
    arm_cfg: DictConfig | None
    connected: bool = False
    action_space: BoxSpace = field(init=False)
    observation_space: BoxSpace = field(init=False)
    tcp_mappers: dict[str, TianjiTcpPoseMapper] = field(init=False)

    def __post_init__(self) -> None:
        self.action_space = self.action_layout.action_space()
        self.observation_space = self.action_layout.observation_space()
        action_tcp_position_tolerance_m = float(self.cfg.runtime.action_tcp_position_tolerance_m)
        if action_tcp_position_tolerance_m < 0.0:
            raise ValueError("env.runtime.action_tcp_position_tolerance_m must be non-negative")
        if action_tcp_position_tolerance_m > 0.0:
            if self.arm_cfg is None:
                raise ValueError("env.runtime.action_tcp_position_tolerance_m requires robot.hardware.arm")
            self.tcp_mappers = {
                segment.side: TianjiTcpPoseMapper.fk_from_config(self.arm_cfg, segment.side)
                for segment in self.action_layout.segments
                if segment.role == "arm"
            }
        else:
            self.tcp_mappers = {}

    def reset(self) -> tuple[tuple[float, ...], dict]:
        self.connect()
        sent_actions = ()
        states = ()
        if bool(self.cfg.reset.move_to_initial_state):
            sent_actions = self._move_to_initial_state()
            states = self._wait_until_actions_reached(sent_actions)
        else:
            states = self.controller.read_state()
        observation = self.action_layout.observation_from_states(states)
        return observation, _env_info(
            robot_states=states,
            target_actions=sent_actions,
            sent_actions=sent_actions,
            device_frames=(),
        )

    def connect(self) -> None:
        if not self.connected:
            self.controller.connect()
            self.connected = True

    def initial_state_actions(self) -> tuple[RobotAction, ...]:
        target_action = target_flat_action_from_targets(self.cfg.reset.targets, self.action_layout.segments)
        return self.action_layout.actions_from_flat(target_action, timestamp_ns=now_ns())

    def move_to_actions_until_reached(
        self,
        actions: tuple[RobotAction, ...],
        log_label: str,
    ) -> tuple[RobotState, ...]:
        self._ensure_connected()
        if not actions:
            return self.controller.read_state()
        sent_actions = self._send_reset_actions(actions, log_label)
        return self._wait_until_actions_reached(sent_actions, log_label)

    def wait_until_actions_reached(
        self,
        actions: tuple[RobotAction, ...],
        log_label: str,
    ) -> tuple[RobotState, ...]:
        self._ensure_connected()
        return self._wait_until_arm_actions_reached(actions, log_label)

    def step(self, action: Sequence[float]) -> tuple[tuple[float, ...], float, bool, bool, dict]:
        self._ensure_connected()
        timestamp_ns = now_ns()
        actions = self.action_layout.actions_from_flat(action, timestamp_ns=timestamp_ns)
        action_blocking = bool(self.cfg.runtime.action_blocking)
        send_started = time.monotonic()
        self.controller.send_actions(actions, blocking=False)
        send_finished = time.monotonic()
        wait_started = time.monotonic()
        if action_blocking:
            states = self._wait_until_arm_actions_reached(actions, "env step arm target")
        else:
            states = self.controller.read_state()
        read_finished = time.monotonic()
        observation = self.action_layout.observation_from_states(states)
        info = _env_info(
            robot_states=states,
            target_actions=actions,
            sent_actions=actions,
            device_frames=(),
        )
        info["timing"] = {
            "controller_send_actions_s": send_finished - send_started,
            "controller_read_state_s": read_finished - send_finished,
            "controller_step_s": read_finished - send_started,
            "controller_action_blocking": action_blocking,
            "controller_action_wait_s": read_finished - wait_started,
        }
        return (
            observation,
            0.0,
            False,
            False,
            info,
        )

    def close(self) -> None:
        self.controller.close()
        self.connected = False

    def _ensure_connected(self) -> None:
        if not self.connected:
            raise RuntimeError("TianjiSharpaEnv is not connected")

    def _move_to_initial_state(self) -> tuple[RobotAction, ...]:
        return self._send_reset_actions(self.initial_state_actions(), "env reset target")

    def _send_reset_actions(self, actions: tuple[RobotAction, ...], log_label: str) -> tuple[RobotAction, ...]:
        settle_steps = int(self.cfg.reset.settle_steps)
        if settle_steps <= 0:
            raise ValueError("env.reset.settle_steps must be positive")
        period_s = 1.0 / float(self.cfg.runtime.rate_hz)
        _LOGGER.info(
            "%s send start settle_steps=%d rate_hz=%.3f",
            log_label,
            settle_steps,
            float(self.cfg.runtime.rate_hz),
        )
        for _ in range(settle_steps):
            loop_start = time.monotonic()
            self.controller.send_actions(_with_timestamp(actions), blocking=False)
            _sleep_remaining(loop_start, period_s)
        sent_actions = _with_timestamp(actions)
        _LOGGER.info("%s send done", log_label)
        return sent_actions

    def _wait_until_actions_reached(
        self,
        actions: tuple[RobotAction, ...],
        log_label: str = "env reset target",
    ) -> tuple[RobotState, ...]:
        if not actions:
            return self.controller.read_state()
        arm_tolerance_rad = deg_to_rad(float(self.cfg.reset.arm_tolerance_deg))
        end_effector_tolerance_rad = float(self.cfg.reset.end_effector_tolerance_rad)
        wait_timeout_s = float(self.cfg.reset.wait_timeout_s)
        wait_poll_s = float(self.cfg.reset.wait_poll_s)
        progress_log_interval_s = float(self.cfg.reset.progress_log_interval_s)
        if wait_timeout_s <= 0.0:
            raise ValueError("env.reset.wait_timeout_s must be positive")
        if wait_poll_s <= 0.0:
            raise ValueError("env.reset.wait_poll_s must be positive")
        if progress_log_interval_s <= 0.0:
            raise ValueError("env.reset.progress_log_interval_s must be positive")
        start_s = time.monotonic()
        deadline = time.monotonic() + wait_timeout_s
        last_progress_log_s = 0.0
        while True:
            states = self.controller.read_state()
            if _actions_reached(states, actions, arm_tolerance_rad, end_effector_tolerance_rad):
                _LOGGER.info("%s reached elapsed_s=%.3f", log_label, time.monotonic() - start_s)
                return states
            now_s = time.monotonic()
            if now_s - last_progress_log_s >= progress_log_interval_s:
                last_progress_log_s = now_s
                _LOGGER.info(
                    "%s waiting elapsed_s=%.3f error=%s",
                    log_label,
                    now_s - start_s,
                    _actions_error_summary(states, actions),
                )
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"{log_label} not reached within timeout: "
                    f"{_actions_error_summary(states, actions)}"
                )
            self.controller.send_actions(_with_timestamp(actions), blocking=False)
            time.sleep(wait_poll_s)

    def _wait_until_arm_actions_reached(
        self,
        actions: tuple[RobotAction, ...],
        log_label: str,
    ) -> tuple[RobotState, ...]:
        arm_actions = _arm_actions_only(actions)
        if not arm_actions:
            return self.controller.read_state()
        arm_tolerance_rad = deg_to_rad(float(self.cfg.runtime.action_arm_tolerance_deg))
        tcp_position_tolerance_m = float(self.cfg.runtime.action_tcp_position_tolerance_m)
        wait_timeout_s = float(self.cfg.runtime.action_wait_timeout_s)
        wait_poll_s = float(self.cfg.runtime.action_wait_poll_s)
        progress_log_interval_s = float(self.cfg.runtime.action_progress_log_interval_s)
        if tcp_position_tolerance_m < 0.0:
            raise ValueError("env.runtime.action_tcp_position_tolerance_m must be non-negative")
        if wait_timeout_s <= 0.0:
            raise ValueError("env.runtime.action_wait_timeout_s must be positive")
        if wait_poll_s <= 0.0:
            raise ValueError("env.runtime.action_wait_poll_s must be positive")
        if progress_log_interval_s <= 0.0:
            raise ValueError("env.runtime.action_progress_log_interval_s must be positive")
        start_s = time.monotonic()
        deadline = time.monotonic() + wait_timeout_s
        last_progress_log_s = start_s
        while True:
            states = self.controller.read_state()
            if _arm_actions_reached(states, arm_actions, arm_tolerance_rad) and _arm_tcp_positions_reached(
                states,
                arm_actions,
                self.tcp_mappers,
                tcp_position_tolerance_m,
            ):
                return states
            now_s = time.monotonic()
            if now_s - last_progress_log_s >= progress_log_interval_s:
                last_progress_log_s = now_s
                _LOGGER.info(
                    "%s waiting elapsed_s=%.3f error=%s",
                    log_label,
                    now_s - start_s,
                    _arm_actions_error_summary(states, arm_actions, self.tcp_mappers, tcp_position_tolerance_m),
                )
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"{log_label} not reached within timeout: "
                    f"{_arm_actions_error_summary(states, arm_actions, self.tcp_mappers, tcp_position_tolerance_m)}"
                )
            time.sleep(wait_poll_s)


def _env_info(
    robot_states: tuple[RobotState, ...],
    target_actions: tuple[RobotAction, ...],
    sent_actions: tuple[RobotAction, ...],
    device_frames: tuple[DeviceFrame, ...],
) -> dict:
    return {
        "robot_states": robot_states,
        "target_actions": target_actions,
        "sent_actions": sent_actions,
        "device_frames": device_frames,
    }


def _sleep_remaining(loop_start: float, period_s: float) -> None:
    remaining = period_s - (time.monotonic() - loop_start)
    if remaining > 0.0:
        time.sleep(remaining)


def _with_timestamp(actions: tuple[RobotAction, ...]) -> tuple[RobotAction, ...]:
    timestamp_ns = now_ns()
    return tuple(
        RobotAction(
            side=action.side,
            timestamp_ns=timestamp_ns,
            arm_q_rad=action.arm_q_rad,
            end_effector_q_rad=action.end_effector_q_rad,
        )
        for action in actions
    )


def _arm_actions_only(actions: tuple[RobotAction, ...]) -> tuple[RobotAction, ...]:
    return tuple(
        RobotAction(
            side=action.side,
            timestamp_ns=action.timestamp_ns,
            arm_q_rad=action.arm_q_rad,
            end_effector_q_rad=None,
        )
        for action in actions
        if action.arm_q_rad is not None
    )


def _arm_actions_reached(
    states: tuple[RobotState, ...],
    actions: tuple[RobotAction, ...],
    arm_tolerance_rad: float,
) -> bool:
    return _actions_reached(states, actions, arm_tolerance_rad, 0.0)


def _arm_actions_error_summary(
    states: tuple[RobotState, ...],
    actions: tuple[RobotAction, ...],
    tcp_mappers: dict[str, TianjiTcpPoseMapper],
    tcp_position_tolerance_m: float,
) -> str:
    parts = [_actions_error_summary(states, actions)]
    if tcp_position_tolerance_m > 0.0:
        parts.append(_arm_tcp_position_error_summary(states, actions, tcp_mappers))
    return ", ".join(part for part in parts if part)


def _arm_tcp_positions_reached(
    states: tuple[RobotState, ...],
    actions: tuple[RobotAction, ...],
    tcp_mappers: dict[str, TianjiTcpPoseMapper],
    tcp_position_tolerance_m: float,
) -> bool:
    if tcp_position_tolerance_m <= 0.0:
        return True
    state_by_side = {state.side: state for state in states}
    for action in actions:
        if action.arm_q_rad is None:
            continue
        state = state_by_side[action.side]
        if state.arm_joint is None:
            raise RuntimeError(f"RobotState missing arm_joint for side {action.side}")
        if _tcp_position_error_m(tcp_mappers[action.side], state.arm_joint.q_rad, action.arm_q_rad) > tcp_position_tolerance_m:
            return False
    return True


def _arm_tcp_position_error_summary(
    states: tuple[RobotState, ...],
    actions: tuple[RobotAction, ...],
    tcp_mappers: dict[str, TianjiTcpPoseMapper],
) -> str:
    state_by_side = {state.side: state for state in states}
    parts = []
    for action in actions:
        if action.arm_q_rad is None:
            continue
        state = state_by_side[action.side]
        if state.arm_joint is None:
            raise RuntimeError(f"RobotState missing arm_joint for side {action.side}")
        error_m = _tcp_position_error_m(tcp_mappers[action.side], state.arm_joint.q_rad, action.arm_q_rad)
        parts.append(f"{action.side}.tcp_pos_error_m={error_m:.6f}")
    return ", ".join(parts)


def _tcp_position_error_m(
    mapper: TianjiTcpPoseMapper,
    actual_q_rad: tuple[float, ...],
    expected_q_rad: tuple[float, ...],
) -> float:
    actual_xyz = _tcp_xyz_m(mapper, actual_q_rad)
    expected_xyz = _tcp_xyz_m(mapper, expected_q_rad)
    return math.sqrt(sum((actual_xyz[index] - expected_xyz[index]) ** 2 for index in range(3)))


def _tcp_xyz_m(mapper: TianjiTcpPoseMapper, q_rad: tuple[float, ...]) -> tuple[float, float, float]:
    pose = mapper.tcp_pose_world_from_q_rad(q_rad)
    return (float(pose[0][3]), float(pose[1][3]), float(pose[2][3]))


def _actions_reached(
    states: tuple[RobotState, ...],
    actions: tuple[RobotAction, ...],
    arm_tolerance_rad: float,
    end_effector_tolerance_rad: float,
) -> bool:
    state_by_side = {state.side: state for state in states}
    for action in actions:
        state = state_by_side[action.side]
        if action.arm_q_rad is not None:
            if state.arm_joint is None:
                raise RuntimeError(f"RobotState missing arm_joint for side {action.side}")
            if _max_abs_error(state.arm_joint.q_rad, action.arm_q_rad) > arm_tolerance_rad:
                return False
        if action.end_effector_q_rad is not None:
            if state.end_effector_joint is None:
                raise RuntimeError(f"RobotState missing end_effector_joint for side {action.side}")
            if _max_abs_error(state.end_effector_joint.q_rad, action.end_effector_q_rad) > end_effector_tolerance_rad:
                return False
    return True


def _actions_error_summary(states: tuple[RobotState, ...], actions: tuple[RobotAction, ...]) -> str:
    state_by_side = {state.side: state for state in states}
    parts = []
    for action in actions:
        state = state_by_side[action.side]
        if action.arm_q_rad is not None:
            if state.arm_joint is None:
                raise RuntimeError(f"RobotState missing arm_joint for side {action.side}")
            parts.append(_error_summary_part(action.side, "arm", state.arm_joint.q_rad, action.arm_q_rad))
        if action.end_effector_q_rad is not None:
            if state.end_effector_joint is None:
                raise RuntimeError(f"RobotState missing end_effector_joint for side {action.side}")
            parts.append(
                _error_summary_part(
                    action.side,
                    "end_effector",
                    state.end_effector_joint.q_rad,
                    action.end_effector_q_rad,
                )
            )
    return ", ".join(parts)


def _error_summary_part(side: str, role: str, actual: tuple[float, ...], expected: tuple[float, ...]) -> str:
    error, index = _max_abs_error_with_index(actual, expected)
    return (
        f"{side}.{role}_max_error_rad={error:.6f}"
        f"(joint_index={index}, expected={expected[index]:.6f}, actual={actual[index]:.6f})"
    )


def _max_abs_error(actual: tuple[float, ...], expected: tuple[float, ...]) -> float:
    error, _ = _max_abs_error_with_index(actual, expected)
    return error


def _max_abs_error_with_index(actual: tuple[float, ...], expected: tuple[float, ...]) -> tuple[float, int]:
    if len(actual) != len(expected):
        raise ValueError("robot state length does not match target length")
    if not actual:
        raise ValueError("robot state length must be positive")
    errors = tuple(abs(actual_value - expected_value) for actual_value, expected_value in zip(actual, expected))
    max_index = max(range(len(errors)), key=lambda index: errors[index])
    return errors[max_index], max_index
