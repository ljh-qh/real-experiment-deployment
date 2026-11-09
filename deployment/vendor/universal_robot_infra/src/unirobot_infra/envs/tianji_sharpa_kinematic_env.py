from collections.abc import Sequence
from dataclasses import dataclass, field
import time

from omegaconf import DictConfig

from unirobot_infra.core.time import now_ns
from unirobot_infra.core.types import DeviceFrame, JointState, RobotAction, RobotState
from unirobot_infra.envs.layout import RobotActionLayout
from unirobot_infra.envs.spaces import BoxSpace
from unirobot_infra.envs.target_action import target_flat_action_from_targets


@dataclass
class TianjiSharpaKinematicEnv:
    action_layout: RobotActionLayout
    cfg: DictConfig
    connected: bool = False
    action_space: BoxSpace = field(init=False)
    observation_space: BoxSpace = field(init=False)
    _states_by_side: dict[str, RobotState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.action_space = self.action_layout.action_space()
        self.observation_space = self.action_layout.observation_space()

    def reset(self) -> tuple[tuple[float, ...], dict]:
        self.connect()
        initial_actions = self.initial_state_actions()
        self._apply_actions(initial_actions)
        sent_actions = initial_actions if bool(self.cfg.reset.move_to_initial_state) else ()
        states = self._states()
        observation = self.action_layout.observation_from_states(states)
        return observation, _env_info(
            robot_states=states,
            target_actions=sent_actions,
            sent_actions=sent_actions,
            device_frames=(),
        )

    def connect(self) -> None:
        if not self.connected:
            self._initialize_states()
            self.connected = True

    def initial_state_actions(self) -> tuple[RobotAction, ...]:
        initial_state = str(self.cfg.kinematic.initial_state)
        if initial_state == "reset_targets":
            target_action = target_flat_action_from_targets(self.cfg.reset.targets, self.action_layout.segments)
            return self.action_layout.actions_from_flat(target_action, timestamp_ns=now_ns())
        if initial_state == "zero":
            return self.action_layout.actions_from_flat(self.action_layout.zero_action(), timestamp_ns=now_ns())
        raise ValueError("env.kinematic.initial_state must be reset_targets or zero")

    def move_to_actions_until_reached(
        self,
        actions: tuple[RobotAction, ...],
        log_label: str,
    ) -> tuple[RobotState, ...]:
        self._ensure_connected()
        self._apply_actions(actions)
        return self._states()

    def wait_until_actions_reached(
        self,
        actions: tuple[RobotAction, ...],
        log_label: str,
    ) -> tuple[RobotState, ...]:
        self._ensure_connected()
        self._apply_actions(actions)
        return self._states()

    def step(self, action: Sequence[float]) -> tuple[tuple[float, ...], float, bool, bool, dict]:
        self._ensure_connected()
        started = time.monotonic()
        actions = self.action_layout.actions_from_flat(action, timestamp_ns=now_ns())
        self._apply_actions(actions)
        states = self._states()
        finished = time.monotonic()
        observation = self.action_layout.observation_from_states(states)
        info = _env_info(
            robot_states=states,
            target_actions=actions,
            sent_actions=actions,
            device_frames=(),
        )
        info["timing"] = {
            "kinematic_step_s": finished - started,
        }
        return observation, 0.0, False, False, info

    def close(self) -> None:
        self.connected = False
        self._states_by_side = {}

    def _ensure_connected(self) -> None:
        if not self.connected:
            raise RuntimeError("TianjiSharpaKinematicEnv is not connected")

    def _initialize_states(self) -> None:
        self._states_by_side = {}
        timestamp_ns = now_ns()
        for side in self.action_layout.sides:
            self._states_by_side[side] = RobotState(side=side, timestamp_ns=timestamp_ns)

    def _apply_actions(self, actions: tuple[RobotAction, ...]) -> None:
        timestamp_ns = now_ns()
        for action in actions:
            current_state = self._states_by_side[action.side]
            arm_joint = current_state.arm_joint
            end_effector_joint = current_state.end_effector_joint
            if action.arm_q_rad is not None:
                arm_joint = JointState(
                    name=f"{action.side}_arm",
                    q_rad=tuple(float(value) for value in action.arm_q_rad),
                    dq_rad_s=(0.0,) * len(action.arm_q_rad),
                    timestamp_ns=timestamp_ns,
                )
            if action.end_effector_q_rad is not None:
                end_effector_joint = JointState(
                    name=f"{action.side}_end_effector",
                    q_rad=tuple(float(value) for value in action.end_effector_q_rad),
                    dq_rad_s=(0.0,) * len(action.end_effector_q_rad),
                    timestamp_ns=timestamp_ns,
                )
            self._states_by_side[action.side] = RobotState(
                side=action.side,
                timestamp_ns=timestamp_ns,
                arm_joint=arm_joint,
                end_effector_joint=end_effector_joint,
            )

    def _states(self) -> tuple[RobotState, ...]:
        return tuple(self._states_by_side[side] for side in self.action_layout.sides)


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
