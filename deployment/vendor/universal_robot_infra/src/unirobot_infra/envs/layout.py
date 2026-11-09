from collections.abc import Sequence
from dataclasses import dataclass

from omegaconf import DictConfig

from unirobot_infra.core.time import now_ns
from unirobot_infra.core.types import JointState, RobotAction, RobotPart, RobotState, Side
from unirobot_infra.core.units import deg_to_rad
from unirobot_infra.envs.spaces import BoxSpace


DEFAULT_OBSERVATION_STATES = ("joint_pos", "joint_vel")
SUPPORTED_OBSERVATION_STATES = frozenset(
    (
        "joint_pos",
        "joint_vel",
        "joint_torque",
        "joint_pos_external",
        "joint_pos_commanded",
        "joint_current",
        "joint_temperature",
        "est_joint_friction",
        "est_joint_friction_derivative",
        "est_joint_force",
    )
)


@dataclass(frozen=True)
class ActionSegment:
    side: Side
    role: RobotPart
    start: int
    stop: int
    joint_names: tuple[str, ...]
    low_rad: tuple[float, ...]
    high_rad: tuple[float, ...]

    @property
    def size(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class ObservationItem:
    side: Side
    role: RobotPart
    state_name: str
    token: str


@dataclass(frozen=True)
class RobotActionLayout:
    segments: tuple[ActionSegment, ...]

    @classmethod
    def from_robot_config(cls, robot_cfg: DictConfig) -> "RobotActionLayout":
        segments = []
        cursor = 0
        for side in robot_cfg.sides:
            for role in robot_cfg.sides[side].parts:
                joint_names, low_rad, high_rad = _joint_bounds_rad(robot_cfg, role)
                stop = cursor + len(joint_names)
                segments.append(
                    ActionSegment(
                        side=side,
                        role=role,
                        start=cursor,
                        stop=stop,
                        joint_names=joint_names,
                        low_rad=low_rad,
                        high_rad=high_rad,
                    )
                )
                cursor = stop
        return cls(segments=tuple(segments))

    @property
    def size(self) -> int:
        if not self.segments:
            return 0
        return self.segments[-1].stop

    @property
    def sides(self) -> tuple[Side, ...]:
        sides = []
        for segment in self.segments:
            if segment.side not in sides:
                sides.append(segment.side)
        return tuple(sides)

    def action_space(self) -> BoxSpace:
        return BoxSpace(
            low=tuple(value for segment in self.segments for value in segment.low_rad),
            high=tuple(value for segment in self.segments for value in segment.high_rad),
        )

    def observation_space(self, state_names: Sequence[str] = DEFAULT_OBSERVATION_STATES) -> BoxSpace:
        low: list[float] = []
        high: list[float] = []
        for state_name in _validated_state_names(state_names):
            if state_name in {"joint_pos", "joint_pos_external", "joint_pos_commanded"}:
                low.extend(value for segment in self.segments for value in segment.low_rad)
                high.extend(value for segment in self.segments for value in segment.high_rad)
            elif state_name in SUPPORTED_OBSERVATION_STATES:
                low.extend((float("-inf"),) * self.size)
                high.extend((float("inf"),) * self.size)
            else:
                raise ValueError("unsupported state observation")
        return BoxSpace(
            low=tuple(low),
            high=tuple(high),
        )

    def observation_space_from_items(self, items: Sequence[ObservationItem]) -> BoxSpace:
        low: list[float] = []
        high: list[float] = []
        for item in items:
            segment = self._segment(item.side, item.role)
            state_name = _validated_state_name(item.state_name)
            if state_name in {"joint_pos", "joint_pos_external", "joint_pos_commanded"}:
                low.extend(segment.low_rad)
                high.extend(segment.high_rad)
            elif state_name in SUPPORTED_OBSERVATION_STATES:
                low.extend((float("-inf"),) * segment.size)
                high.extend((float("inf"),) * segment.size)
            else:
                raise ValueError("unsupported state observation")
        return BoxSpace(low=tuple(low), high=tuple(high))

    def has_segment(self, side: Side, role: RobotPart) -> bool:
        return any(segment.side == side and segment.role == role for segment in self.segments)

    def zero_action(self) -> tuple[float, ...]:
        return (0.0,) * self.size

    def actions_from_flat(self, values: Sequence[float], timestamp_ns: int | None = None) -> tuple[RobotAction, ...]:
        value_tuple = self._validate_values(values)
        action_time_ns = now_ns() if timestamp_ns is None else timestamp_ns
        by_side: dict[str, dict[str, tuple[float, ...] | None]] = {}
        for side in self.sides:
            by_side[side] = {"arm": None, "end_effector": None}
        for segment in self.segments:
            by_side[segment.side][segment.role] = value_tuple[segment.start : segment.stop]
        return tuple(
            RobotAction(
                side=side,
                timestamp_ns=action_time_ns,
                arm_q_rad=by_side[side]["arm"],
                end_effector_q_rad=by_side[side]["end_effector"],
            )
            for side in self.sides
        )

    def overlay_actions(
        self,
        base_values: Sequence[float],
        actions: tuple[RobotAction, ...],
    ) -> tuple[float, ...]:
        output = list(self._validate_values(base_values))
        action_by_side = {action.side: action for action in actions}
        for segment in self.segments:
            if segment.side in action_by_side:
                q_rad = _action_segment_q_rad(action_by_side[segment.side], segment.role)
                if q_rad is not None:
                    if len(q_rad) != segment.size:
                        raise ValueError("RobotAction q_rad length does not match layout segment")
                    output[segment.start : segment.stop] = [float(value) for value in q_rad]
        return tuple(output)

    def observation_from_states(
        self,
        states: tuple[RobotState, ...],
        state_names: Sequence[str] = DEFAULT_OBSERVATION_STATES,
    ) -> tuple[float, ...]:
        state_by_side = {state.side: state for state in states}
        values: list[float] = []
        for state_name in _validated_state_names(state_names):
            for segment in self.segments:
                part_state = _state_segment_part_state(state_by_side[segment.side], segment.role)
                values.extend(_state_segment_values(part_state, segment, state_name))
        return tuple(values)

    def observation_from_state_items(
        self,
        states: tuple[RobotState, ...],
        items: Sequence[ObservationItem],
    ) -> tuple[float, ...]:
        state_by_side = {state.side: state for state in states}
        values: list[float] = []
        for item in items:
            segment = self._segment(item.side, item.role)
            state = state_by_side[item.side]
            part_state = _state_segment_part_state(state, item.role)
            values.extend(_state_segment_values(part_state, segment, item.state_name))
        return tuple(values)

    def hold_action_from_states(self, states: tuple[RobotState, ...]) -> tuple[float, ...]:
        return self.observation_from_states(states, ("joint_pos",))

    def hold_action_from_observation(
        self,
        observation: Sequence[float],
        state_names: Sequence[str] = DEFAULT_OBSERVATION_STATES,
    ) -> tuple[float, ...]:
        names = _validated_state_names(state_names)
        if len(observation) != self.size * len(names):
            raise ValueError("observation length does not match layout")
        for state_index, state_name in enumerate(names):
            if state_name == "joint_pos":
                start = state_index * self.size
                return tuple(float(observation[start + index]) for index in range(self.size))
        raise ValueError("observation does not include joint_pos")

    def _validate_values(self, values: Sequence[float]) -> tuple[float, ...]:
        if len(values) != self.size:
            raise ValueError("flat action length does not match layout")
        return tuple(float(value) for value in values)

    def _segment(self, side: Side, role: RobotPart) -> ActionSegment:
        for segment in self.segments:
            if segment.side == side and segment.role == role:
                return segment
        raise ValueError(f"robot config missing segment {side}_{role}")


def _joint_bounds_rad(
    robot_cfg: DictConfig,
    role: str,
) -> tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
    if role == "arm":
        joints = robot_cfg.hardware.arm.joints
        return (
            tuple(joint.name for joint in joints),
            tuple(deg_to_rad(float(joint.min_deg)) for joint in joints),
            tuple(deg_to_rad(float(joint.max_deg)) for joint in joints),
        )
    if role == "end_effector":
        joints = robot_cfg.hardware.end_effector.joints
        return (
            tuple(joint.name for joint in joints),
            tuple(float(joint.min_rad) for joint in joints),
            tuple(float(joint.max_rad) for joint in joints),
        )
    raise ValueError("unsupported robot part role")


def _action_segment_q_rad(action: RobotAction, role: RobotPart) -> tuple[float, ...] | None:
    if role == "arm":
        return action.arm_q_rad
    if role == "end_effector":
        return action.end_effector_q_rad
    raise ValueError("unsupported robot part role")


def _state_segment_part_state(state: RobotState, role: RobotPart):
    if role == "arm":
        if state.tianji is not None:
            return state.tianji
        if state.arm_joint is None:
            raise ValueError("RobotState missing arm_joint")
        return state.arm_joint
    if role == "end_effector":
        if state.sharpa is not None:
            return state.sharpa
        if state.end_effector_joint is None:
            raise ValueError("RobotState missing end_effector_joint")
        return state.end_effector_joint
    raise ValueError("unsupported robot part role")


def _state_segment_values(part_state, segment: ActionSegment, state_name: str) -> tuple[float, ...]:
    if isinstance(part_state, JointState):
        if state_name == "joint_pos":
            values = part_state.q_rad
        elif state_name == "joint_vel":
            values = part_state.dq_rad_s
        else:
            raise ValueError(f"JointState does not support observation {state_name}")
    else:
        values = part_state.observation_values(state_name)
    if len(values) != segment.size:
        raise ValueError(f"RobotState {state_name} length does not match layout segment")
    return tuple(float(value) for value in values)


def _validated_state_names(state_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(state_name) for state_name in state_names)
    if not names:
        raise ValueError("state observation names must not be empty")
    for state_name in names:
        _validated_state_name(state_name)
    return names


def _validated_state_name(state_name: str) -> str:
    if state_name not in SUPPORTED_OBSERVATION_STATES:
        raise ValueError(f"unsupported state observation: {state_name}")
    return state_name
