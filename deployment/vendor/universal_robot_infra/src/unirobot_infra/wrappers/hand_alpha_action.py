from collections.abc import Sequence
from dataclasses import dataclass

from omegaconf import DictConfig

from unirobot_infra.core.time import now_ns
from unirobot_infra.core.types import RobotAction, RobotState
from unirobot_infra.envs.layout import ActionSegment, RobotActionLayout
from unirobot_infra.wrappers.base import ActionWrapper, EnvLike


class HandAlphaActionWrapper(ActionWrapper):
    def __init__(self, env: EnvLike, cfg: DictConfig) -> None:
        super().__init__(env)
        self.cfg = cfg
        self._base_action_layout = env.action_layout
        self._hand_segments = tuple(
            segment for segment in self._base_action_layout.segments if segment.role == "end_effector"
        )
        if bool(cfg.enabled):
            if not self._hand_segments:
                raise ValueError("hand_alpha_action requires end_effector action segment")
            self._grasp_state = _configured_hand_state(
                cfg.grasp_state,
                self._hand_segments,
                "grasp_state",
            )
            self._open_state = _configured_hand_state(
                cfg.open_state,
                self._hand_segments,
                "open_state",
            )
            for segment in self._hand_segments:
                _validate_alpha_endpoint(segment.side, self._grasp_state[segment.side], self._open_state[segment.side])
            self.action_layout = _hand_alpha_action_layout(
                self._base_action_layout,
                self._open_state,
                self._grasp_state,
            )
            self.action_space = self.action_layout.action_space()
        else:
            self._grasp_state = {}
            self._open_state = {}

    def action(self, action: Sequence[float]) -> tuple[float, ...]:
        if not bool(self.cfg.enabled):
            return tuple(float(value) for value in action)
        return self.action_layout.expand_action(action)


def hand_state_alpha(
    q_rad: Sequence[float],
    open_state: Sequence[float],
    grasp_state: Sequence[float],
) -> float:
    return _hand_state_alpha(
        tuple(float(value) for value in q_rad),
        tuple(float(value) for value in open_state),
        tuple(float(value) for value in grasp_state),
    )


@dataclass(frozen=True)
class HandAlphaActionLayout(RobotActionLayout):
    base_layout: RobotActionLayout
    open_state: dict[str, tuple[float, ...]]
    grasp_state: dict[str, tuple[float, ...]]

    def expand_action(self, action: Sequence[float]) -> tuple[float, ...]:
        return _expanded_flat_action(action, self, self.base_layout, self.open_state, self.grasp_state)

    def compress_action(self, action: Sequence[float]) -> tuple[float, ...]:
        return _compressed_flat_action(action, self.base_layout, self, self.open_state, self.grasp_state)

    def actions_from_flat(self, values: Sequence[float], timestamp_ns: int | None = None) -> tuple[RobotAction, ...]:
        action_time_ns = now_ns() if timestamp_ns is None else timestamp_ns
        return self.base_layout.actions_from_flat(self.expand_action(values), timestamp_ns=action_time_ns)

    def hold_action_from_states(self, states: tuple[RobotState, ...]) -> tuple[float, ...]:
        return self.compress_action(self.base_layout.hold_action_from_states(states))

    def overlay_actions(
        self,
        base_values: Sequence[float],
        actions: tuple[RobotAction, ...],
    ) -> tuple[float, ...]:
        output = list(_validated_flat_action(base_values, self.size))
        action_by_side = {action.side: action for action in actions}
        base_segment_by_role = {
            (segment.side, segment.role): segment
            for segment in self.base_layout.segments
        }
        for segment in self.segments:
            if segment.side not in action_by_side:
                continue
            action = action_by_side[segment.side]
            if segment.role == "arm":
                q_rad = action.arm_q_rad
                if q_rad is not None:
                    if len(q_rad) != segment.size:
                        raise ValueError("RobotAction arm q_rad length does not match layout segment")
                    output[segment.start : segment.stop] = [float(value) for value in q_rad]
            elif segment.role == "end_effector":
                q_rad = action.end_effector_q_rad
                if q_rad is not None:
                    base_segment = base_segment_by_role[(segment.side, segment.role)]
                    if len(q_rad) != base_segment.size:
                        raise ValueError("RobotAction end_effector q_rad length does not match base layout segment")
                    output[segment.start] = _hand_state_alpha(
                        tuple(float(value) for value in q_rad),
                        self.open_state[segment.side],
                        self.grasp_state[segment.side],
                    )
            else:
                raise ValueError("unsupported robot part role")
        return tuple(output)


def _hand_alpha_action_layout(
    base_layout: RobotActionLayout,
    open_state: dict[str, tuple[float, ...]],
    grasp_state: dict[str, tuple[float, ...]],
) -> HandAlphaActionLayout:
    segments = []
    cursor = 0
    for segment in base_layout.segments:
        if segment.role == "end_effector":
            size = 1
            joint_names = (f"{segment.side}_hand_alpha",)
            low_rad = (0.0,)
            high_rad = (1.0,)
        else:
            size = segment.size
            joint_names = segment.joint_names
            low_rad = segment.low_rad
            high_rad = segment.high_rad
        segments.append(
            ActionSegment(
                side=segment.side,
                role=segment.role,
                start=cursor,
                stop=cursor + size,
                joint_names=joint_names,
                low_rad=low_rad,
                high_rad=high_rad,
            )
        )
        cursor += size
    return HandAlphaActionLayout(
        segments=tuple(segments),
        base_layout=base_layout,
        open_state=open_state,
        grasp_state=grasp_state,
    )


def _expanded_flat_action(
    action: Sequence[float],
    compressed_layout: RobotActionLayout,
    base_layout: RobotActionLayout,
    open_state: dict[str, tuple[float, ...]],
    grasp_state: dict[str, tuple[float, ...]],
) -> tuple[float, ...]:
    compressed = _validated_flat_action(action, compressed_layout.size)
    expanded = [0.0] * base_layout.size
    compressed_segment_by_role = {
        (segment.side, segment.role): segment
        for segment in compressed_layout.segments
    }
    for base_segment in base_layout.segments:
        compressed_segment = compressed_segment_by_role[(base_segment.side, base_segment.role)]
        if base_segment.role == "end_effector":
            alpha = compressed[compressed_segment.start]
            if alpha < 0.0 or alpha > 1.0:
                raise ValueError(f"hand_alpha_action alpha for side={base_segment.side} must be in [0, 1]")
            expanded[base_segment.start : base_segment.stop] = _interpolate_hand_state(
                alpha,
                open_state[base_segment.side],
                grasp_state[base_segment.side],
            )
        else:
            expanded[base_segment.start : base_segment.stop] = compressed[compressed_segment.start : compressed_segment.stop]
    return tuple(expanded)


def _compressed_flat_action(
    action: Sequence[float],
    base_layout: RobotActionLayout,
    compressed_layout: RobotActionLayout,
    open_state: dict[str, tuple[float, ...]],
    grasp_state: dict[str, tuple[float, ...]],
) -> tuple[float, ...]:
    expanded = _validated_flat_action(action, base_layout.size)
    compressed = [0.0] * compressed_layout.size
    compressed_segment_by_role = {
        (segment.side, segment.role): segment
        for segment in compressed_layout.segments
    }
    for base_segment in base_layout.segments:
        compressed_segment = compressed_segment_by_role[(base_segment.side, base_segment.role)]
        if base_segment.role == "end_effector":
            hand_q = expanded[base_segment.start : base_segment.stop]
            compressed[compressed_segment.start] = _hand_state_alpha(
                hand_q,
                open_state[base_segment.side],
                grasp_state[base_segment.side],
            )
        else:
            compressed[compressed_segment.start : compressed_segment.stop] = expanded[base_segment.start : base_segment.stop]
    return tuple(compressed)


def _validated_flat_action(action: Sequence[float], expected_size: int) -> tuple[float, ...]:
    if len(action) != expected_size:
        raise ValueError("flat action length does not match layout")
    return tuple(float(value) for value in action)


def _configured_hand_state(
    cfg_value,
    segments: tuple[ActionSegment, ...],
    name: str,
) -> dict[str, tuple[float, ...]]:
    state_by_side: dict[str, tuple[float, ...]] = {}
    for segment in segments:
        side = segment.side
        if side not in cfg_value:
            raise ValueError(f"{name}.{side} is required")
        q_rad = tuple(float(value) for value in cfg_value[side])
        if len(q_rad) != segment.size:
            raise ValueError(f"{name}.{side} length {len(q_rad)} does not match hand action size {segment.size}")
        state_by_side[side] = q_rad
    return state_by_side


def _validate_alpha_endpoint(side: str, grasp_state: tuple[float, ...], open_state: tuple[float, ...]) -> None:
    if len(grasp_state) != len(open_state):
        raise ValueError(f"{side} hand alpha endpoints have different lengths")
    if len(grasp_state) == 0:
        raise ValueError(f"{side} hand alpha endpoint must not be empty")
    if _hand_alpha_denominator(open_state, grasp_state) <= 0.0:
        raise ValueError(f"{side} hand alpha endpoints are identical")


def _interpolate_hand_state(
    alpha: float,
    open_state: tuple[float, ...],
    grasp_state: tuple[float, ...],
) -> tuple[float, ...]:
    return tuple(
        float(open_value + alpha * (grasp_value - open_value))
        for open_value, grasp_value in zip(open_state, grasp_state, strict=True)
    )


def _hand_state_alpha(
    q_rad: tuple[float, ...],
    open_state: tuple[float, ...],
    grasp_state: tuple[float, ...],
) -> float:
    if len(q_rad) != len(open_state):
        raise ValueError("hand q_rad length does not match alpha endpoint")
    denominator = _hand_alpha_denominator(open_state, grasp_state)
    alpha = (
        sum(
            (value - open_value) * (grasp_value - open_value)
            for value, open_value, grasp_value in zip(q_rad, open_state, grasp_state, strict=True)
        )
        / denominator
    )
    return float(min(1.0, max(0.0, alpha)))


def _hand_alpha_denominator(open_state: tuple[float, ...], grasp_state: tuple[float, ...]) -> float:
    return sum(
        (grasp_value - open_value) * (grasp_value - open_value)
        for open_value, grasp_value in zip(open_state, grasp_state, strict=True)
    )
