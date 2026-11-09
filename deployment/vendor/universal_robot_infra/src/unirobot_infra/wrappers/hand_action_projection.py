from collections.abc import Sequence

from omegaconf import DictConfig

from unirobot_infra.envs.layout import ActionSegment
from unirobot_infra.wrappers.base import ActionWrapper, EnvLike


class HandActionProjectionWrapper(ActionWrapper):
    def __init__(self, env: EnvLike, cfg: DictConfig) -> None:
        super().__init__(env)
        self.cfg = cfg
        self._segments = tuple(segment for segment in self.action_layout.segments if segment.role == "end_effector")
        if bool(cfg.enabled):
            if not self._segments:
                raise ValueError("hand_action_projection requires end_effector action segment")
            self._grasp_state = _configured_hand_state(
                cfg.grasp_state,
                self._segments,
                "grasp_state",
            )
            self._open_state = _configured_hand_state(
                cfg.open_state,
                self._segments,
                "open_state",
            )
            for segment in self._segments:
                _validate_projection_endpoint(
                    segment.side,
                    self._grasp_state[segment.side],
                    self._open_state[segment.side],
                )
        else:
            self._grasp_state = {}
            self._open_state = {}

    def action(self, action: Sequence[float]) -> tuple[float, ...]:
        if not bool(self.cfg.enabled):
            return tuple(float(value) for value in action)
        return self._project_action(action)

    def _project_action(self, action: Sequence[float]) -> tuple[float, ...]:
        projected = list(_validated_flat_action(action, self.action_layout.size))
        for segment in self._segments:
            hand_q = tuple(projected[segment.start : segment.stop])
            projected[segment.start : segment.stop] = _project_q_to_hand_axis(
                hand_q,
                self._grasp_state[segment.side],
                self._open_state[segment.side],
            )
        return tuple(projected)


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


def _validate_projection_endpoint(side: str, grasp_state: tuple[float, ...], open_state: tuple[float, ...]) -> None:
    if len(grasp_state) != len(open_state):
        raise ValueError(f"{side} projection endpoints have different lengths")
    if len(grasp_state) == 0:
        raise ValueError(f"{side} projection endpoint must not be empty")
    if _projection_denominator(grasp_state, open_state) <= 0.0:
        raise ValueError(f"{side} projection endpoints are identical")


def _project_q_to_hand_axis(
    q_rad: tuple[float, ...],
    grasp_state: tuple[float, ...],
    open_state: tuple[float, ...],
) -> tuple[float, ...]:
    if len(q_rad) != len(grasp_state):
        raise ValueError("hand q_rad length does not match projection endpoint")
    direction = tuple(open_value - grasp_value for grasp_value, open_value in zip(grasp_state, open_state, strict=True))
    denominator = sum(value * value for value in direction)
    alpha = (
        sum((value - grasp_value) * delta for value, grasp_value, delta in zip(q_rad, grasp_state, direction, strict=True))
        / denominator
    )
    alpha = min(1.0, max(0.0, alpha))
    return tuple(float(grasp_value + alpha * delta) for grasp_value, delta in zip(grasp_state, direction, strict=True))


def _projection_denominator(grasp_state: tuple[float, ...], open_state: tuple[float, ...]) -> float:
    return sum(
        (open_value - grasp_value) * (open_value - grasp_value)
        for grasp_value, open_value in zip(grasp_state, open_state, strict=True)
    )
