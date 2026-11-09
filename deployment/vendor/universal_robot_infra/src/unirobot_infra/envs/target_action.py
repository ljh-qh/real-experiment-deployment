from collections.abc import Sequence

from omegaconf import DictConfig

from unirobot_infra.core.types import RobotAction
from unirobot_infra.core.units import deg_to_rad, rad_to_deg
from unirobot_infra.envs.layout import ActionSegment


def target_flat_action_from_targets(targets: DictConfig, segments: Sequence[ActionSegment]) -> tuple[float, ...]:
    values: list[float] = []
    for segment in segments:
        target = targets[segment.side]
        if segment.role == "arm":
            values.extend(
                _deg_to_rad_tuple(
                    target.arm_q_deg,
                    segment.size,
                    f"env.reset.targets.{segment.side}.arm_q_deg",
                )
            )
        elif segment.role == "end_effector":
            values.extend(
                _float_tuple(
                    target.end_effector_q_rad,
                    segment.size,
                    f"env.reset.targets.{segment.side}.end_effector_q_rad",
                )
            )
        else:
            raise ValueError("unsupported robot part role")
    return tuple(values)


def target_summary(targets: DictConfig, segments: Sequence[ActionSegment]) -> str:
    parts = []
    for segment in segments:
        target = targets[segment.side]
        if segment.role == "arm":
            parts.append(f"{segment.side}:arm_deg={format_q(target.arm_q_deg)}")
        elif segment.role == "end_effector":
            parts.append(f"{segment.side}:hand_rad={format_q(target.end_effector_q_rad)}")
        else:
            raise ValueError("unsupported robot part role")
    return "; ".join(parts)


def sent_actions_summary(actions: Sequence[RobotAction]) -> str:
    parts = []
    for action in actions:
        if action.arm_q_rad is not None:
            parts.append(
                f"{action.side}:arm_deg={format_q(tuple(rad_to_deg(value) for value in action.arm_q_rad))}"
            )
        if action.end_effector_q_rad is not None:
            parts.append(f"{action.side}:hand_rad={format_q(action.end_effector_q_rad)}")
    return "; ".join(parts)


def format_q(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(value):.6f}" for value in values) + "]"


def _deg_to_rad_tuple(values: Sequence[float], length: int, field_name: str) -> tuple[float, ...]:
    result = tuple(deg_to_rad(float(value)) for value in values)
    if len(result) != length:
        raise ValueError(f"{field_name} length is not {length}")
    return result


def _float_tuple(values: Sequence[float], length: int, field_name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != length:
        raise ValueError(f"{field_name} length is not {length}")
    return result
