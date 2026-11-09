from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from omegaconf import DictConfig

from unirobot_infra.core.types import RobotAction, RobotState
from unirobot_infra.envs.layout import ActionSegment, RobotActionLayout
from unirobot_infra.robots.arm.tianji_tcp_pose import TianjiTcpPoseMapper, apply_delta_tcp_pose_world


_DELTA_EE_JOINT_NAMES = (
    "delta_ee_world_dx_m",
    "delta_ee_world_dy_m",
    "delta_ee_world_dz_m",
    "delta_ee_world_r6d_c1_x",
    "delta_ee_world_r6d_c1_y",
    "delta_ee_world_r6d_c1_z",
    "delta_ee_world_r6d_c2_x",
    "delta_ee_world_r6d_c2_y",
    "delta_ee_world_r6d_c2_z",
)


@dataclass(frozen=True)
class DeltaEeTrace:
    base_pose_world: tuple[float, ...]
    desired_pose_world: tuple[float, ...]
    target_pose_world: tuple[float, ...]
    delta_xyz_m: tuple[float, float, float]


@dataclass(frozen=True)
class DeltaEeExpansion:
    action: tuple[float, ...]
    trace_by_side: dict[str, DeltaEeTrace]
    target_q_rad_by_side: dict[str, tuple[float, ...]]


class DeltaEeActionExpander:
    def __init__(self, base_action_layout: RobotActionLayout, cfg: DictConfig, arm_cfg: DictConfig) -> None:
        if str(cfg.frame) != "robot_world":
            raise ValueError("tianji_delta_ee_action.frame must be robot_world")
        self.cfg = cfg
        self.base_action_layout = base_action_layout
        self.arm_segments = tuple(segment for segment in base_action_layout.segments if segment.role == "arm")
        if not self.arm_segments:
            raise ValueError("tianji_delta_ee_action requires arm action segment")
        self.mappers = {
            segment.side: TianjiTcpPoseMapper.from_config(
                arm_cfg,
                segment.side,
                max_joint_step_deg=float(cfg.ik.max_joint_step_deg),
                joint_limit_mode=str(cfg.ik.joint_limit_mode),
            )
            for segment in self.arm_segments
        }
        self.action_layout = delta_ee_action_layout(base_action_layout, cfg)
        self._compressed_segment_by_role = {
            (segment.side, segment.role): segment
            for segment in self.action_layout.segments
        }

    def expand(
        self,
        action: Sequence[float],
        states: tuple[RobotState, ...],
        current_q_rad_by_side: dict[str, tuple[float, ...]] | None = None,
    ) -> DeltaEeExpansion:
        compressed = validated_flat_action(action, self.action_layout.size)
        state_by_side = {state.side: state for state in states}
        expanded = [0.0] * self.base_action_layout.size
        trace_by_side: dict[str, DeltaEeTrace] = {}
        target_q_rad_by_side: dict[str, tuple[float, ...]] = {}
        for base_segment in self.base_action_layout.segments:
            compressed_segment = self._compressed_segment_by_role[(base_segment.side, base_segment.role)]
            if base_segment.role == "arm":
                side = base_segment.side
                delta_world = compressed[compressed_segment.start : compressed_segment.stop]
                validate_delta_ee_world_action(side, delta_world, self.cfg)
                current_q_rad = (
                    current_q_rad_by_side[side]
                    if current_q_rad_by_side is not None
                    else arm_q_rad(state_by_side[side])
                )
                mapper = self.mappers[side]
                base_pose_world = mapper.tcp_pose_world_from_q_rad(current_q_rad)
                desired_pose_world = apply_delta_tcp_pose_world(base_pose_world, delta_world)
                target_q_rad = mapper.q_rad_from_delta_world(
                    current_q_rad=current_q_rad,
                    delta_world=delta_world,
                    last_q_rad=current_q_rad,
                )
                target_pose_world = mapper.tcp_pose_world_from_q_rad(target_q_rad)
                target_q_rad = tuple(float(value) for value in target_q_rad)
                trace_by_side[side] = DeltaEeTrace(
                    base_pose_world=pose_flat(base_pose_world),
                    desired_pose_world=pose_flat(desired_pose_world),
                    target_pose_world=pose_flat(target_pose_world),
                    delta_xyz_m=(float(delta_world[0]), float(delta_world[1]), float(delta_world[2])),
                )
                target_q_rad_by_side[side] = target_q_rad
                expanded[base_segment.start : base_segment.stop] = target_q_rad
            else:
                expanded[base_segment.start : base_segment.stop] = compressed[
                    compressed_segment.start : compressed_segment.stop
                ]
        return DeltaEeExpansion(
            action=tuple(expanded),
            trace_by_side=trace_by_side,
            target_q_rad_by_side=target_q_rad_by_side,
        )


class DeltaEeCommandController:
    def __init__(self, env, expander: DeltaEeActionExpander, initial_states: tuple[RobotState, ...]) -> None:
        self.env = env
        self.expander = expander
        self.latest_states = tuple(initial_states)
        self._chunk_q_rad_by_side: dict[str, tuple[float, ...]] = {}
        self._last_target_actions: tuple[RobotAction, ...] | None = None

    def send_action(self, action: Sequence[float]) -> tuple[tuple[float, ...], float, bool, bool, dict]:
        expansion = self.expander.expand(action, self.latest_states)
        observation, reward, terminated, truncated, info = self.env.step(expansion.action)
        self._update_after_step(info, expansion)
        return observation, reward, terminated, truncated, info

    def begin_chunk(self) -> None:
        state_by_side = {state.side: state for state in self.latest_states}
        self._chunk_q_rad_by_side = {
            segment.side: arm_q_rad(state_by_side[segment.side])
            for segment in self.expander.arm_segments
        }

    def send_chunk_action(self, action: Sequence[float]) -> tuple[tuple[float, ...], float, bool, bool, dict]:
        if not self._chunk_q_rad_by_side:
            raise RuntimeError("delta ee command chunk requires begin_chunk before send_chunk_action")
        expansion = self.expander.expand(
            action,
            self.latest_states,
            current_q_rad_by_side=self._chunk_q_rad_by_side,
        )
        observation, reward, terminated, truncated, info = self.env.step(expansion.action)
        self._chunk_q_rad_by_side = dict(expansion.target_q_rad_by_side)
        self._update_after_step(info, expansion)
        return observation, reward, terminated, truncated, info

    def wait_until_current_target_reached(self, log_label: str) -> tuple[RobotState, ...]:
        if self._last_target_actions is None:
            raise RuntimeError("delta ee command controller has no target action to wait for")
        states = tuple(self.env.wait_until_actions_reached(self._last_target_actions, log_label))
        self.latest_states = states
        return states

    def end_chunk(self) -> None:
        self._chunk_q_rad_by_side = {}

    def _update_after_step(self, info: dict, expansion: DeltaEeExpansion) -> None:
        info["delta_ee_trace"] = expansion.trace_by_side
        self._last_target_actions = tuple(info["target_actions"])
        self.latest_states = tuple(info["robot_states"])


@dataclass(frozen=True)
class TianjiDeltaEeActionLayout(RobotActionLayout):
    base_layout: RobotActionLayout

    def actions_from_flat(self, values: Sequence[float], timestamp_ns: int | None = None) -> tuple[RobotAction, ...]:
        raise RuntimeError("tianji_delta_ee_action requires current robot state; use env.step(action)")

    def hold_action_from_states(self, states: tuple[RobotState, ...]) -> tuple[float, ...]:
        base_hold_action = self.base_layout.hold_action_from_states(states)
        return self.compress_hold_action(base_hold_action)

    def compress_hold_action(self, base_action: Sequence[float]) -> tuple[float, ...]:
        base_values = validated_flat_action(base_action, self.base_layout.size)
        compressed = [0.0] * self.size
        compressed_segment_by_role = {
            (segment.side, segment.role): segment
            for segment in self.segments
        }
        for base_segment in self.base_layout.segments:
            compressed_segment = compressed_segment_by_role[(base_segment.side, base_segment.role)]
            if base_segment.role == "arm":
                compressed[compressed_segment.start : compressed_segment.stop] = [0.0] * compressed_segment.size
            else:
                compressed[compressed_segment.start : compressed_segment.stop] = base_values[
                    base_segment.start : base_segment.stop
                ]
        return tuple(compressed)

    def overlay_actions(
        self,
        base_values: Sequence[float],
        actions: tuple[RobotAction, ...],
    ) -> tuple[float, ...]:
        output = list(validated_flat_action(base_values, self.size))
        action_by_side = {action.side: action for action in actions}
        for segment in self.segments:
            if segment.side not in action_by_side:
                continue
            action = action_by_side[segment.side]
            if segment.role == "arm" and action.arm_q_rad is not None:
                raise RuntimeError("tianji_delta_ee_action cannot overlay absolute arm q_rad without robot state")
            if segment.role == "end_effector" and action.end_effector_q_rad is not None:
                if len(action.end_effector_q_rad) != segment.size:
                    raise ValueError("RobotAction end_effector q_rad length does not match layout segment")
                output[segment.start : segment.stop] = [float(value) for value in action.end_effector_q_rad]
        return tuple(output)


def delta_ee_action_layout(base_layout: RobotActionLayout, cfg: DictConfig) -> TianjiDeltaEeActionLayout:
    translation_limit_m = float(cfg.translation_limit_m)
    rotation_6d_limit = float(cfg.rotation_6d_limit)
    if translation_limit_m <= 0.0:
        raise ValueError("tianji_delta_ee_action.translation_limit_m must be positive")
    if rotation_6d_limit <= 0.0:
        raise ValueError("tianji_delta_ee_action.rotation_6d_limit must be positive")
    segments = []
    cursor = 0
    for segment in base_layout.segments:
        if segment.role == "arm":
            size = 9
            joint_names = tuple(f"{segment.side}_{name}" for name in _DELTA_EE_JOINT_NAMES)
            low_rad = (-translation_limit_m, -translation_limit_m, -translation_limit_m)
            low_rad = low_rad + (-rotation_6d_limit,) * 6
            high_rad = (translation_limit_m, translation_limit_m, translation_limit_m)
            high_rad = high_rad + (rotation_6d_limit,) * 6
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
    return TianjiDeltaEeActionLayout(segments=tuple(segments), base_layout=base_layout)


def validated_flat_action(action: Sequence[float], expected_size: int) -> tuple[float, ...]:
    if len(action) != expected_size:
        raise ValueError("flat action length does not match layout")
    return tuple(float(value) for value in action)


def validate_delta_ee_world_action(side: str, delta_world: Sequence[float], cfg: DictConfig) -> None:
    delta = tuple(float(value) for value in delta_world)
    if len(delta) != 9:
        raise ValueError("delta ee action segment must have length 9")
    translation_limit_m = float(cfg.translation_limit_m)
    rotation_6d_limit = float(cfg.rotation_6d_limit)
    if max(abs(value) for value in delta[:3]) > translation_limit_m:
        raise ValueError(f"tianji_delta_ee_action translation for side={side} exceeds limit")
    if max(abs(value) for value in delta[3:]) > rotation_6d_limit:
        raise ValueError(f"tianji_delta_ee_action rotation for side={side} exceeds limit")


def arm_q_rad(state: RobotState) -> tuple[float, ...]:
    if state.arm_joint is not None:
        return tuple(float(value) for value in state.arm_joint.q_rad)
    if state.tianji is not None:
        return tuple(float(value) for value in state.tianji.joint_pos)
    raise RuntimeError(f"RobotState missing arm joint state for side {state.side}")


def pose_flat(pose_world_m: Sequence[Sequence[float]]) -> tuple[float, ...]:
    return tuple(float(pose_world_m[row][col]) for row in range(4) for col in range(4))
