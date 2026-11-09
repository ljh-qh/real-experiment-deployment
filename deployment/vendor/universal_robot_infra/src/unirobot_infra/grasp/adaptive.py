from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
import math
from pathlib import Path

import yaml

from unirobot_infra.core.types import SharpaHandState


@dataclass(frozen=True)
class AnchorTrajectory:
    side: str
    joint_order: tuple[str, ...]
    anchors_q_rad: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not self.side:
            raise ValueError("adaptive grasp side must not be empty")
        if not self.joint_order:
            raise ValueError("adaptive grasp joint_order must not be empty")
        if len(self.anchors_q_rad) < 2:
            raise ValueError("adaptive grasp requires at least 2 anchors")
        for q_rad in self.anchors_q_rad:
            if len(q_rad) != len(self.joint_order):
                raise ValueError("adaptive grasp anchor q_rad length does not match joint_order")

    @property
    def dof(self) -> int:
        return len(self.joint_order)

    @property
    def sample_count(self) -> int:
        return len(self.anchors_q_rad)

    def interpolate(self, alpha: float) -> tuple[float, ...]:
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("grasp alpha must be in [0, 1]")
        position = float(alpha) * (self.sample_count - 1)
        left_index = int(math.floor(position))
        if left_index == self.sample_count - 1:
            return self.anchors_q_rad[-1]
        right_index = left_index + 1
        t = position - left_index
        left_q = self.anchors_q_rad[left_index]
        right_q = self.anchors_q_rad[right_index]
        return tuple(float(left + t * (right - left)) for left, right in zip(left_q, right_q, strict=True))


@dataclass(frozen=True)
class AdaptiveGraspFeedbackConfig:
    enabled: bool
    torque_delta_threshold_by_group: Mapping[str, float]
    position_lag_threshold_rad_by_group: Mapping[str, float]
    min_alpha_for_stop: float


@dataclass(frozen=True)
class AdaptiveGraspStop:
    group: str
    signal: str
    value: float
    threshold: float
    alpha: float


@dataclass(frozen=True)
class AdaptiveGraspCommand:
    alpha: float
    q_rad: tuple[float, ...]
    raw_q_rad: tuple[float, ...]
    new_stops: tuple[AdaptiveGraspStop, ...]
    stops: tuple[AdaptiveGraspStop, ...]


@dataclass(frozen=True)
class AdaptiveGraspFeedbackStats:
    group: str
    max_torque_delta: float
    max_position_lag_rad: float


class AdaptiveGraspState(StrEnum):
    RELEASED = "released"
    GRASPING = "grasping"
    HOLDING = "holding"
    OBJECT_MISSING = "object_missing"
    RELEASING = "releasing"


@dataclass(frozen=True)
class AdaptiveGraspHoldConfig:
    enabled: bool
    min_torque_delta_ratio: float
    max_torque_delta_ratio: float
    missing_torque_delta_ratio: float
    close_step_rad: float
    open_step_rad: float
    max_extra_close_rad: float
    missing_target_margin_rad: float
    missing_consecutive_frames: int
    log_interval_s: float


@dataclass(frozen=True)
class AdaptiveGraspHoldGroup:
    group: str
    action: str
    torque_delta: float
    min_torque_delta: float
    max_torque_delta: float
    missing_torque_delta: float
    target_distance_rad: float


@dataclass(frozen=True)
class AdaptiveGraspHoldCommand:
    state: AdaptiveGraspState
    q_rad: tuple[float, ...]
    groups: tuple[AdaptiveGraspHoldGroup, ...]
    object_missing: bool


class AdaptiveGraspPlanner:
    def __init__(
        self,
        trajectory: AnchorTrajectory,
        group_indices: Mapping[str, Sequence[int]],
        feedback: AdaptiveGraspFeedbackConfig,
    ) -> None:
        self.trajectory = trajectory
        self.group_indices = _validated_group_indices(group_indices, trajectory.dof)
        self.feedback = _validated_feedback_config(feedback, self.group_indices)
        self._baseline_torque: tuple[float, ...] | None = None
        self._last_command_q_rad: tuple[float, ...] | None = None
        self._held_q_by_group: dict[str, tuple[float, ...]] = {}
        self._stop_by_group: dict[str, AdaptiveGraspStop] = {}
        self._max_torque_delta_by_group: dict[str, float] = {group: 0.0 for group in self.group_indices}
        self._max_position_lag_by_group: dict[str, float] = {group: 0.0 for group in self.group_indices}

    def reset(self, state: SharpaHandState) -> None:
        self._baseline_torque = _checked_values(state.joint_torque, self.trajectory.dof, "joint_torque")
        self._last_command_q_rad = _checked_values(state.joint_pos, self.trajectory.dof, "joint_pos")
        self._held_q_by_group = {}
        self._stop_by_group = {}
        self._max_torque_delta_by_group = {group: 0.0 for group in self.group_indices}
        self._max_position_lag_by_group = {group: 0.0 for group in self.group_indices}

    def command(self, alpha: float, state: SharpaHandState) -> AdaptiveGraspCommand:
        if self._baseline_torque is None or self._last_command_q_rad is None:
            raise RuntimeError("AdaptiveGraspPlanner.reset must be called before command")
        state_q = _checked_values(state.joint_pos, self.trajectory.dof, "joint_pos")
        state_torque = _checked_values(state.joint_torque, self.trajectory.dof, "joint_torque")
        new_stops = self._detect_new_stops(float(alpha), state_q, state_torque)
        raw_q = self.trajectory.interpolate(float(alpha))
        q = list(raw_q)
        for group, held_q in self._held_q_by_group.items():
            for local_index, joint_index in enumerate(self.group_indices[group]):
                q[joint_index] = held_q[local_index]
        command_q = tuple(q)
        self._last_command_q_rad = command_q
        return AdaptiveGraspCommand(
            alpha=float(alpha),
            q_rad=command_q,
            raw_q_rad=raw_q,
            new_stops=new_stops,
            stops=tuple(self._stop_by_group[group] for group in self._stop_by_group),
        )

    @property
    def stopped_groups(self) -> tuple[str, ...]:
        return tuple(self._stop_by_group)

    @property
    def all_groups_stopped(self) -> bool:
        return len(self._stop_by_group) == len(self.group_indices)

    def feedback_stats(self) -> tuple[AdaptiveGraspFeedbackStats, ...]:
        return tuple(
            AdaptiveGraspFeedbackStats(
                group=group,
                max_torque_delta=self._max_torque_delta_by_group[group],
                max_position_lag_rad=self._max_position_lag_by_group[group],
            )
            for group in self.group_indices
        )

    def _detect_new_stops(
        self,
        alpha: float,
        state_q: tuple[float, ...],
        state_torque: tuple[float, ...],
    ) -> tuple[AdaptiveGraspStop, ...]:
        new_stops = []
        for group, indices in self.group_indices.items():
            torque_delta, position_lag = self._group_feedback(group, indices, state_q, state_torque)
            if not self.feedback.enabled or alpha < float(self.feedback.min_alpha_for_stop):
                continue
            if group in self._stop_by_group:
                continue
            stop = self._group_stop(group, alpha, torque_delta, position_lag)
            if stop is None:
                continue
            self._stop_by_group[group] = stop
            self._held_q_by_group[group] = tuple(state_q[index] for index in indices)
            new_stops.append(stop)
        return tuple(new_stops)

    def _group_stop(
        self,
        group: str,
        alpha: float,
        torque_delta: float,
        position_lag: float,
    ) -> AdaptiveGraspStop | None:
        torque_delta_threshold = float(self.feedback.torque_delta_threshold_by_group[group])
        if torque_delta >= torque_delta_threshold:
            return AdaptiveGraspStop(
                group=group,
                signal="joint_torque_delta",
                value=torque_delta,
                threshold=torque_delta_threshold,
                alpha=float(alpha),
            )
        position_lag_threshold_rad = float(self.feedback.position_lag_threshold_rad_by_group[group])
        if position_lag >= position_lag_threshold_rad:
            return AdaptiveGraspStop(
                group=group,
                signal="position_lag",
                value=position_lag,
                threshold=position_lag_threshold_rad,
                alpha=float(alpha),
            )
        return None

    def _group_feedback(
        self,
        group: str,
        indices: tuple[int, ...],
        state_q: tuple[float, ...],
        state_torque: tuple[float, ...],
    ) -> tuple[float, float]:
        torque_delta = max(abs(state_torque[index] - self._baseline_torque[index]) for index in indices)
        position_lag = max(abs(self._last_command_q_rad[index] - state_q[index]) for index in indices)
        self._max_torque_delta_by_group[group] = max(self._max_torque_delta_by_group[group], torque_delta)
        self._max_position_lag_by_group[group] = max(self._max_position_lag_by_group[group], position_lag)
        return torque_delta, position_lag


class AdaptiveGraspStateMachine:
    def __init__(
        self,
        planner: AdaptiveGraspPlanner,
        hold: AdaptiveGraspHoldConfig,
    ) -> None:
        self.planner = planner
        self.hold = _validated_hold_config(hold)
        self.state = AdaptiveGraspState.RELEASED
        self._hold_q: tuple[float, ...] | None = None
        self._contact_groups: tuple[str, ...] = ()
        self._missing_frame_count = 0
        self._open_q = self.planner.trajectory.interpolate(0.0)
        self._target_q = self.planner.trajectory.interpolate(1.0)
        self._close_limit_q = _close_limit_q(self._open_q, self._target_q, self.hold.max_extra_close_rad)

    @property
    def trajectory(self) -> AnchorTrajectory:
        return self.planner.trajectory

    @property
    def stopped_groups(self) -> tuple[str, ...]:
        return self.planner.stopped_groups

    @property
    def all_groups_stopped(self) -> bool:
        return self.planner.all_groups_stopped

    def reset_for_grasp(self, state: SharpaHandState) -> None:
        self.planner.reset(state)
        self.state = AdaptiveGraspState.GRASPING
        self._hold_q = _checked_values(state.joint_pos, self.trajectory.dof, "joint_pos")
        self._contact_groups = ()
        self._missing_frame_count = 0

    def grasp_command(self, alpha: float, state: SharpaHandState) -> AdaptiveGraspCommand:
        if self.state != AdaptiveGraspState.GRASPING:
            raise RuntimeError(f"adaptive grasp state must be grasping before grasp command: {self.state}")
        command = self.planner.command(alpha, state)
        self._hold_q = command.q_rad
        return command

    def finish_grasp(self) -> AdaptiveGraspState:
        if self.state != AdaptiveGraspState.GRASPING:
            raise RuntimeError(f"adaptive grasp state must be grasping before finish: {self.state}")
        self._contact_groups = self.planner.stopped_groups
        if self.hold.enabled and not self._contact_groups:
            self.state = AdaptiveGraspState.OBJECT_MISSING
        else:
            self.state = AdaptiveGraspState.HOLDING
        return self.state

    def hold_command(self, state: SharpaHandState) -> AdaptiveGraspHoldCommand:
        if self._hold_q is None:
            raise RuntimeError("adaptive grasp hold requires a previous grasp command")
        if self.state == AdaptiveGraspState.OBJECT_MISSING:
            return AdaptiveGraspHoldCommand(
                state=self.state,
                q_rad=self._hold_q,
                groups=(),
                object_missing=True,
            )
        if self.state != AdaptiveGraspState.HOLDING:
            raise RuntimeError(f"adaptive grasp state must be holding before hold command: {self.state}")
        if not self.hold.enabled:
            return AdaptiveGraspHoldCommand(
                state=self.state,
                q_rad=self._hold_q,
                groups=(),
                object_missing=False,
            )

        state_q = _checked_values(state.joint_pos, self.trajectory.dof, "joint_pos")
        state_torque = _checked_values(state.joint_torque, self.trajectory.dof, "joint_torque")
        q = list(self._hold_q)
        groups = []
        for group, indices in self.planner.group_indices.items():
            torque_delta, target_distance_rad = self._hold_feedback(group, indices, state_q, state_torque)
            min_torque_delta = self._hold_torque_delta(group, self.hold.min_torque_delta_ratio)
            max_torque_delta = self._hold_torque_delta(group, self.hold.max_torque_delta_ratio)
            missing_torque_delta = self._hold_torque_delta(group, self.hold.missing_torque_delta_ratio)
            action = "hold"
            if torque_delta < min_torque_delta:
                if _step_group_towards(q, indices, self._close_limit_q, self.hold.close_step_rad):
                    action = "close"
            elif torque_delta > max_torque_delta:
                if _step_group_towards(q, indices, self._open_q, self.hold.open_step_rad):
                    action = "open"
            groups.append(
                AdaptiveGraspHoldGroup(
                    group=group,
                    action=action,
                    torque_delta=torque_delta,
                    min_torque_delta=min_torque_delta,
                    max_torque_delta=max_torque_delta,
                    missing_torque_delta=missing_torque_delta,
                    target_distance_rad=target_distance_rad,
                )
            )
        self._hold_q = tuple(q)
        object_missing = self._object_missing(state_q, state_torque)
        if object_missing:
            self.state = AdaptiveGraspState.OBJECT_MISSING
        return AdaptiveGraspHoldCommand(
            state=self.state,
            q_rad=self._hold_q,
            groups=tuple(groups),
            object_missing=object_missing,
        )

    def start_release(self) -> None:
        self.state = AdaptiveGraspState.RELEASING

    def mark_released(self) -> None:
        self.state = AdaptiveGraspState.RELEASED
        self._hold_q = None
        self._contact_groups = ()
        self._missing_frame_count = 0

    def _hold_feedback(
        self,
        group: str,
        indices: tuple[int, ...],
        state_q: tuple[float, ...],
        state_torque: tuple[float, ...],
    ) -> tuple[float, float]:
        if self.planner._baseline_torque is None:
            raise RuntimeError("adaptive grasp hold requires baseline torque")
        torque_delta = max(abs(state_torque[index] - self.planner._baseline_torque[index]) for index in indices)
        target_distance_rad = max(abs(self._target_q[index] - state_q[index]) for index in indices)
        return torque_delta, target_distance_rad

    def _hold_torque_delta(self, group: str, ratio: float) -> float:
        return float(self.planner.feedback.torque_delta_threshold_by_group[group]) * float(ratio)

    def _object_missing(
        self,
        state_q: tuple[float, ...],
        state_torque: tuple[float, ...],
    ) -> bool:
        if not self._contact_groups:
            self._missing_frame_count += 1
        else:
            missing_groups = 0
            for group in self._contact_groups:
                indices = self.planner.group_indices[group]
                torque_delta, target_distance_rad = self._hold_feedback(group, indices, state_q, state_torque)
                if (
                    torque_delta <= self._hold_torque_delta(group, self.hold.missing_torque_delta_ratio)
                    and target_distance_rad <= self.hold.missing_target_margin_rad
                ):
                    missing_groups += 1
            if missing_groups == len(self._contact_groups):
                self._missing_frame_count += 1
            else:
                self._missing_frame_count = 0
        return self._missing_frame_count >= self.hold.missing_consecutive_frames


def load_anchor_trajectory(anchor_dir: Path, side: str) -> AnchorTrajectory:
    paths = tuple(sorted(anchor_dir.glob("*.yaml")))
    if len(paths) < 2:
        raise ValueError(f"adaptive grasp requires at least 2 anchor files: {anchor_dir}")
    joint_order = None
    anchors = []
    for sample_index, path in enumerate(paths):
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        if data is None:
            raise ValueError(f"anchor file is empty: {path}")
        if str(data["side"]) != side:
            raise ValueError(f"anchor file side={data['side']} does not match {side}: {path}")
        if int(data["sample_count"]) != len(paths):
            raise ValueError(f"anchor sample_count does not match file count: {path}")
        anchor = data["anchor"]
        if int(anchor["sample_index"]) != sample_index:
            raise ValueError(f"anchor sample_index does not match sorted file index: {path}")
        path_joint_order = tuple(str(name) for name in data["joint_order"])
        if joint_order is None:
            joint_order = path_joint_order
        elif path_joint_order != joint_order:
            raise ValueError(f"anchor joint_order does not match first anchor: {path}")
        q_rad = tuple(float(value) for value in anchor["q_rad"])
        if len(q_rad) != len(joint_order):
            raise ValueError(f"anchor q_rad length does not match joint_order: {path}")
        anchors.append(q_rad)
    return AnchorTrajectory(
        side=side,
        joint_order=joint_order,
        anchors_q_rad=tuple(anchors),
    )


def group_indices_from_config(group_cfg) -> dict[str, tuple[int, ...]]:
    return {str(group): tuple(int(index) for index in group_cfg[group]) for group in group_cfg}


def _validated_group_indices(group_indices: Mapping[str, Sequence[int]], dof: int) -> dict[str, tuple[int, ...]]:
    groups: dict[str, tuple[int, ...]] = {}
    used_indices: set[int] = set()
    for group, indices in group_indices.items():
        group_name = str(group)
        index_tuple = tuple(int(index) for index in indices)
        if not index_tuple:
            raise ValueError(f"adaptive grasp group {group_name} must not be empty")
        for index in index_tuple:
            if not 0 <= index < dof:
                raise ValueError(f"adaptive grasp group {group_name} index out of range: {index}")
            if index in used_indices:
                raise ValueError(f"adaptive grasp joint index is assigned to multiple groups: {index}")
            used_indices.add(index)
        groups[group_name] = index_tuple
    if not groups:
        raise ValueError("adaptive grasp requires at least one stop group")
    return groups


def _validated_feedback_config(
    feedback: AdaptiveGraspFeedbackConfig,
    group_indices: Mapping[str, Sequence[int]],
) -> AdaptiveGraspFeedbackConfig:
    torque_delta_threshold_by_group = {
        group: float(feedback.torque_delta_threshold_by_group[group])
        for group in group_indices
    }
    position_lag_threshold_rad_by_group = {
        group: float(feedback.position_lag_threshold_rad_by_group[group])
        for group in group_indices
    }
    for group in group_indices:
        if torque_delta_threshold_by_group[group] < 0.0:
            raise ValueError(f"adaptive grasp torque threshold must not be negative: {group}")
        if position_lag_threshold_rad_by_group[group] < 0.0:
            raise ValueError(f"adaptive grasp position lag threshold must not be negative: {group}")
    return AdaptiveGraspFeedbackConfig(
        enabled=bool(feedback.enabled),
        torque_delta_threshold_by_group=torque_delta_threshold_by_group,
        position_lag_threshold_rad_by_group=position_lag_threshold_rad_by_group,
        min_alpha_for_stop=float(feedback.min_alpha_for_stop),
    )


def _validated_hold_config(hold: AdaptiveGraspHoldConfig) -> AdaptiveGraspHoldConfig:
    min_torque_delta_ratio = float(hold.min_torque_delta_ratio)
    max_torque_delta_ratio = float(hold.max_torque_delta_ratio)
    missing_torque_delta_ratio = float(hold.missing_torque_delta_ratio)
    close_step_rad = float(hold.close_step_rad)
    open_step_rad = float(hold.open_step_rad)
    max_extra_close_rad = float(hold.max_extra_close_rad)
    missing_target_margin_rad = float(hold.missing_target_margin_rad)
    missing_consecutive_frames = int(hold.missing_consecutive_frames)
    log_interval_s = float(hold.log_interval_s)
    if min_torque_delta_ratio < 0.0:
        raise ValueError("adaptive grasp hold min_torque_delta_ratio must not be negative")
    if max_torque_delta_ratio < min_torque_delta_ratio:
        raise ValueError("adaptive grasp hold max_torque_delta_ratio must be >= min_torque_delta_ratio")
    if missing_torque_delta_ratio < 0.0:
        raise ValueError("adaptive grasp hold missing_torque_delta_ratio must not be negative")
    if missing_torque_delta_ratio > min_torque_delta_ratio:
        raise ValueError("adaptive grasp hold missing_torque_delta_ratio must be <= min_torque_delta_ratio")
    if close_step_rad <= 0.0:
        raise ValueError("adaptive grasp hold close_step_rad must be positive")
    if open_step_rad <= 0.0:
        raise ValueError("adaptive grasp hold open_step_rad must be positive")
    if max_extra_close_rad < 0.0:
        raise ValueError("adaptive grasp hold max_extra_close_rad must not be negative")
    if missing_target_margin_rad < 0.0:
        raise ValueError("adaptive grasp hold missing_target_margin_rad must not be negative")
    if missing_consecutive_frames <= 0:
        raise ValueError("adaptive grasp hold missing_consecutive_frames must be positive")
    if log_interval_s <= 0.0:
        raise ValueError("adaptive grasp hold log_interval_s must be positive")
    return AdaptiveGraspHoldConfig(
        enabled=bool(hold.enabled),
        min_torque_delta_ratio=min_torque_delta_ratio,
        max_torque_delta_ratio=max_torque_delta_ratio,
        missing_torque_delta_ratio=missing_torque_delta_ratio,
        close_step_rad=close_step_rad,
        open_step_rad=open_step_rad,
        max_extra_close_rad=max_extra_close_rad,
        missing_target_margin_rad=missing_target_margin_rad,
        missing_consecutive_frames=missing_consecutive_frames,
        log_interval_s=log_interval_s,
    )


def _close_limit_q(
    open_q: tuple[float, ...],
    target_q: tuple[float, ...],
    max_extra_close_rad: float,
) -> tuple[float, ...]:
    limits = []
    for open_value, target_value in zip(open_q, target_q, strict=True):
        delta = target_value - open_value
        if delta > 0.0:
            limits.append(target_value + max_extra_close_rad)
        elif delta < 0.0:
            limits.append(target_value - max_extra_close_rad)
        else:
            limits.append(target_value)
    return tuple(limits)


def _step_towards(value: float, target: float, max_step: float) -> float:
    delta = target - value
    if abs(delta) <= max_step:
        return target
    return value + math.copysign(max_step, delta)


def _step_group_towards(
    q: list[float],
    indices: tuple[int, ...],
    target_q: tuple[float, ...],
    max_step: float,
) -> bool:
    changed = False
    for index in indices:
        next_value = _step_towards(q[index], target_q[index], max_step)
        if next_value != q[index]:
            changed = True
        q[index] = next_value
    return changed


def _checked_values(values: Sequence[float], dof: int, field_name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != dof:
        raise ValueError(f"adaptive grasp {field_name} length does not match dof")
    return result
