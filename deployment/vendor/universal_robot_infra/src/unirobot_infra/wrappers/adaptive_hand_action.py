from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig

from unirobot_infra.core.types import RobotState, SharpaHandState
from unirobot_infra.envs.layout import ActionSegment
from unirobot_infra.grasp.adaptive import (
    AdaptiveGraspFeedbackConfig,
    AdaptiveGraspHoldConfig,
    AdaptiveGraspPlanner,
    AdaptiveGraspState,
    AdaptiveGraspStateMachine,
    group_indices_from_config,
    load_anchor_trajectory,
)
from unirobot_infra.wrappers.base import EnvLike, EnvWrapper
from unirobot_infra.wrappers.hand_alpha_action import HandAlphaActionLayout, HandAlphaActionWrapper


class AdaptiveHandActionWrapper(EnvWrapper):
    """Apply adaptive grasp feedback while preserving hand-alpha actions."""

    def __init__(self, env: EnvLike, cfg: DictConfig) -> None:
        super().__init__(env)
        self.cfg = cfg
        self._enabled = bool(cfg.enabled)
        self._latest_robot_states: tuple[RobotState, ...] | None = None
        self._controllers: dict[str, _AdaptiveHandSideController] = {}
        self._base_hand_segments: dict[str, ActionSegment] = {}
        if not self._enabled:
            return
        if not isinstance(env, HandAlphaActionWrapper) or not isinstance(env.action_layout, HandAlphaActionLayout):
            raise ValueError("adaptive_hand_action requires enabled hand_alpha_action wrapper immediately inside it")
        self._hand_alpha_env = env
        self._compressed_layout = env.action_layout
        self._base_layout = env.action_layout.base_layout
        self.action_layout = env.action_layout
        self.action_space = env.action_space
        self._alpha_segments = {
            segment.side: segment
            for segment in self._compressed_layout.segments
            if segment.role == "end_effector"
        }
        self._base_hand_segments = {
            segment.side: segment
            for segment in self._base_layout.segments
            if segment.role == "end_effector"
        }
        if set(self._alpha_segments) != set(self._base_hand_segments):
            raise ValueError("adaptive_hand_action hand-alpha sides do not match base hand sides")
        self._controllers = {
            side: _AdaptiveHandSideController.from_config(side, cfg)
            for side in self._alpha_segments
        }

    def reset(self) -> tuple[tuple[float, ...], dict]:
        observation, info = self.env.reset()
        if self._enabled:
            self._latest_robot_states = tuple(info["robot_states"])
            for controller in self._controllers.values():
                controller.mark_released()
        return observation, info

    def step(self, action: Sequence[float]) -> tuple[tuple[float, ...], float, bool, bool, dict]:
        if not self._enabled:
            return self.env.step(action)
        if self._latest_robot_states is None:
            raise RuntimeError("adaptive_hand_action requires reset before step")
        base_action = list(self._compressed_layout.expand_action(action))
        state_by_side = {state.side: state for state in self._latest_robot_states}
        summaries = {}
        for side, alpha in self._alpha_by_side(action).items():
            command = self._controllers[side].command(alpha, _sharpa_state(state_by_side[side]))
            segment = self._base_hand_segments[side]
            base_action[segment.start : segment.stop] = command.q_rad
            summaries[side] = command.summary
        observation, reward, terminated, truncated, info = self._hand_alpha_env.env.step(tuple(base_action))
        self._latest_robot_states = tuple(info["robot_states"])
        info = dict(info)
        info["adaptive_hand_action"] = summaries
        return observation, reward, terminated, truncated, info

    def _alpha_by_side(self, action: Sequence[float]) -> dict[str, float]:
        values = tuple(float(value) for value in action)
        if len(values) != self.action_layout.size:
            raise ValueError("adaptive_hand_action action length does not match action layout")
        output = {}
        for side, segment in self._alpha_segments.items():
            alpha = values[segment.start]
            if alpha < 0.0 or alpha > 1.0:
                raise ValueError(f"adaptive_hand_action alpha for side={side} must be in [0, 1]")
            output[side] = alpha
        return output


@dataclass(frozen=True)
class _AdaptiveHandCommand:
    q_rad: tuple[float, ...]
    summary: dict[str, object]


class _AdaptiveHandSideController:
    def __init__(
        self,
        *,
        side: str,
        state_machine: AdaptiveGraspStateMachine,
        grasp_start_alpha: float,
        hold_start_alpha: float,
        release_start_alpha: float,
        reset_below_alpha: float,
        finish_when_all_groups_stopped: bool,
        raise_on_object_missing: bool,
    ) -> None:
        self.side = side
        self.state_machine = state_machine
        self.grasp_start_alpha = grasp_start_alpha
        self.hold_start_alpha = hold_start_alpha
        self.release_start_alpha = release_start_alpha
        self.reset_below_alpha = reset_below_alpha
        self.finish_when_all_groups_stopped = finish_when_all_groups_stopped
        self.raise_on_object_missing = raise_on_object_missing

    @classmethod
    def from_config(cls, side: str, cfg: DictConfig) -> "_AdaptiveHandSideController":
        trajectory = load_anchor_trajectory(Path(str(cfg.anchor_dir_by_side[side])), side)
        planner = AdaptiveGraspPlanner(
            trajectory=trajectory,
            group_indices=group_indices_from_config(cfg.stop_groups),
            feedback=AdaptiveGraspFeedbackConfig(
                enabled=bool(cfg.feedback.enabled),
                torque_delta_threshold_by_group=_thresholds_for_side(
                    cfg.feedback.torque_delta_threshold_by_group,
                    cfg.feedback.torque_delta_threshold_by_side,
                    side,
                ),
                position_lag_threshold_rad_by_group=cfg.feedback.position_lag_threshold_rad_by_group,
                min_alpha_for_stop=float(cfg.feedback.min_alpha_for_stop),
            ),
        )
        state_machine = AdaptiveGraspStateMachine(
            planner=planner,
            hold=AdaptiveGraspHoldConfig(
                enabled=bool(cfg.hold.enabled),
                min_torque_delta_ratio=float(cfg.hold.min_torque_delta_ratio),
                max_torque_delta_ratio=float(cfg.hold.max_torque_delta_ratio),
                missing_torque_delta_ratio=float(cfg.hold.missing_torque_delta_ratio),
                close_step_rad=float(cfg.hold.close_step_rad),
                open_step_rad=float(cfg.hold.open_step_rad),
                max_extra_close_rad=float(cfg.hold.max_extra_close_rad),
                missing_target_margin_rad=float(cfg.hold.missing_target_margin_rad),
                missing_consecutive_frames=int(cfg.hold.missing_consecutive_frames),
                log_interval_s=float(cfg.hold.log_interval_s),
            ),
        )
        return cls(
            side=side,
            state_machine=state_machine,
            grasp_start_alpha=float(cfg.alpha.grasp_start),
            hold_start_alpha=float(cfg.alpha.hold_start),
            release_start_alpha=float(cfg.alpha.release_start),
            reset_below_alpha=float(cfg.alpha.reset_below),
            finish_when_all_groups_stopped=bool(cfg.alpha.finish_when_all_groups_stopped),
            raise_on_object_missing=bool(cfg.raise_on_object_missing),
        )

    def mark_released(self) -> None:
        self.state_machine.mark_released()

    def command(self, alpha: float, state: SharpaHandState) -> _AdaptiveHandCommand:
        alpha = float(alpha)
        current_state = self.state_machine.state
        if alpha <= self.release_start_alpha:
            q_rad = self._release_command(alpha)
            if alpha <= self.reset_below_alpha:
                self.state_machine.mark_released()
            return self._command(q_rad, alpha, current_state, "release")
        if current_state in {AdaptiveGraspState.RELEASED, AdaptiveGraspState.RELEASING, AdaptiveGraspState.OBJECT_MISSING}:
            if alpha < self.grasp_start_alpha:
                return self._command(self.state_machine.trajectory.interpolate(alpha), alpha, current_state, "pre_grasp")
            self.state_machine.reset_for_grasp(state)
            current_state = self.state_machine.state
        if current_state == AdaptiveGraspState.GRASPING:
            grasp_command = self.state_machine.grasp_command(alpha, state)
            if self._should_finish_grasp(alpha):
                finished_state = self.state_machine.finish_grasp()
                if finished_state == AdaptiveGraspState.OBJECT_MISSING and self.raise_on_object_missing:
                    raise RuntimeError(f"adaptive_hand_action object missing side={self.side}")
            return self._command(
                grasp_command.q_rad,
                alpha,
                current_state,
                "grasp",
                new_stops=tuple(stop.group for stop in grasp_command.new_stops),
                stopped_groups=self.state_machine.stopped_groups,
            )
        if current_state == AdaptiveGraspState.HOLDING:
            hold_command = self.state_machine.hold_command(state)
            if hold_command.object_missing and self.raise_on_object_missing:
                raise RuntimeError(f"adaptive_hand_action object missing side={self.side}")
            return self._command(
                hold_command.q_rad,
                alpha,
                current_state,
                "hold",
                hold_actions={group.group: group.action for group in hold_command.groups},
                object_missing=hold_command.object_missing,
                stopped_groups=self.state_machine.stopped_groups,
            )
        raise RuntimeError(f"unsupported adaptive_hand_action state side={self.side}: {current_state}")

    def _should_finish_grasp(self, alpha: float) -> bool:
        if alpha >= self.hold_start_alpha:
            return True
        return self.finish_when_all_groups_stopped and self.state_machine.all_groups_stopped

    def _release_command(self, alpha: float) -> tuple[float, ...]:
        if self.state_machine.state not in {AdaptiveGraspState.RELEASED, AdaptiveGraspState.RELEASING}:
            self.state_machine.start_release()
        return self.state_machine.trajectory.interpolate(alpha)

    def _command(
        self,
        q_rad: tuple[float, ...],
        alpha: float,
        previous_state: AdaptiveGraspState,
        phase: str,
        **extra_summary: object,
    ) -> _AdaptiveHandCommand:
        summary = {
            "side": self.side,
            "alpha": float(alpha),
            "phase": phase,
            "previous_state": str(previous_state),
            "state": str(self.state_machine.state),
            **extra_summary,
        }
        return _AdaptiveHandCommand(q_rad=tuple(float(value) for value in q_rad), summary=summary)


def _thresholds_for_side(base_cfg, by_side_cfg, side: str) -> dict[str, float]:
    output = {str(group): float(value) for group, value in _items(base_cfg)}
    if by_side_cfg is not None and side in by_side_cfg:
        output.update({str(group): float(value) for group, value in _items(by_side_cfg[side])})
    return output


def _items(mapping: Mapping) -> tuple[tuple[object, object], ...]:
    return tuple(mapping.items())


def _sharpa_state(state: RobotState) -> SharpaHandState:
    if state.sharpa is None:
        raise RuntimeError(f"adaptive_hand_action requires SharpaHandState with torque feedback for side={state.side}")
    return state.sharpa
