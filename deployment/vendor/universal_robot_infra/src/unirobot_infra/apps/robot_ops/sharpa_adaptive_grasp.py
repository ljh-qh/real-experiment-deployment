from pathlib import Path
import time

import hydra
from omegaconf import DictConfig
from rich.console import Console

from unirobot_infra.grasp.adaptive import (
    AdaptiveGraspFeedbackConfig,
    AdaptiveGraspPlanner,
    group_indices_from_config,
    load_anchor_trajectory,
)
from unirobot_infra.robots.end_effector.sharpa import SharpaHand
from unirobot_infra.utils.logging import get_logger, setup_logging
from unirobot_infra.utils.sdk_output_router import start_sdk_output_router_from_config
from unirobot_infra.utils.sdk_paths import sharpa_manus_sdk_paths


CONFIG_PATH = str(Path(__file__).resolve().parents[4] / "configs")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="apps/robot_ops/sharpa_adaptive_grasp/default")
def main(cfg: DictConfig) -> None:
    console = Console(stderr=True)
    setup_logging(cfg.logging.level, console=console)
    logger = get_logger(__name__)
    with start_sdk_output_router_from_config(cfg.app.sdk_output) as sdk_output_router:
        if sdk_output_router is not None:
            logger.info("SDK 输出重定向目录=%s", sdk_output_router.run_dir)
        run_sharpa_adaptive_grasp(cfg, logger)


def run_sharpa_adaptive_grasp(cfg: DictConfig, logger) -> None:
    side = str(cfg.adaptive_grasp.side)
    action = str(cfg.adaptive_grasp.action)
    _validate_side(cfg, side)
    _validate_grasp_runtime_config(cfg)
    trajectory = load_anchor_trajectory(Path(str(cfg.adaptive_grasp.anchor_dir)), side)
    planner = _build_planner(cfg, trajectory) if action == "grasp" else None
    logger.info(
        "Sharpa hand action 启动 side=%s action=%s anchors=%s duration_s=%.3f rate_hz=%.3f feedback=%s dry_run=%s",
        side,
        action,
        trajectory.sample_count,
        float(cfg.adaptive_grasp.duration_s),
        float(cfg.adaptive_grasp.rate_hz),
        "on" if action == "grasp" and bool(cfg.adaptive_grasp.feedback.enabled) else "off",
        bool(cfg.adaptive_grasp.dry_run),
    )
    if bool(cfg.adaptive_grasp.dry_run):
        if action == "open":
            _run_dry_open(cfg, trajectory, logger)
        else:
            _run_dry_adaptive_grasp(cfg, planner, logger)
        return
    hand = _build_hand(cfg, side)
    try:
        hand.connect()
        if action == "open":
            _run_hand_open(cfg, hand, trajectory, logger)
        else:
            _run_hand_adaptive_grasp(cfg, hand, planner, logger)
    finally:
        hand.close()


def _build_planner(cfg: DictConfig, trajectory) -> AdaptiveGraspPlanner:
    return AdaptiveGraspPlanner(
        trajectory=trajectory,
        group_indices=group_indices_from_config(cfg.adaptive_grasp.stop_groups),
        feedback=AdaptiveGraspFeedbackConfig(
            enabled=bool(cfg.adaptive_grasp.feedback.enabled),
            torque_delta_threshold_by_group=_float_by_group(
                cfg.adaptive_grasp.feedback.torque_delta_threshold_by_group
            ),
            position_lag_threshold_rad_by_group=_float_by_group(
                cfg.adaptive_grasp.feedback.position_lag_threshold_rad_by_group
            ),
            min_alpha_for_stop=float(cfg.adaptive_grasp.feedback.min_alpha_for_stop),
        ),
    )


def _run_hand_adaptive_grasp(cfg: DictConfig, hand: SharpaHand, planner: AdaptiveGraspPlanner, logger) -> None:
    settle_q = planner.trajectory.interpolate(0.0)
    hand.send_joint_position_rad(settle_q)
    time.sleep(float(cfg.adaptive_grasp.initial_settle_s))
    planner.reset(hand.read_state())
    period_s = 1.0 / float(cfg.adaptive_grasp.rate_hz)
    duration_s = float(cfg.adaptive_grasp.duration_s)
    start_s = time.monotonic()
    command = None
    while True:
        loop_start = time.monotonic()
        elapsed_s = loop_start - start_s
        alpha = min(1.0, elapsed_s / duration_s)
        state = hand.read_state()
        command = planner.command(alpha, state)
        hand.send_joint_position_rad(command.q_rad)
        _log_new_stops(command.new_stops, logger)
        if alpha >= 1.0 or planner.all_groups_stopped:
            break
        _sleep_remaining(loop_start, period_s)
    logger.info(
        "Sharpa adaptive grasp 完成 alpha=%.3f stopped=%s max|q|=%.3f",
        0.0 if command is None else command.alpha,
        ",".join(planner.stopped_groups) if planner.stopped_groups else "none",
        0.0 if command is None else max(abs(value) for value in command.q_rad),
    )
    _log_feedback_stats(planner, logger)


def _run_hand_open(cfg: DictConfig, hand: SharpaHand, trajectory, logger) -> None:
    target_q = trajectory.interpolate(0.0)
    start_q = _checked_state_q(hand.read_state().joint_pos, trajectory.dof)
    period_s = 1.0 / float(cfg.adaptive_grasp.rate_hz)
    duration_s = float(cfg.adaptive_grasp.duration_s)
    start_s = time.monotonic()
    command_q = start_q
    while True:
        loop_start = time.monotonic()
        elapsed_s = loop_start - start_s
        alpha = min(1.0, elapsed_s / duration_s)
        command_q = _interpolate_q(start_q, target_q, alpha)
        hand.send_joint_position_rad(command_q)
        if alpha >= 1.0:
            break
        _sleep_remaining(loop_start, period_s)
    logger.info("Sharpa open 完成 alpha=1.000 max|q|=%.3f", max(abs(value) for value in command_q))


def _run_dry_adaptive_grasp(cfg: DictConfig, planner: AdaptiveGraspPlanner, logger) -> None:
    steps = int(cfg.adaptive_grasp.dry_run_steps)
    if steps <= 0:
        raise ValueError("adaptive_grasp.dry_run_steps must be positive")
    for step in range(steps + 1):
        alpha = step / steps
        q_rad = planner.trajectory.interpolate(alpha)
        logger.info("dry_run alpha=%.3f max|q|=%.3f", alpha, max(abs(value) for value in q_rad))


def _run_dry_open(cfg: DictConfig, trajectory, logger) -> None:
    q_rad = trajectory.interpolate(0.0)
    logger.info("dry_run open max|q|=%.3f", max(abs(value) for value in q_rad))


def _build_hand(cfg: DictConfig, side: str) -> SharpaHand:
    ee_cfg = cfg.robot.hardware.end_effector
    sdk_paths = sharpa_manus_sdk_paths(Path(str(ee_cfg.sdk.vendor_root)))
    return SharpaHand(ee_cfg, side=side, sdk_paths=sdk_paths)


def _validate_side(cfg: DictConfig, side: str) -> None:
    if side not in cfg.robot.sides:
        raise ValueError(f"adaptive_grasp.side={side} not found in robot.sides")
    if "end_effector" not in cfg.robot.sides[side].parts:
        raise ValueError(f"adaptive_grasp.side={side} must have end_effector part")


def _validate_grasp_runtime_config(cfg: DictConfig) -> None:
    if str(cfg.adaptive_grasp.action) not in {"open", "grasp"}:
        raise ValueError("adaptive_grasp.action must be open or grasp")
    if float(cfg.adaptive_grasp.rate_hz) <= 0.0:
        raise ValueError("adaptive_grasp.rate_hz must be positive")
    if float(cfg.adaptive_grasp.duration_s) <= 0.0:
        raise ValueError("adaptive_grasp.duration_s must be positive")
    if float(cfg.adaptive_grasp.initial_settle_s) < 0.0:
        raise ValueError("adaptive_grasp.initial_settle_s must not be negative")
    for group in cfg.adaptive_grasp.stop_groups:
        if float(cfg.adaptive_grasp.feedback.torque_delta_threshold_by_group[group]) < 0.0:
            raise ValueError(f"adaptive_grasp.feedback.torque_delta_threshold_by_group.{group} must not be negative")
        if float(cfg.adaptive_grasp.feedback.position_lag_threshold_rad_by_group[group]) < 0.0:
            raise ValueError(
                f"adaptive_grasp.feedback.position_lag_threshold_rad_by_group.{group} must not be negative"
            )
    if not 0.0 <= float(cfg.adaptive_grasp.feedback.min_alpha_for_stop) <= 1.0:
        raise ValueError("adaptive_grasp.feedback.min_alpha_for_stop must be in [0, 1]")


def _float_by_group(cfg_value) -> dict[str, float]:
    return {str(group): float(cfg_value[group]) for group in cfg_value}


def _checked_state_q(q_rad, dof: int) -> tuple[float, ...]:
    q = tuple(float(value) for value in q_rad)
    if len(q) != dof:
        raise RuntimeError(f"Sharpa hand joint_pos length must be {dof}, got {len(q)}")
    return q


def _interpolate_q(start_q: tuple[float, ...], target_q: tuple[float, ...], alpha: float) -> tuple[float, ...]:
    return tuple(float(start + (target - start) * alpha) for start, target in zip(start_q, target_q, strict=True))


def _log_new_stops(stops, logger) -> None:
    for stop in stops:
        logger.info(
            "冻结手指 group=%s signal=%s value=%.6f threshold=%.6f alpha=%.3f",
            stop.group,
            stop.signal,
            stop.value,
            stop.threshold,
            stop.alpha,
        )


def _log_feedback_stats(planner: AdaptiveGraspPlanner, logger) -> None:
    for stats in planner.feedback_stats():
        logger.info(
            "空抓反馈统计 group=%s max_torque_delta=%.6f max_position_lag_rad=%.6f",
            stats.group,
            stats.max_torque_delta,
            stats.max_position_lag_rad,
        )


def _sleep_remaining(loop_start: float, period_s: float) -> None:
    remaining_s = period_s - (time.monotonic() - loop_start)
    if remaining_s > 0.0:
        time.sleep(remaining_s)


if __name__ == "__main__":
    main()
