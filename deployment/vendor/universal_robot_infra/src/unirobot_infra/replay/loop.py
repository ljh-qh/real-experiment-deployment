from __future__ import annotations

import time
from typing import Protocol

import numpy as np
from omegaconf import DictConfig
from tqdm.auto import tqdm

from unirobot_infra.controllers.delta_ee import DeltaEeCommandController
from unirobot_infra.core.types import RobotState
from unirobot_infra.replay.lerobot_episode import LeRobotEpisodeData
from unirobot_infra.wrappers.base import EnvWrapper
from unirobot_infra.wrappers.tianji_delta_ee_action import TianjiDeltaEeActionWrapper


class ReplayTrajectoryLogger(Protocol):
    def record(self, action_index: int, info: dict) -> None: ...

    def record_chunk_end(
        self,
        chunk_start_action_index: int,
        chunk_end_action_index: int,
        states: tuple[RobotState, ...],
        info: dict,
    ) -> None: ...


def replay_actions(
    cfg: DictConfig,
    logger,
    env,
    episode_data: LeRobotEpisodeData,
    rate_hz: float,
    trajectory_logger: ReplayTrajectoryLogger | None,
    initial_info: dict | None = None,
) -> None:
    action_space = str(cfg.app.replay.action_space)
    if action_space == "joint":
        trajectory = episode_data.observations
    elif action_space == "delta_ee":
        trajectory = episode_data.actions
    else:
        raise ValueError(f"app.replay.action_space must be delta_ee or joint, got {action_space!r}")
    start_frame = int(cfg.app.replay.start_frame)
    end_frame_value = cfg.app.replay.end_frame
    max_frames = int(cfg.app.replay.max_frames)
    if start_frame < 0:
        raise ValueError("app.replay.start_frame must be non-negative")
    if max_frames < 0:
        raise ValueError("app.replay.max_frames must be non-negative")
    if start_frame >= len(trajectory):
        raise ValueError(f"app.replay.start_frame={start_frame} is outside episode length {len(trajectory)}")
    if end_frame_value is not None and max_frames != 0:
        raise ValueError("app.replay.end_frame and app.replay.max_frames cannot both be set")
    if end_frame_value is not None:
        end_frame = int(end_frame_value)
        if end_frame < start_frame:
            raise ValueError("app.replay.end_frame must be greater than or equal to start_frame")
        if end_frame >= len(trajectory):
            raise ValueError(f"app.replay.end_frame={end_frame} is outside episode length {len(trajectory)}")
        stop_frame = end_frame + 1
    else:
        stop_frame = len(trajectory) if max_frames == 0 else min(len(trajectory), start_frame + max_frames)

    period_s = 1.0 / rate_hz
    if action_space == "joint":
        _replay_joint_actions(
            cfg=cfg,
            logger=logger,
            env=env,
            trajectory=trajectory,
            start_frame=start_frame,
            stop_frame=stop_frame,
            period_s=period_s,
            trajectory_logger=trajectory_logger,
            initial_info=initial_info,
        )
    else:
        delta_ee_controller = _delta_ee_controller_for_replay(cfg, env, initial_info, trajectory_logger)
        if bool(cfg.app.replay.chunk.enabled):
            _replay_actions_in_command_target_chunks(
                cfg=cfg,
                logger=logger,
                controller=delta_ee_controller,
                actions=trajectory,
                start_frame=start_frame,
                stop_frame=stop_frame,
                period_s=period_s,
                trajectory_logger=trajectory_logger,
            )
        else:
            logger.info(
                "Replaying frames %s..%s with robot commands blocking=%s",
                start_frame,
                stop_frame - 1,
                bool(cfg.env.runtime.action_blocking),
            )
            for frame_index in tqdm(range(start_frame, stop_frame), desc="Replay LeRobot", unit="frame"):
                loop_start = time.monotonic()
                action = tuple(float(value) for value in trajectory[frame_index])
                if delta_ee_controller is None:
                    _, _, _, _, info = env.step(action)
                else:
                    _, _, _, _, info = delta_ee_controller.send_action(action)
                if trajectory_logger is not None:
                    trajectory_logger.record(frame_index, info)
                _sleep_remaining(loop_start, period_s)
    logger.success("Replay finished: %s frames", stop_frame - start_frame)


def _replay_actions_in_command_target_chunks(
    cfg: DictConfig,
    logger,
    controller: DeltaEeCommandController | None,
    actions: np.ndarray,
    start_frame: int,
    stop_frame: int,
    period_s: float,
    trajectory_logger: ReplayTrajectoryLogger | None,
) -> None:
    if controller is None:
        raise ValueError("app.replay.chunk.enabled requires wrappers.tianji_delta_ee_action.enabled=true")
    if bool(cfg.env.runtime.action_blocking):
        raise ValueError("app.replay.chunk.enabled requires env.runtime.action_blocking=false")
    chunk_size = int(cfg.app.replay.chunk.size)
    if chunk_size <= 0:
        raise ValueError("app.replay.chunk.size must be positive")
    logger.info(
        "Replaying frames %s..%s with command-target chunks size=%s chunk_end_blocking=True",
        start_frame,
        stop_frame - 1,
        chunk_size,
    )
    progress = tqdm(total=stop_frame - start_frame, desc="Replay LeRobot", unit="frame")
    try:
        chunk_start = start_frame
        while chunk_start < stop_frame:
            chunk_stop = min(stop_frame, chunk_start + chunk_size)
            controller.begin_chunk()
            try:
                last_info = None
                for frame_index in range(chunk_start, chunk_stop):
                    loop_start = time.monotonic()
                    action = tuple(float(value) for value in actions[frame_index])
                    _, _, _, _, info = controller.send_chunk_action(action)
                    last_info = info
                    if trajectory_logger is not None:
                        trajectory_logger.record(frame_index, info)
                    progress.update(1)
                    _sleep_remaining(loop_start, period_s)
                if last_info is None:
                    raise RuntimeError("chunk replay produced no frame info")
                chunk_end_states = controller.wait_until_current_target_reached(
                    f"replay chunk {chunk_start}..{chunk_stop - 1} arm target"
                )
                if trajectory_logger is not None:
                    trajectory_logger.record_chunk_end(
                        chunk_start_action_index=chunk_start,
                        chunk_end_action_index=chunk_stop - 1,
                        states=tuple(chunk_end_states),
                        info=last_info,
                    )
            finally:
                controller.end_chunk()
            chunk_start = chunk_stop
    finally:
        progress.close()


def _replay_joint_actions(
    cfg: DictConfig,
    logger,
    env,
    trajectory: np.ndarray,
    start_frame: int,
    stop_frame: int,
    period_s: float,
    trajectory_logger: ReplayTrajectoryLogger | None,
    initial_info: dict | None,
) -> None:
    max_step_rad_value = cfg.app.replay.joint.max_step_rad
    max_step_rad = None if max_step_rad_value is None else float(max_step_rad_value)
    previous_command: np.ndarray | None = None
    if max_step_rad is not None:
        previous_command = _joint_anchor_from_initial_info(env, initial_info)
    clamped_frame_count = 0
    logger.info(
        "Replaying joint frames %s..%s from observation.state with robot commands blocking=%s max_step_rad=%s",
        start_frame,
        stop_frame - 1,
        bool(cfg.env.runtime.action_blocking),
        max_step_rad,
    )
    for frame_index in tqdm(range(start_frame, stop_frame), desc="Replay LeRobot", unit="frame"):
        loop_start = time.monotonic()
        frame = np.asarray(trajectory[frame_index], dtype=np.float64)
        command = frame
        if max_step_rad is not None:
            if previous_command is None:
                logger.info(
                    "Joint replay max_step_rad=%.4f is set but reset joint anchor is unavailable; frame %s sent unclamped",
                    max_step_rad,
                    frame_index,
                )
            else:
                # 截断基准是上一帧已发命令，保证累计步长不超过限制
                step = frame - previous_command
                clipped_step = np.clip(step, -max_step_rad, max_step_rad)
                command = previous_command + clipped_step
                if np.any(clipped_step != step):
                    clamped_frame_count += 1
            previous_command = command
        action = tuple(float(value) for value in command)
        _, _, _, _, info = env.step(action)
        if trajectory_logger is not None:
            trajectory_logger.record(frame_index, info)
        _sleep_remaining(loop_start, period_s)
    if max_step_rad is not None:
        logger.info(
            "Joint replay clamped %s of %s frames to max_step_rad=%.4f rad",
            clamped_frame_count,
            stop_frame - start_frame,
            max_step_rad,
        )


def _joint_anchor_from_initial_info(env, initial_info: dict | None) -> np.ndarray | None:
    if initial_info is None:
        return None
    states = tuple(initial_info["robot_states"])
    return np.asarray(env.action_layout.observation_from_states(states, ("joint_pos",)), dtype=np.float64)


def _sleep_remaining(loop_start: float, period_s: float) -> None:
    remaining_s = period_s - (time.monotonic() - loop_start)
    if remaining_s > 0.0:
        time.sleep(remaining_s)


def _delta_ee_controller_for_replay(
    cfg: DictConfig,
    env,
    initial_info: dict | None,
    trajectory_logger: ReplayTrajectoryLogger | None,
) -> DeltaEeCommandController | None:
    if str(cfg.app.replay.action_space) == "joint":
        return None
    needs_controller = bool(cfg.app.replay.chunk.enabled) or trajectory_logger is not None
    if not needs_controller:
        return None
    if not bool(cfg.wrappers.tianji_delta_ee_action.enabled):
        return None
    if initial_info is None:
        raise RuntimeError("delta ee replay controller requires reset info")
    wrapper = _find_delta_ee_action_wrapper(env)
    return DeltaEeCommandController(
        env=wrapper.env,
        expander=wrapper.action_expander,
        initial_states=tuple(initial_info["robot_states"]),
    )


def _find_delta_ee_action_wrapper(env) -> TianjiDeltaEeActionWrapper:
    current = env
    while isinstance(current, EnvWrapper):
        if isinstance(current, TianjiDeltaEeActionWrapper):
            return current
        current = current.env
    raise RuntimeError("wrappers.tianji_delta_ee_action.enabled=true but wrapper is not installed")
