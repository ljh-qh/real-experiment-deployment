"""Replay a LeRobot dataset episode on the robot."""

from __future__ import annotations

from pathlib import Path
import time

import hydra
import numpy as np
from omegaconf import DictConfig
from rich.console import Console

from unirobot_infra.replay.lerobot_episode import (
    LeRobotEpisodeData,
    load_lerobot_episode_actions,
    load_lerobot_episode_data,
    validate_action_dim as _validate_action_dim,
)
from unirobot_infra.replay.loop import replay_actions
from unirobot_infra.replay.tcp_trajectory_log import build_tcp_trajectory_logger
from unirobot_infra.utils.logging import get_logger, setup_logging
from unirobot_infra.utils.sdk_output_router import start_sdk_output_router_from_config
from unirobot_infra.wrappers.builder import build_env


CONFIG_PATH = str(Path(__file__).resolve().parents[3] / "configs")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="apps/replay/default")
def main(cfg: DictConfig) -> None:
    console = Console(stderr=True)
    setup_logging(cfg.logging.level, console=console)
    logger = get_logger(__name__)
    with start_sdk_output_router_from_config(cfg.app.sdk_output) as sdk_output_router:
        if sdk_output_router is not None:
            logger.info("SDK 输出重定向目录=%s", sdk_output_router.run_dir)
        replay_lerobot(cfg, logger)


def replay_lerobot(cfg: DictConfig, logger) -> None:
    validate_replay_config(cfg)
    action_space = str(cfg.app.replay.action_space)
    episode_data = load_lerobot_episode_data(cfg.app.dataset, logger)
    rate_hz = episode_data.dataset_fps if cfg.app.replay.rate_hz is None else float(cfg.app.replay.rate_hz)
    if rate_hz <= 0.0:
        raise ValueError("app.replay.rate_hz must be positive")
    logger.info(
        "Loaded LeRobot episode repo=%s root=%s episode=%s frames=%s dataset_fps=%s replay_hz=%.2f",
        cfg.app.dataset.repo_id,
        cfg.app.dataset.root,
        int(cfg.app.dataset.episode_index),
        len(episode_data.actions),
        episode_data.dataset_fps,
        rate_hz,
    )
    logger.info(
        "Replay action_space=%s trajectory_source=%s start_frame=%s end_frame=%s max_frames=%s",
        action_space,
        "observation.state" if action_space == "joint" else "action",
        int(cfg.app.replay.start_frame),
        cfg.app.replay.end_frame,
        int(cfg.app.replay.max_frames),
    )

    logger.info("Building robot environment")
    env = build_env(cfg)
    logger.info("Robot environment built with action_dim=%s", env.action_layout.size)
    trajectory_logger = None
    try:
        if action_space == "joint":
            logger.info(
                "Validating joint trajectory dim=%s against robot action_dim=%s",
                episode_data.observations.shape[1],
                env.action_layout.size,
            )
            validate_joint_trajectory_dim(episode_data.observations, env.action_layout.size)
        else:
            logger.info(
                "Validating LeRobot action dim=%s against robot action_dim=%s",
                episode_data.actions.shape[1],
                env.action_layout.size,
            )
            _validate_action_dim(episode_data.actions, env.action_layout.size)
        trajectory_logger = build_tcp_trajectory_logger(cfg, episode_data, logger)
        logger.info(
            "Resetting robot before replay move_to_initial_state=%s settle_steps=%s wait_timeout_s=%s",
            bool(cfg.env.reset.move_to_initial_state),
            int(cfg.env.reset.settle_steps),
            float(cfg.env.reset.wait_timeout_s),
        )
        reset_started = time.monotonic()
        _, reset_info = env.reset()
        logger.info("Robot reset finished in %.2fs", time.monotonic() - reset_started)
        if trajectory_logger is not None:
            trajectory_logger.record_reset(reset_info)
        replay_actions(
            cfg=cfg,
            logger=logger,
            env=env,
            episode_data=episode_data,
            rate_hz=rate_hz,
            trajectory_logger=trajectory_logger,
            initial_info=reset_info,
        )
    finally:
        if trajectory_logger is not None:
            trajectory_logger.close()
        logger.info("Closing robot environment")
        env.close()
        logger.info("Robot environment closed")


def validate_replay_config(cfg: DictConfig) -> None:
    action_space = str(cfg.app.replay.action_space)
    if action_space == "delta_ee":
        return
    if action_space != "joint":
        raise ValueError(f"app.replay.action_space must be delta_ee or joint, got {action_space!r}")
    if "tianji_delta_ee_action" in cfg.wrappers and bool(cfg.wrappers.tianji_delta_ee_action.enabled):
        raise ValueError(
            "app.replay.action_space=joint requires wrappers.tianji_delta_ee_action.enabled=false "
            "because the delta EE wrapper rewrites the base joint action layout"
        )
    if bool(cfg.app.replay.chunk.enabled):
        raise ValueError(
            "app.replay.action_space=joint conflicts with app.replay.chunk.enabled=true "
            "because chunk replay drives the delta EE command controller"
        )
    if bool(cfg.app.replay.trajectory_log.enabled):
        raise ValueError(
            "app.replay.action_space=joint requires app.replay.trajectory_log.enabled=false "
            "because trajectory_log.type=delta_ee_tcp depends on the delta EE wrapper; "
            "disable trajectory_log or switch to action_space=delta_ee"
        )


def validate_joint_trajectory_dim(observations: np.ndarray, expected_dim: int) -> None:
    if observations.ndim != 2:
        raise ValueError("joint replay requires observation.state to be a 2D array")
    if observations.shape[1] != expected_dim:
        raise ValueError(
            f"joint replay requires observation.state dim {observations.shape[1]} "
            f"to match robot action layout {expected_dim}; convert the dataset with the same "
            "robot config and keep observation.state joint order aligned with the base action layout"
        )


def _load_episode_actions(cfg: DictConfig, logger):
    return load_lerobot_episode_actions(cfg.app.dataset, logger)


def _load_episode_data(cfg: DictConfig, logger) -> LeRobotEpisodeData:
    return load_lerobot_episode_data(cfg.app.dataset, logger)


if __name__ == "__main__":
    main()
