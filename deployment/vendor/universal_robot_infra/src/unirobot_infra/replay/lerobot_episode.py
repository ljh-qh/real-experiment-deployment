from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from omegaconf import DictConfig


@dataclass(frozen=True)
class LeRobotEpisodeData:
    actions: np.ndarray
    observations: np.ndarray
    frame_indices: np.ndarray
    timestamps: np.ndarray
    dataset_fps: float


def load_lerobot_episode_actions(dataset_cfg: DictConfig, logger) -> tuple[np.ndarray, float]:
    episode_data = load_lerobot_episode_data(dataset_cfg, logger)
    return episode_data.actions, episode_data.dataset_fps


def load_lerobot_episode_data(dataset_cfg: DictConfig, logger) -> LeRobotEpisodeData:
    validate_dataset_config(dataset_cfg)
    logger.info("Importing LeRobot dataset dependencies")
    import_started = time.monotonic()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    logger.info("LeRobot dataset dependencies imported in %.2fs", time.monotonic() - import_started)

    root = None if dataset_cfg.root is None else Path(str(dataset_cfg.root))
    logger.info(
        "Creating LeRobotDataset repo_id=%s root=%s episode=%s tolerance_s=%s",
        dataset_cfg.repo_id,
        root,
        int(dataset_cfg.episode_index),
        float(dataset_cfg.tolerance_s),
    )
    dataset_started = time.monotonic()
    dataset = LeRobotDataset(
        repo_id=str(dataset_cfg.repo_id),
        root=root,
        episodes=[int(dataset_cfg.episode_index)],
        tolerance_s=float(dataset_cfg.tolerance_s),
        download_videos=False,
    )
    logger.info(
        "LeRobotDataset created in %.2fs rows=%s fps=%s",
        time.monotonic() - dataset_started,
        len(dataset.hf_dataset),
        float(dataset.fps),
    )
    logger.info("Selecting LeRobot episode_index=%s from episode columns", int(dataset_cfg.episode_index))
    select_started = time.monotonic()
    episode_dataset = dataset.select_columns(
        ["episode_index", "frame_index", "timestamp", "observation.state", "action"]
    )
    episode_indices = np.flatnonzero(
        np.asarray(episode_dataset["episode_index"], dtype=np.int64) == int(dataset_cfg.episode_index)
    )
    logger.info(
        "Selected %s action rows in %.2fs",
        len(episode_indices),
        time.monotonic() - select_started,
    )
    if len(episode_indices) == 0:
        raise ValueError(f"episode_index={int(dataset_cfg.episode_index)} not found in dataset")
    logger.info("Extracting action values for %s episode frames", len(episode_indices))
    action_started = time.monotonic()
    actions = _column_to_array(episode_dataset["action"], episode_indices)
    observations = _column_to_array(episode_dataset["observation.state"], episode_indices)
    frame_indices = _scalar_column_to_array(episode_dataset["frame_index"], episode_indices, np.int64)
    timestamps = _scalar_column_to_array(episode_dataset["timestamp"], episode_indices, np.float64)
    logger.info(
        "Extracted actions shape=%s observations shape=%s in %.2fs",
        actions.shape,
        observations.shape,
        time.monotonic() - action_started,
    )
    return LeRobotEpisodeData(
        actions=actions,
        observations=observations,
        frame_indices=frame_indices,
        timestamps=timestamps,
        dataset_fps=float(dataset.fps),
    )


def validate_dataset_config(dataset_cfg: DictConfig) -> None:
    repo_id = str(dataset_cfg.repo_id)
    if Path(repo_id).exists():
        raise ValueError(
            f"app.dataset.repo_id looks like a local path: {repo_id}. "
            "Use app.dataset.root for local dataset paths and keep app.dataset.repo_id as an id such as local/my_dataset."
        )


def validate_action_dim(actions: np.ndarray, expected_dim: int) -> None:
    if actions.ndim != 2:
        raise ValueError("LeRobot action must be a 2D array")
    if actions.shape[1] != expected_dim:
        raise ValueError(f"LeRobot action dim {actions.shape[1]} does not match robot action layout {expected_dim}")


def _column_to_array(column, episode_indices: np.ndarray) -> np.ndarray:
    values = np.asarray([column[int(index)] for index in episode_indices], dtype=np.float32)
    if values.ndim == 1:
        return values.reshape(len(episode_indices), 1)
    return values.reshape(values.shape[0], -1)


def _scalar_column_to_array(column, episode_indices: np.ndarray, dtype) -> np.ndarray:
    return np.asarray([column[int(index)] for index in episode_indices], dtype=dtype).reshape(-1)

