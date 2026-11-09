from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import cv2
import numpy as np
from omegaconf import DictConfig
from tqdm.auto import tqdm

from unirobot_infra.dataset.convert_config import (
    ActionConfig,
    ConvertCameraConfig,
    ConvertConfig,
    ConvertTactileImageConfig,
    FieldSpec,
    FrameFilterConfig,
    StateConfig,
)
from unirobot_infra.dataset.dataset_card import build_lerobot_dataset_card_kwargs, write_lerobot_dataset_card
from unirobot_infra.dataset.episode_reader import EpisodeReader
from unirobot_infra.dataset.resampler import ResampledEpisode, resample_episode
from unirobot_infra.dataset.resampling_timeline import write_resampling_timeline_html
from unirobot_infra.robots.arm.tianji_tcp_pose import TianjiTcpPoseMapper, rotation_6d_from_matrix
from unirobot_infra.utils.logging import get_logger
from unirobot_infra.wrappers.hand_alpha_action import hand_state_alpha


logger = get_logger(__name__)
DELTA_EE_WORLD_WITH_HAND_ALPHA = "delta_ee_world_with_hand_alpha"
DELTA_EE_WORLD_WITH_HAND_JOINT = "delta_ee_world_with_hand_joint"
TCP_POSE_WORLD_WITH_HAND_ALPHA_STATE = "tcp_pose_world_with_hand_alpha"
TCP_POSE_WORLD_WITH_HAND_JOINT_STATE = "tcp_pose_world_with_hand_joint"


@dataclass(frozen=True)
class _DeltaEeWorldHandActionArrays:
    arm_q_by_side: dict[str, np.ndarray]
    hand_q_by_side: dict[str, np.ndarray]
    mappers_by_side: dict[str, TianjiTcpPoseMapper]


@dataclass(frozen=True)
class _TcpPoseWorldHandStateArrays:
    arm_q_by_side: dict[str, np.ndarray]
    hand_q_by_side: dict[str, np.ndarray]
    mappers_by_side: dict[str, TianjiTcpPoseMapper]


@dataclass(frozen=True)
class _OutputFrameSelection:
    source_indices: np.ndarray
    target_indices: np.ndarray | None
    dropped_frames: int


def convert_to_lerobot(
    data_dir: str | Path,
    output_dir: str | Path,
    config: ConvertConfig,
    task_name: str,
    language_instruction: str,
    *,
    push_to_hub: bool = False,
    inspect_timestamps: bool = False,
    config_path: str | None = None,
    overwrite: bool = False,
    robot_cfg: DictConfig | None = None,
) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.io_utils import write_info

    if config.action.type not in {"joint_position", DELTA_EE_WORLD_WITH_HAND_ALPHA, DELTA_EE_WORLD_WITH_HAND_JOINT}:
        raise NotImplementedError(f"Action type {config.action.type!r} is not supported")
    if config.frame_filter.enabled and config.action.source != "next_state":
        raise ValueError("frame_filter requires action.source=next_state")
    if _requires_robot_cfg(config) and robot_cfg is None:
        raise ValueError("TCP pose conversion requires robot_cfg")

    data_path = Path(data_dir)
    episode_dirs = _episode_dirs(data_path)
    if not episode_dirs:
        raise FileNotFoundError(f"No episode directories found in {data_path}")
    logger.info("Found %s episodes in %s", len(episode_dirs), data_path)

    with EpisodeReader(episode_dirs[0]) as reader:
        state_dim = _state_dim(reader, config)
        action_dim = _action_dim(reader, config)
        tactile_force_dim = _tactile_force_dim(reader, config)
        camera_image_shapes = _camera_image_shapes(episode_dirs[0], config)
        tactile_image_shapes = _tactile_image_shapes(reader, config.tactile_image_specs)
    logger.info("LeRobot dimensions state=%s action=%s tactile_force=%s", state_dim, action_dim, tactile_force_dim)

    output_path = Path(output_dir)
    if output_path.exists():
        logger.warning("Output directory already exists: %s", output_path)
        if not overwrite:
            answer = input("Overwrite? [y/N] ").strip().lower()
            if answer != "y":
                logger.info("Aborted.")
                return
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=config.repo_id,
        root=output_path,
        robot_type=config.robot_type,
        fps=config.target_fps,
        features=_build_features(
            config,
            state_dim,
            action_dim,
            tactile_force_dim,
            camera_image_shapes,
            tactile_image_shapes,
        ),
        image_writer_threads=4,
    )
    dataset.meta.info["task_name"] = task_name
    write_info(dataset.meta.info, dataset.root)

    total_output_frames = 0
    stale_episode_report: list[dict[str, object]] = []
    for episode_index, episode_dir in enumerate(episode_dirs):
        output_frames, stale_report = _convert_episode(
            dataset=dataset,
            episode_dir=episode_dir,
            episode_number=episode_index + 1,
            num_episodes=len(episode_dirs),
            config=config,
            language_instruction=language_instruction,
            robot_cfg=robot_cfg,
        )
        total_output_frames += output_frames
        if stale_report:
            stale_episode_report.append({"episode": episode_dir.name, "stale": stale_report})
        logger.info("  Saving LeRobot episode %s/%s", episode_index + 1, len(episode_dirs))
        dataset.save_episode()
        if inspect_timestamps:
            timeline_path = write_resampling_timeline_html(
                episode_dir,
                config.target_fps,
                state_keys=config.state_keys,
                camera_names=config.camera_source_names,
                tactile_images=config.tactile_image_specs,
                tactile_force=config.tactile_force,
                include_action=config.action.source in {"sent_action", "policy_action"},
            )
            logger.info("  Wrote resampling timeline: %s", timeline_path.name)
        logger.info("  Saved %s frames", output_frames)

    dataset.finalize()
    readme_path = write_lerobot_dataset_card(
        output_path,
        config,
        task_name=task_name,
        language_instruction=language_instruction,
        config_path=config_path,
        raw_data_dir=str(data_path),
    )
    logger.success(
        "Conversion complete: %s episodes, %s frames @ %sHz, dataset=%s",
        len(episode_dirs),
        total_output_frames,
        config.target_fps,
        output_path,
    )
    logger.info("Generated dataset card: %s", readme_path.name)
    _log_stale_report(stale_episode_report)

    if push_to_hub:
        card_kwargs = build_lerobot_dataset_card_kwargs(
            output_path,
            config,
            task_name=task_name,
            language_instruction=language_instruction,
            config_path=config_path,
            raw_data_dir=str(data_path),
        )
        logger.info("Pushing dataset to HuggingFace Hub: %s", config.repo_id)
        dataset.push_to_hub(tags=["tianji", "sharpa"], private=False, push_videos=True, **card_kwargs)
        logger.success("Pushed to hub: %s", config.repo_id)


def _convert_episode(
    dataset,
    episode_dir: Path,
    episode_number: int,
    num_episodes: int,
    config: ConvertConfig,
    language_instruction: str,
    robot_cfg: DictConfig | None,
) -> tuple[int, dict[str, int]]:
    with EpisodeReader(episode_dir) as reader:
        include_action = config.action.source in {"sent_action", "policy_action"}
        resampled = resample_episode(
            reader,
            config.target_fps,
            state_keys=_resampling_state_keys(config),
            camera_names=config.camera_source_names,
            tactile_images=config.tactile_image_specs,
            tactile_force=config.tactile_force,
            include_action=include_action,
        )
        n_resampled = len(resampled.timestamps)
        n_output = n_resampled - 1 if config.action.source == "next_state" else n_resampled
        if n_output <= 0:
            raise ValueError(f"{episode_dir} has no output frames after resampling")

        duration_s = float(resampled.timestamps[-1] - resampled.timestamps[0])
        logger.info(
            "Episode %s/%s [%s]: duration=%.3fs resampled=%s output=%s",
            episode_number,
            num_episodes,
            episode_dir.name,
            duration_s,
            n_resampled,
            n_output,
        )

        logger.info("  Reading %s camera streams", len(config.cameras))
        state_arrays = _resampled_state_arrays(reader, resampled, config, robot_cfg=robot_cfg)
        state_values = _state_values(state_arrays, config.state, n_resampled)
        output_selection = _output_frame_selection(state_values, config)
        n_output = len(output_selection.source_indices)
        if n_output <= 0:
            raise ValueError(f"{episode_dir} has no output frames after frame filtering")
        if output_selection.dropped_frames > 0:
            logger.info(
                "  Frame filter dropped %s/%s resampled frames, output=%s",
                output_selection.dropped_frames,
                n_resampled,
                n_output,
            )
        action_arrays = _resampled_action_arrays(reader, resampled, config, robot_cfg=robot_cfg)
        tactile_force_arrays = _resampled_tactile_force_arrays(reader, resampled, config)
        camera_frames = _read_episode_camera_frames(
            episode_dir=episode_dir,
            resampled=resampled,
            config=config,
            output_indices=output_selection.source_indices,
        )
        logger.info("  Reading %s tactile image streams", len(config.tactile_images))
        tactile_frames = _read_episode_tactile_frames(
            episode_dir=episode_dir,
            resampled=resampled,
            config=config,
            output_indices=output_selection.source_indices,
        )
        stale_report = _stale_report(resampled)

    for frame_index in tqdm(
        range(n_output),
        desc=f"Episode {episode_number}/{num_episodes} add_frame",
        unit="frame",
        leave=False,
    ):
        source_index = int(output_selection.source_indices[frame_index])
        target_index = (
            None
            if output_selection.target_indices is None
            else int(output_selection.target_indices[frame_index])
        )
        frame = {
            "observation.state": state_values[source_index],
            "action": _action_at(action_arrays, source_index, config.action, target_index=target_index),
            "task": language_instruction,
        }
        if tactile_force_arrays is not None:
            frame["observation.tactile_force"] = _concat_arrays_at(tactile_force_arrays, source_index)
        for camera_name, camera_cfg in config.cameras.items():
            frame[f"observation.images.{camera_name}"] = _transform_image(
                camera_frames[camera_name][frame_index],
                camera_cfg,
            )
        for tactile_name, tactile_cfg in config.tactile_images.items():
            frame[f"observation.images.{tactile_name}"] = _transform_tactile_image(
                tactile_frames[tactile_name][frame_index],
                tactile_cfg,
            )
        dataset.add_frame(frame)

    return n_output, stale_report


def _episode_dirs(data_path: Path) -> list[Path]:
    return sorted(
        [path for path in data_path.iterdir() if path.is_dir() and path.name.startswith("episode_")],
        key=lambda path: int(path.name.split("_")[1]),
    )


def _resampling_state_keys(config: ConvertConfig) -> tuple[FieldSpec, ...]:
    specs = list(config.state.source_specs)
    if config.action.source == "next_state":
        specs.extend(config.action.source_specs)
    return _unique_field_specs(tuple(specs))


def _unique_field_specs(specs: tuple[FieldSpec, ...]) -> tuple[FieldSpec, ...]:
    output = []
    seen = set()
    for spec in specs:
        if spec.token not in seen:
            output.append(spec)
            seen.add(spec.token)
    return tuple(output)


def _field_specs_dim(reader: EpisodeReader, specs: tuple[FieldSpec, ...]) -> int:
    return sum(_sample_dim(reader.stream_field(spec.stream, spec.field)) for spec in specs)


def _state_dim(reader: EpisodeReader, config: ConvertConfig) -> int:
    if config.state.type == "field_values":
        return _field_specs_dim(reader, config.state.keys)
    if config.state.type == TCP_POSE_WORLD_WITH_HAND_ALPHA_STATE:
        return len(config.state.sides) * 10
    if config.state.type == TCP_POSE_WORLD_WITH_HAND_JOINT_STATE:
        return _tcp_pose_world_hand_joint_dim(reader, config.state)
    raise NotImplementedError(f"State type {config.state.type!r} is not supported")


def _action_dim(reader: EpisodeReader, config: ConvertConfig) -> int:
    if config.action.type == DELTA_EE_WORLD_WITH_HAND_ALPHA:
        return len(config.action.sides) * 10
    if config.action.type == DELTA_EE_WORLD_WITH_HAND_JOINT:
        return _delta_ee_world_hand_joint_dim(reader, config.action)
    if config.action.type != "joint_position":
        raise NotImplementedError(f"Action type {config.action.type!r} is not supported")
    if config.action.source == "next_state":
        if not config.action.keys:
            raise ValueError("action.keys must not be empty when action.source=next_state")
        return _field_specs_dim(reader, config.action.keys)
    if config.action.source in {"sent_action", "policy_action"}:
        return _sample_dim(reader.action_values(config.action.source))
    raise ValueError("action.source must be next_state, sent_action, or policy_action")


def _tcp_pose_world_hand_joint_dim(reader: EpisodeReader, state_cfg: StateConfig) -> int:
    if state_cfg.hand_keys is None:
        raise ValueError("tcp_pose_world_with_hand_joint state is missing hand_keys")
    return sum(
        9 + _sample_dim(reader.stream_field(state_cfg.hand_keys[side].stream, state_cfg.hand_keys[side].field))
        for side in state_cfg.sides
    )


def _delta_ee_world_hand_joint_dim(reader: EpisodeReader, action_cfg: ActionConfig) -> int:
    if action_cfg.hand_keys is None:
        raise ValueError("delta_ee_world_with_hand_joint action is missing hand_keys")
    return sum(
        9 + _sample_dim(reader.stream_field(action_cfg.hand_keys[side].stream, action_cfg.hand_keys[side].field))
        for side in action_cfg.sides
    )


def _tactile_force_dim(reader: EpisodeReader, config: ConvertConfig) -> int | None:
    if config.tactile_force is None:
        return None
    return sum(
        _sample_dim(reader.tactile_field(sensor.side, sensor.finger, config.tactile_force.field))
        for sensor in config.tactile_force.sensors
    )


def _sample_dim(values: np.ndarray) -> int:
    if values.ndim == 1:
        return 1
    return int(np.prod(values.shape[1:]))


def _build_features(
    config: ConvertConfig,
    state_dim: int,
    action_dim: int,
    tactile_force_dim: int | None,
    camera_image_shapes: dict[str, tuple[int, int, int]],
    tactile_image_shapes: dict[str, tuple[int, int, int]],
) -> dict:
    tactile_image_dtype = _tactile_image_dtype(config.tactile_image_format)
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": None,
        },
    }
    if tactile_force_dim is not None:
        features["observation.tactile_force"] = {
            "dtype": "float32",
            "shape": (tactile_force_dim,),
            "names": None,
        }
    for camera_name, shape in camera_image_shapes.items():
        features[f"observation.images.{camera_name}"] = {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channel"],
        }
    for tactile_name, shape in tactile_image_shapes.items():
        features[f"observation.images.{tactile_name}"] = {
            "dtype": tactile_image_dtype,
            "shape": shape,
            "names": ["height", "width", "channel"],
        }
    return features


def _validate_tactile_image_format(tactile_image_format: str) -> None:
    if tactile_image_format not in {"mp4", "png"}:
        raise ValueError("tactile_image_format must be mp4 or png")


def _tactile_image_dtype(tactile_image_format: str) -> str:
    _validate_tactile_image_format(tactile_image_format)
    if tactile_image_format == "mp4":
        return "video"
    return "image"


def _camera_image_shapes(
    episode_dir: Path,
    config: ConvertConfig,
) -> dict[str, tuple[int, int, int]]:
    return {
        camera_name: _camera_image_shape(episode_dir, camera_name, camera_cfg)
        for camera_name, camera_cfg in config.cameras.items()
    }


def _camera_image_shape(
    episode_dir: Path,
    camera_name: str,
    camera_cfg: ConvertCameraConfig,
) -> tuple[int, int, int]:
    image = _read_video_frames(episode_dir / f"{camera_cfg.source}.mp4", np.asarray([0], dtype=np.int64))[0]
    transformed = _transform_image(image, camera_cfg)
    if transformed.ndim != 3 or transformed.shape[2] != 3:
        raise ValueError(f"camera image must be RGB HxWx3, got shape={transformed.shape}")
    return (int(transformed.shape[0]), int(transformed.shape[1]), int(transformed.shape[2]))


def _tactile_image_shapes(
    reader: EpisodeReader,
    tactile_images: tuple[ConvertTactileImageConfig, ...],
) -> dict[str, tuple[int, int, int]]:
    return {
        tactile_image.name: _tactile_image_shape(reader, tactile_image)
        for tactile_image in tactile_images
    }


def _tactile_image_shape(
    reader: EpisodeReader,
    tactile_cfg: ConvertTactileImageConfig,
) -> tuple[int, int, int]:
    if tactile_cfg.resize is not None:
        return (tactile_cfg.resize[1], tactile_cfg.resize[0], 3)
    height, width = reader.tactile_image_shape(tactile_cfg.side, tactile_cfg.finger, tactile_cfg.modality)
    return (height, width, 3)


def _resampled_field_arrays(
    reader: EpisodeReader,
    resampled: ResampledEpisode,
    specs: tuple[FieldSpec, ...],
) -> dict[str, np.ndarray]:
    arrays = {}
    for spec in specs:
        raw = reader.stream_field(spec.stream, spec.field)
        arrays[spec.token] = raw[resampled.state_indices[spec.stream]]
    return arrays


def _resampled_state_arrays(
    reader: EpisodeReader,
    resampled: ResampledEpisode,
    config: ConvertConfig,
    *,
    robot_cfg: DictConfig | None = None,
) -> dict[str, np.ndarray] | _TcpPoseWorldHandStateArrays:
    if config.state.type == "field_values":
        return _resampled_field_arrays(reader, resampled, config.state.keys)
    if config.state.type == TCP_POSE_WORLD_WITH_HAND_ALPHA_STATE:
        if robot_cfg is None:
            raise ValueError("tcp_pose_world_with_hand_alpha state requires robot_cfg")
        return _tcp_pose_world_hand_state_arrays(reader, resampled, config.state, robot_cfg)
    if config.state.type == TCP_POSE_WORLD_WITH_HAND_JOINT_STATE:
        if robot_cfg is None:
            raise ValueError("tcp_pose_world_with_hand_joint state requires robot_cfg")
        return _tcp_pose_world_hand_state_arrays(reader, resampled, config.state, robot_cfg)
    raise NotImplementedError(f"State type {config.state.type!r} is not supported")


def _resampled_action_arrays(
    reader: EpisodeReader,
    resampled: ResampledEpisode,
    config: ConvertConfig,
    *,
    robot_cfg: DictConfig | None = None,
) -> dict[str, np.ndarray] | np.ndarray | _DeltaEeWorldHandActionArrays:
    if config.action.type == DELTA_EE_WORLD_WITH_HAND_ALPHA:
        if robot_cfg is None:
            raise ValueError("delta_ee_world_with_hand_alpha conversion requires robot_cfg")
        return _delta_ee_world_hand_action_arrays(reader, resampled, config.action, robot_cfg)
    if config.action.type == DELTA_EE_WORLD_WITH_HAND_JOINT:
        if robot_cfg is None:
            raise ValueError("delta_ee_world_with_hand_joint conversion requires robot_cfg")
        return _delta_ee_world_hand_action_arrays(reader, resampled, config.action, robot_cfg)
    if config.action.source == "next_state":
        return _resampled_field_arrays(reader, resampled, config.action.keys)
    if resampled.action_indices is None:
        raise RuntimeError("resampled action indices are missing")
    return reader.action_values(config.action.source)[resampled.action_indices]


def _resampled_tactile_force_arrays(
    reader: EpisodeReader,
    resampled: ResampledEpisode,
    config: ConvertConfig,
) -> dict[str, np.ndarray] | None:
    if config.tactile_force is None:
        return None
    arrays = {}
    for sensor in config.tactile_force.sensors:
        raw = reader.tactile_field(sensor.side, sensor.finger, config.tactile_force.field)
        arrays[sensor.name] = raw[resampled.tactile_force_row_indices[sensor.name]]
    return arrays


def _state_values(
    state_arrays: dict[str, np.ndarray] | _TcpPoseWorldHandStateArrays,
    state_cfg: StateConfig,
    n_frames: int,
) -> np.ndarray:
    return np.asarray(
        [_state_at(state_arrays, index, state_cfg) for index in range(n_frames)],
        dtype=np.float32,
    )


def _output_frame_selection(state_values: np.ndarray, config: ConvertConfig) -> _OutputFrameSelection:
    kept_indices = _filtered_frame_indices(state_values, config.frame_filter, config.state)
    if config.action.source == "next_state":
        if len(kept_indices) < 2:
            raise ValueError("frame_filter kept fewer than 2 frames for next_state action")
        source_indices = kept_indices[:-1]
        target_indices = kept_indices[1:]
    else:
        source_indices = kept_indices
        target_indices = None
    return _OutputFrameSelection(
        source_indices=source_indices,
        target_indices=target_indices,
        dropped_frames=int(len(state_values) - len(kept_indices)),
    )


def _filtered_frame_indices(
    state_values: np.ndarray,
    frame_filter: FrameFilterConfig,
    state_cfg: StateConfig,
) -> np.ndarray:
    n_frames = int(len(state_values))
    if n_frames == 0:
        return np.asarray([], dtype=np.int64)
    if not frame_filter.enabled:
        return np.arange(n_frames, dtype=np.int64)
    if frame_filter.type != "stationary_tcp_pose_world_with_hand_alpha":
        raise ValueError("enabled frame_filter requires type=stationary_tcp_pose_world_with_hand_alpha")
    if state_cfg.type != TCP_POSE_WORLD_WITH_HAND_ALPHA_STATE:
        raise ValueError("stationary_tcp_pose_world_with_hand_alpha frame_filter requires matching state type")
    if state_values.shape[1] != len(state_cfg.sides) * 10:
        raise ValueError("tcp_pose_world_with_hand_alpha state dimension does not match sides")
    stationary = _stationary_tcp_pose_world_intervals(state_values, frame_filter, state_cfg)
    keep = np.ones(n_frames, dtype=np.bool_)
    run_start = None
    for index, is_stationary in enumerate((*stationary.tolist(), False)):
        if is_stationary and run_start is None:
            run_start = index
        elif not is_stationary and run_start is not None:
            run_end = index - 1
            run_length = run_end - run_start + 1
            if run_length >= frame_filter.min_run_frames:
                keep[run_start + 1 : run_end + 2] = False
            run_start = None
    return np.flatnonzero(keep).astype(np.int64)


def _stationary_tcp_pose_world_intervals(
    state_values: np.ndarray,
    frame_filter: FrameFilterConfig,
    state_cfg: StateConfig,
) -> np.ndarray:
    if len(state_values) < 2:
        return np.asarray([], dtype=np.bool_)
    delta = state_values[1:].astype(np.float64) - state_values[:-1].astype(np.float64)
    max_xyz = np.zeros(len(delta), dtype=np.float64)
    max_rotation_6d = np.zeros(len(delta), dtype=np.float64)
    max_hand_alpha = np.zeros(len(delta), dtype=np.float64)
    for side_index in range(len(state_cfg.sides)):
        offset = side_index * 10
        max_xyz = np.maximum(max_xyz, np.linalg.norm(delta[:, offset : offset + 3], axis=1))
        max_rotation_6d = np.maximum(
            max_rotation_6d,
            np.linalg.norm(delta[:, offset + 3 : offset + 9], axis=1),
        )
        max_hand_alpha = np.maximum(max_hand_alpha, np.abs(delta[:, offset + 9]))
    return (
        (max_xyz < frame_filter.max_tcp_translation_m)
        & (max_rotation_6d < frame_filter.max_rotation_6d)
        & (max_hand_alpha < frame_filter.max_hand_alpha)
    )


def _read_episode_camera_frames(
    episode_dir: Path,
    resampled: ResampledEpisode,
    config: ConvertConfig,
    output_indices: np.ndarray,
) -> dict[str, list[np.ndarray]]:
    frames = {}
    for camera_name, camera_cfg in config.cameras.items():
        mp4_path = episode_dir / f"{camera_cfg.source}.mp4"
        frames[camera_name] = _read_video_frames(
            mp4_path,
            resampled.camera_frame_indices[camera_cfg.source][output_indices],
        )
    return frames


def _read_episode_tactile_frames(
    episode_dir: Path,
    resampled: ResampledEpisode,
    config: ConvertConfig,
    output_indices: np.ndarray,
) -> dict[str, list[np.ndarray]]:
    frames = {}
    for tactile_name, tactile_cfg in config.tactile_images.items():
        video_path = episode_dir / "tactile" / f"{tactile_cfg.side}_{tactile_cfg.finger}_{tactile_cfg.modality}.mkv"
        frames[tactile_name] = _read_video_frames(
            video_path,
            resampled.tactile_frame_indices[tactile_name][output_indices],
        )
    return frames


def _read_video_frames(video_path: Path, frame_indices: np.ndarray) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"failed to open video: {video_path}")
    frames = []
    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            capture.release()
            raise RuntimeError(f"failed to read frame {int(frame_index)} from {video_path}")
        frames.append(_video_frame_to_rgb(frame_bgr))
    capture.release()
    return frames


def _video_frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return np.repeat(frame[:, :, None], 3, axis=2).copy()
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame[..., ::-1].copy()
    raise ValueError(f"video frame must be grayscale or BGR HxWx3, got shape={frame.shape}")


def _transform_image(image: np.ndarray, camera_cfg: ConvertCameraConfig) -> np.ndarray:
    crop = camera_cfg.crop
    h_end = image.shape[0] - crop.bottom if crop.bottom > 0 else image.shape[0]
    w_end = image.shape[1] - crop.right if crop.right > 0 else image.shape[1]
    cropped = image[crop.top:h_end, crop.left:w_end]
    if cropped.size == 0:
        raise ValueError("camera crop produced an empty image")
    if camera_cfg.resize is None:
        return np.asarray(cropped, dtype=np.uint8)
    resized = cv2.resize(cropped, camera_cfg.resize, interpolation=cv2.INTER_AREA)
    return np.asarray(resized, dtype=np.uint8)


def _transform_tactile_image(image: np.ndarray, tactile_cfg: ConvertTactileImageConfig) -> np.ndarray:
    image_rgb = _ensure_rgb_image(image)
    if tactile_cfg.resize is None:
        return np.asarray(image_rgb, dtype=np.uint8)
    resized = cv2.resize(image_rgb, tactile_cfg.resize, interpolation=cv2.INTER_AREA)
    return np.asarray(resized, dtype=np.uint8)


def _ensure_rgb_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"tactile image must be grayscale or RGB HxWx3, got shape={image.shape}")


def _concat_arrays_at(arrays: dict[str, np.ndarray], index: int) -> np.ndarray:
    parts = [np.asarray(value[index], dtype=np.float32).reshape(-1) for value in arrays.values()]
    return np.concatenate(parts).astype(np.float32)


def _state_at(
    state_arrays: dict[str, np.ndarray] | _TcpPoseWorldHandStateArrays,
    index: int,
    state_cfg: StateConfig,
) -> np.ndarray:
    if state_cfg.type == "field_values":
        if not isinstance(state_arrays, dict):
            raise TypeError("field_values state arrays must be a dict")
        return _concat_arrays_at(state_arrays, index)
    if state_cfg.type == TCP_POSE_WORLD_WITH_HAND_ALPHA_STATE:
        return _tcp_pose_world_hand_alpha_state_at(state_arrays, index, state_cfg)
    if state_cfg.type == TCP_POSE_WORLD_WITH_HAND_JOINT_STATE:
        return _tcp_pose_world_hand_joint_state_at(state_arrays, index, state_cfg)
    raise NotImplementedError(f"State type {state_cfg.type!r} is not supported")


def _action_at(
    action_arrays: dict[str, np.ndarray] | np.ndarray | _DeltaEeWorldHandActionArrays,
    index: int,
    action: ActionConfig | str,
    target_index: int | None = None,
) -> np.ndarray:
    if isinstance(action, str):
        action_type = "joint_position"
        action_source = action
    else:
        action_type = action.type
        action_source = action.source
    if action_type == DELTA_EE_WORLD_WITH_HAND_ALPHA:
        if isinstance(action, str):
            raise TypeError("delta_ee_world_with_hand_alpha action requires ActionConfig")
        action_target_index = index + 1 if target_index is None else int(target_index)
        return _delta_ee_world_hand_alpha_action_at(action_arrays, index, action_target_index, action)
    if action_type == DELTA_EE_WORLD_WITH_HAND_JOINT:
        if isinstance(action, str):
            raise TypeError("delta_ee_world_with_hand_joint action requires ActionConfig")
        action_target_index = index + 1 if target_index is None else int(target_index)
        return _delta_ee_world_hand_joint_action_at(action_arrays, index, action_target_index, action)
    if action_type != "joint_position":
        raise NotImplementedError(f"Action type {action_type!r} is not supported")
    if action_source == "next_state":
        if not isinstance(action_arrays, dict):
            raise TypeError("next_state action arrays must be a dict")
        action_target_index = index + 1 if target_index is None else int(target_index)
        return _concat_arrays_at(action_arrays, action_target_index)
    if isinstance(action_arrays, dict):
        raise TypeError("recorded action arrays must be an ndarray")
    return np.asarray(action_arrays[index], dtype=np.float32).reshape(-1)


def _tcp_pose_world_hand_state_arrays(
    reader: EpisodeReader,
    resampled: ResampledEpisode,
    state_cfg: StateConfig,
    robot_cfg: DictConfig,
) -> _TcpPoseWorldHandStateArrays:
    if state_cfg.arm_keys is None or state_cfg.hand_keys is None:
        raise ValueError(f"{state_cfg.type} state is missing arm_keys or hand_keys")
    source_arrays = _resampled_field_arrays(reader, resampled, state_cfg.source_specs)
    arm_q_by_side = {
        side: source_arrays[state_cfg.arm_keys[side].token]
        for side in state_cfg.sides
    }
    hand_q_by_side = {
        side: source_arrays[state_cfg.hand_keys[side].token]
        for side in state_cfg.sides
    }
    mappers_by_side = {
        side: TianjiTcpPoseMapper.fk_from_config(robot_cfg.hardware.arm, side)
        for side in state_cfg.sides
    }
    return _TcpPoseWorldHandStateArrays(
        arm_q_by_side=arm_q_by_side,
        hand_q_by_side=hand_q_by_side,
        mappers_by_side=mappers_by_side,
    )


def _tcp_pose_world_hand_alpha_state_at(
    state_arrays: dict[str, np.ndarray] | _TcpPoseWorldHandStateArrays,
    index: int,
    state_cfg: StateConfig,
) -> np.ndarray:
    if not isinstance(state_arrays, _TcpPoseWorldHandStateArrays):
        raise TypeError("tcp_pose_world_with_hand_alpha state arrays have unexpected type")
    if state_cfg.hand_alpha is None:
        raise ValueError("tcp_pose_world_with_hand_alpha state is missing hand_alpha")
    values: list[float] = []
    for side in state_cfg.sides:
        current_arm_q = tuple(float(value) for value in state_arrays.arm_q_by_side[side][index])
        tcp_pose_world = np.asarray(
            state_arrays.mappers_by_side[side].tcp_pose_world_from_q_rad(current_arm_q),
            dtype=np.float64,
        )
        values.extend(float(value) for value in tcp_pose_world[:3, 3])
        values.extend(rotation_6d_from_matrix(tcp_pose_world[:3, :3]))
        current_hand_q = tuple(float(value) for value in state_arrays.hand_q_by_side[side][index])
        values.append(
            hand_state_alpha(
                current_hand_q,
                state_cfg.hand_alpha.open_state[side],
                state_cfg.hand_alpha.grasp_state[side],
            )
        )
    return np.asarray(values, dtype=np.float32)


def _tcp_pose_world_hand_joint_state_at(
    state_arrays: dict[str, np.ndarray] | _TcpPoseWorldHandStateArrays,
    index: int,
    state_cfg: StateConfig,
) -> np.ndarray:
    if not isinstance(state_arrays, _TcpPoseWorldHandStateArrays):
        raise TypeError("tcp_pose_world_with_hand_joint state arrays have unexpected type")
    values: list[float] = []
    for side in state_cfg.sides:
        current_arm_q = tuple(float(value) for value in state_arrays.arm_q_by_side[side][index])
        tcp_pose_world = np.asarray(
            state_arrays.mappers_by_side[side].tcp_pose_world_from_q_rad(current_arm_q),
            dtype=np.float64,
        )
        values.extend(float(value) for value in tcp_pose_world[:3, 3])
        values.extend(rotation_6d_from_matrix(tcp_pose_world[:3, :3]))
        values.extend(float(value) for value in np.asarray(state_arrays.hand_q_by_side[side][index]).reshape(-1))
    return np.asarray(values, dtype=np.float32)


def _delta_ee_world_hand_action_arrays(
    reader: EpisodeReader,
    resampled: ResampledEpisode,
    action_cfg: ActionConfig,
    robot_cfg: DictConfig,
) -> _DeltaEeWorldHandActionArrays:
    if action_cfg.arm_keys is None or action_cfg.hand_keys is None:
        raise ValueError(f"{action_cfg.type} action is missing arm_keys or hand_keys")
    source_arrays = _resampled_field_arrays(reader, resampled, action_cfg.source_specs)
    arm_q_by_side = {
        side: source_arrays[action_cfg.arm_keys[side].token]
        for side in action_cfg.sides
    }
    hand_q_by_side = {
        side: source_arrays[action_cfg.hand_keys[side].token]
        for side in action_cfg.sides
    }
    mappers_by_side = {
        side: TianjiTcpPoseMapper.fk_from_config(robot_cfg.hardware.arm, side)
        for side in action_cfg.sides
    }
    return _DeltaEeWorldHandActionArrays(
        arm_q_by_side=arm_q_by_side,
        hand_q_by_side=hand_q_by_side,
        mappers_by_side=mappers_by_side,
    )


def _delta_ee_world_hand_alpha_action_at(
    action_arrays: dict[str, np.ndarray] | np.ndarray | _DeltaEeWorldHandActionArrays,
    index: int,
    target_index: int,
    action_cfg: ActionConfig,
) -> np.ndarray:
    if not isinstance(action_arrays, _DeltaEeWorldHandActionArrays):
        raise TypeError("delta_ee_world_with_hand_alpha action arrays have unexpected type")
    if action_cfg.hand_alpha is None:
        raise ValueError("delta_ee_world_with_hand_alpha action is missing hand_alpha")
    values: list[float] = []
    for side in action_cfg.sides:
        current_arm_q = tuple(float(value) for value in action_arrays.arm_q_by_side[side][index])
        target_arm_q = tuple(float(value) for value in action_arrays.arm_q_by_side[side][target_index])
        values.extend(action_arrays.mappers_by_side[side].delta_world_from_q_rad(current_arm_q, target_arm_q))
        target_hand_q = tuple(float(value) for value in action_arrays.hand_q_by_side[side][target_index])
        values.append(
            hand_state_alpha(
                target_hand_q,
                action_cfg.hand_alpha.open_state[side],
                action_cfg.hand_alpha.grasp_state[side],
            )
        )
    return np.asarray(values, dtype=np.float32)


def _delta_ee_world_hand_joint_action_at(
    action_arrays: dict[str, np.ndarray] | np.ndarray | _DeltaEeWorldHandActionArrays,
    index: int,
    target_index: int,
    action_cfg: ActionConfig,
) -> np.ndarray:
    if not isinstance(action_arrays, _DeltaEeWorldHandActionArrays):
        raise TypeError("delta_ee_world_with_hand_joint action arrays have unexpected type")
    values: list[float] = []
    for side in action_cfg.sides:
        current_arm_q = tuple(float(value) for value in action_arrays.arm_q_by_side[side][index])
        target_arm_q = tuple(float(value) for value in action_arrays.arm_q_by_side[side][target_index])
        values.extend(action_arrays.mappers_by_side[side].delta_world_from_q_rad(current_arm_q, target_arm_q))
        values.extend(float(value) for value in np.asarray(action_arrays.hand_q_by_side[side][target_index]).reshape(-1))
    return np.asarray(values, dtype=np.float32)


def _requires_robot_cfg(config: ConvertConfig) -> bool:
    return (
        config.state.type in {TCP_POSE_WORLD_WITH_HAND_ALPHA_STATE, TCP_POSE_WORLD_WITH_HAND_JOINT_STATE}
        or config.action.type in {DELTA_EE_WORLD_WITH_HAND_ALPHA, DELTA_EE_WORLD_WITH_HAND_JOINT}
    )


def _stale_report(resampled: ResampledEpisode) -> dict[str, int]:
    report = {}
    for stream_name, match in resampled.matches.items():
        stale_count = int(np.sum(match.stale_mask))
        if stale_count > 0:
            report[stream_name] = stale_count
            logger.warning(
                "  %s stale frames: %s/%s max_gap_ms=%.3f",
                stream_name,
                stale_count,
                len(match.stale_mask),
                float(np.max(match.gaps) * 1000.0),
            )
    return report


def _log_stale_report(stale_episode_report: list[dict[str, object]]) -> None:
    if not stale_episode_report:
        logger.info("All episodes passed stale frame check.")
        return
    logger.warning("Stale frame report: %s affected episodes", len(stale_episode_report))
    for item in stale_episode_report:
        logger.warning("%s: %s", item["episode"], item["stale"])
