from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from unirobot_infra.dataset.convert_config import ConvertTactileForceConfig, ConvertTactileImageConfig, FieldSpec
from unirobot_infra.dataset.episode_reader import EpisodeReader


@dataclass(frozen=True)
class StreamMatch:
    raw_times: np.ndarray
    indices: np.ndarray
    gaps: np.ndarray
    stale_mask: np.ndarray


@dataclass(frozen=True)
class ResampledEpisode:
    timestamps: np.ndarray
    state_indices: dict[str, np.ndarray]
    camera_frame_indices: dict[str, np.ndarray]
    camera_row_indices: dict[str, np.ndarray]
    tactile_frame_indices: dict[str, np.ndarray]
    tactile_row_indices: dict[str, np.ndarray]
    tactile_force_row_indices: dict[str, np.ndarray]
    action_indices: np.ndarray | None
    matches: dict[str, StreamMatch]


def resample_episode(
    reader: EpisodeReader,
    target_fps: int,
    *,
    state_keys: tuple[FieldSpec, ...],
    camera_names: tuple[str, ...] = (),
    tactile_images: tuple[ConvertTactileImageConfig, ...] = (),
    tactile_force: ConvertTactileForceConfig | None = None,
    include_action: bool = False,
    stale_threshold_s: float | None = None,
) -> ResampledEpisode:
    if int(target_fps) <= 0:
        raise ValueError("target_fps must be positive")
    state_streams = _unique_streams(state_keys)
    if not state_streams:
        raise ValueError("state_keys must not be empty")

    stream_times = {stream_name: reader.stream_times(stream_name) for stream_name in state_streams}
    camera_times = {camera_name: reader.camera_times(camera_name) for camera_name in camera_names}
    tactile_times = {
        tactile_image.name: reader.tactile_times(tactile_image.side, tactile_image.finger)
        for tactile_image in tactile_images
    }
    tactile_force_times = {}
    if tactile_force is not None:
        tactile_force_times = {
            sensor.name: reader.tactile_times(sensor.side, sensor.finger)
            for sensor in tactile_force.sensors
        }
    action_times = reader.action_times() if include_action else None

    _validate_non_empty_times(stream_times)
    _validate_non_empty_times(camera_times)
    _validate_non_empty_times(tactile_times)
    _validate_non_empty_times(tactile_force_times)
    if action_times is not None and len(action_times) == 0:
        raise ValueError("action time series must not be empty")

    t_start = max(
        [float(times[0]) for times in stream_times.values()]
        + [float(times[0]) for times in camera_times.values()]
        + [float(times[0]) for times in tactile_times.values()]
        + [float(times[0]) for times in tactile_force_times.values()]
        + ([float(action_times[0])] if action_times is not None else [])
    )
    t_end = min(
        [float(times[-1]) for times in stream_times.values()]
        + [float(times[-1]) for times in camera_times.values()]
        + [float(times[-1]) for times in tactile_times.values()]
        + [float(times[-1]) for times in tactile_force_times.values()]
        + ([float(action_times[-1])] if action_times is not None else [])
    )
    if t_end <= t_start:
        raise ValueError("episode streams have no overlapping time window")

    n_frames = int((t_end - t_start) * int(target_fps)) + 1
    grid = t_start + np.arange(n_frames, dtype=np.float64) / float(target_fps)
    threshold_s = 2.0 / float(target_fps) if stale_threshold_s is None else float(stale_threshold_s)

    matches: dict[str, StreamMatch] = {}
    state_indices = {}
    for stream_name, times in stream_times.items():
        indices = _nearest_indices(times, grid)
        state_indices[stream_name] = indices
        matches[f"state:{stream_name}"] = _stream_match(times, grid, indices, threshold_s)

    camera_frame_indices = {}
    camera_row_indices = {}
    for camera_name, times in camera_times.items():
        indices = _nearest_indices(times, grid)
        camera_row_indices[camera_name] = indices
        camera_frame_indices[camera_name] = reader.camera_frame_indices(camera_name)[indices]
        matches[f"camera:{camera_name}"] = _stream_match(times, grid, indices, threshold_s)

    tactile_row_indices = {}
    tactile_frame_indices = {}
    for tactile_image in tactile_images:
        times = tactile_times[tactile_image.name]
        indices = _nearest_indices(times, grid)
        tactile_row_indices[tactile_image.name] = indices
        tactile_frame_indices[tactile_image.name] = reader.tactile_frame_indices(
            tactile_image.side,
            tactile_image.finger,
            tactile_image.modality,
        )[indices]
        matches[f"tactile:{tactile_image.name}"] = _stream_match(times, grid, indices, threshold_s)

    tactile_force_row_indices = {}
    for name, times in tactile_force_times.items():
        indices = _nearest_indices(times, grid)
        tactile_force_row_indices[name] = indices
        matches[f"tactile_force:{name}"] = _stream_match(times, grid, indices, threshold_s)

    action_indices = None
    if action_times is not None:
        action_indices = _nearest_indices(action_times, grid)
        matches["action"] = _stream_match(action_times, grid, action_indices, threshold_s)

    return ResampledEpisode(
        timestamps=grid,
        state_indices=state_indices,
        camera_frame_indices=camera_frame_indices,
        camera_row_indices=camera_row_indices,
        tactile_frame_indices=tactile_frame_indices,
        tactile_row_indices=tactile_row_indices,
        tactile_force_row_indices=tactile_force_row_indices,
        action_indices=action_indices,
        matches=matches,
    )


def _unique_streams(state_keys: tuple[FieldSpec, ...]) -> tuple[str, ...]:
    streams = []
    for spec in state_keys:
        if spec.stream not in streams:
            streams.append(spec.stream)
    return tuple(streams)


def _validate_non_empty_times(times_by_name: dict[str, np.ndarray]) -> None:
    for stream_name, times in times_by_name.items():
        if len(times) == 0:
            raise ValueError(f"{stream_name} time series must not be empty")


def _nearest_indices(times: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if len(times) == 1:
        return np.zeros(len(grid), dtype=np.int64)
    indices = np.searchsorted(times, grid, side="left")
    indices = np.clip(indices, 1, len(times) - 1)
    left = np.abs(times[indices - 1] - grid)
    right = np.abs(times[indices] - grid)
    return np.where(left <= right, indices - 1, indices)


def _stream_match(
    raw_times: np.ndarray,
    grid: np.ndarray,
    indices: np.ndarray,
    stale_threshold_s: float,
) -> StreamMatch:
    gaps = np.abs(raw_times[indices] - grid)
    return StreamMatch(
        raw_times=raw_times,
        indices=indices,
        gaps=gaps,
        stale_mask=gaps > float(stale_threshold_s),
    )
