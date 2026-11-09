from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from unirobot_infra.core.units import ns_to_s


class EpisodeReader:
    def __init__(self, episode_dir: str | Path) -> None:
        self.episode_dir = Path(episode_dir)
        self._h5 = h5py.File(self.episode_dir / "robot_states.h5", "r")

    @property
    def stream_names(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self._h5["robot_state"].keys())

    @property
    def camera_names(self) -> tuple[str, ...]:
        if "camera" not in self._h5:
            return ()
        return tuple(str(name) for name in self._h5["camera"].keys())

    @property
    def has_actions(self) -> bool:
        return "action" in self._h5 and "sent_action" in self._h5["action"]

    def stream_times(self, stream_name: str) -> np.ndarray:
        return _ns_to_seconds(self._h5[f"robot_state/{stream_name}/sys_time_ns"][:])

    def stream_field(self, stream_name: str, field_name: str) -> np.ndarray:
        return self._h5[f"robot_state/{stream_name}/{field_name}"][:]

    def camera_times(self, camera_name: str) -> np.ndarray:
        return _ns_to_seconds(self._h5[f"camera/{camera_name}/sys_time_ns"][:])

    def camera_frame_indices(self, camera_name: str) -> np.ndarray:
        return self._h5[f"camera/{camera_name}/frame_idx"][:]

    def tactile_times(self, side: str, finger: str) -> np.ndarray:
        return _ns_to_seconds(self._h5[f"tactile/{side}/{finger}/sys_time_ns"][:])

    def tactile_frame_indices(self, side: str, finger: str, modality: str) -> np.ndarray:
        indices = self._h5[f"tactile/{side}/{finger}/{modality}_frame_idx"][:]
        if np.any(indices < 0):
            raise ValueError(f"tactile/{side}/{finger}/{modality} contains missing frame indices")
        return indices

    def tactile_image_shape(self, side: str, finger: str, modality: str) -> tuple[int, int]:
        shapes = self._h5[f"tactile/{side}/{finger}/{modality}_shape"][:]
        if len(shapes) == 0:
            raise ValueError(f"tactile/{side}/{finger}/{modality} shape series must not be empty")
        shape = tuple(int(value) for value in shapes[0])
        if len(shape) != 2:
            raise ValueError(f"tactile/{side}/{finger}/{modality} image shape must be 2D")
        return shape

    def tactile_field(self, side: str, finger: str, field_name: str) -> np.ndarray:
        return _to_dense_rows(self._h5[f"tactile/{side}/{finger}/{field_name}"][:], f"tactile/{side}/{finger}/{field_name}")

    def action_times(self) -> np.ndarray:
        return _ns_to_seconds(self._h5["action/sys_time_ns"][:])

    def action_values(self, key: str) -> np.ndarray:
        return self._h5[f"action/{key}"][:]

    def close(self) -> None:
        self._h5.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def _ns_to_seconds(values: np.ndarray) -> np.ndarray:
    return np.asarray([ns_to_s(int(value)) for value in values], dtype=np.float64)


def _to_dense_rows(values: np.ndarray, name: str) -> np.ndarray:
    if values.dtype != object:
        return values
    if len(values) == 0:
        return np.empty((0, 0), dtype=np.float32)
    rows = [np.asarray(value, dtype=np.float32).reshape(-1) for value in values]
    width = rows[0].shape[0]
    for row in rows:
        if row.shape[0] != width:
            raise ValueError(f"{name} rows must have consistent width")
    return np.stack(rows, axis=0)
