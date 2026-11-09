from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np

from unirobot_infra.core.types import SharpaTactileFrame
from unirobot_infra.utils.logging import get_logger


logger = get_logger(__name__)

_STREAM_CHUNK = 1024
_ACTION_CHUNK = 512
_CAMERA_CHUNK = 256
_TACTILE_CHUNK = 256
_STRING_DTYPE = h5py.string_dtype(encoding="utf-8")
_VLEN_FLOAT32_DTYPE = h5py.vlen_dtype(np.dtype("float32"))
_VLEN_INT64_DTYPE = h5py.vlen_dtype(np.dtype("int64"))


class HDF5EpisodeWriter:
    def __init__(self, path: str | Path, camera_names: Sequence[str] = ()) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.path, "w", track_order=True)
        self._streams: dict[str, _StreamWriter] = {}
        self._actions = _ActionWriter(self._file)
        self._cameras = {
            str(camera_name): _CameraTimestampWriter(self._file, str(camera_name))
            for camera_name in camera_names
        }
        self._tactile = _TactileMetadataWriter(self._file)
        logger.info("recording HDF5 opened: %s", self.path)

    def append_stream_sample(
        self,
        stream_name: str,
        sys_time_ns: int,
        fields: Mapping[str, Any],
    ) -> None:
        if stream_name not in self._streams:
            self._streams[stream_name] = _StreamWriter(self._file, stream_name, fields)
        self._streams[stream_name].append(sys_time_ns, fields)

    def append_stream_samples(
        self,
        stream_name: str,
        sys_time_ns: Sequence[int],
        fields: Sequence[Mapping[str, Any]],
    ) -> None:
        if not fields:
            return
        if stream_name not in self._streams:
            self._streams[stream_name] = _StreamWriter(self._file, stream_name, fields[0])
        self._streams[stream_name].append_batch(sys_time_ns, fields)

    def append_action_sample(
        self,
        sys_time_ns: int,
        policy_action: Sequence[float],
        sent_action: Sequence[float],
        teleop_active: bool,
    ) -> None:
        self._actions.append(
            sys_time_ns=sys_time_ns,
            policy_action=policy_action,
            sent_action=sent_action,
            teleop_active=teleop_active,
        )

    def append_action_samples(
        self,
        sys_time_ns: Sequence[int],
        policy_actions: Sequence[Sequence[float]],
        sent_actions: Sequence[Sequence[float]],
        teleop_active: Sequence[bool],
    ) -> None:
        self._actions.append_batch(
            sys_time_ns=sys_time_ns,
            policy_actions=policy_actions,
            sent_actions=sent_actions,
            teleop_active=teleop_active,
        )

    def append_camera_frame_sample(self, camera_name: str, sys_time_ns: int, frame_idx: int) -> None:
        self._cameras[camera_name].append(sys_time_ns=sys_time_ns, frame_idx=frame_idx)

    def append_camera_frame_samples(
        self,
        camera_name: str,
        sys_time_ns: Sequence[int],
        frame_idx: Sequence[int],
    ) -> None:
        self._cameras[camera_name].append_batch(sys_time_ns=sys_time_ns, frame_idx=frame_idx)

    def append_tactile_frame_sample(
        self,
        frame: SharpaTactileFrame,
        raw_frame_idx: int,
        deform_frame_idx: int,
    ) -> None:
        self._tactile.append(
            frame=frame,
            raw_frame_idx=raw_frame_idx,
            deform_frame_idx=deform_frame_idx,
        )

    def append_tactile_frame_samples(
        self,
        frames: Sequence[SharpaTactileFrame],
        raw_frame_idx: Sequence[int],
        deform_frame_idx: Sequence[int],
    ) -> None:
        self._tactile.append_batch(
            frames=frames,
            raw_frame_idx=raw_frame_idx,
            deform_frame_idx=deform_frame_idx,
        )

    def close(self) -> None:
        for stream in self._streams.values():
            stream.close()
        self._actions.close()
        for camera in self._cameras.values():
            camera.close()
        self._tactile.close()
        self._file.flush()
        self._file.close()
        logger.info("recording HDF5 closed: %s", self.path)


class _StreamWriter:
    def __init__(self, h5_file: h5py.File, stream_name: str, fields: Mapping[str, Any]) -> None:
        self._group = h5_file.create_group(f"robot_state/{stream_name}", track_order=True)
        field_order = ("sys_time_ns", *fields.keys())
        self._group.attrs["field_order"] = np.asarray(field_order, dtype=_STRING_DTYPE)
        self._datasets: dict[str, h5py.Dataset] = {
            "sys_time_ns": self._group.create_dataset(
                "sys_time_ns",
                shape=(_STREAM_CHUNK,),
                maxshape=(None,),
                dtype=np.int64,
                chunks=(_STREAM_CHUNK,),
            )
        }
        for name, value in fields.items():
            self._datasets[name] = _create_dataset(self._group, name, value, _STREAM_CHUNK)
        self._count = 0
        self._capacity = _STREAM_CHUNK

    @property
    def count(self) -> int:
        return self._count

    def append(self, sys_time_ns: int, fields: Mapping[str, Any]) -> None:
        self._ensure_capacity(self._count + 1)
        index = self._count
        self._datasets["sys_time_ns"][index] = int(sys_time_ns)
        for name, value in fields.items():
            self._datasets[name][index] = _dataset_value(value)
        self._count = index + 1

    def append_batch(self, sys_time_ns: Sequence[int], fields: Sequence[Mapping[str, Any]]) -> None:
        count = len(fields)
        if count == 0:
            return
        start = self._count
        stop = start + count
        self._ensure_capacity(stop)
        self._datasets["sys_time_ns"][start:stop] = np.asarray(sys_time_ns, dtype=np.int64)
        for name in fields[0]:
            values = [_dataset_value(sample[name]) for sample in fields]
            self._assign_values(self._datasets[name], start, stop, values)
        self._count = stop

    def close(self) -> None:
        for dataset in self._datasets.values():
            dataset.resize(self._count, axis=0)

    def _grow(self) -> None:
        self._ensure_capacity(self._capacity + _STREAM_CHUNK)

    def _ensure_capacity(self, needed: int) -> None:
        if needed <= self._capacity:
            return
        while needed > self._capacity:
            self._capacity += _STREAM_CHUNK
        for dataset in self._datasets.values():
            dataset.resize(self._capacity, axis=0)

    def _assign_values(self, dataset: h5py.Dataset, start: int, stop: int, values: Sequence[Any]) -> None:
        if isinstance(values[0], str):
            dataset[start:stop] = values
            return
        dataset[start:stop] = np.asarray(values)


class _ActionWriter:
    def __init__(self, h5_file: h5py.File) -> None:
        self._group = h5_file.create_group("action", track_order=True)
        self._group.attrs["field_order"] = np.asarray(
            ("sys_time_ns", "policy_action", "sent_action", "teleop_active"),
            dtype=_STRING_DTYPE,
        )
        self._datasets: dict[str, h5py.Dataset] | None = None
        self._count = 0
        self._capacity = _ACTION_CHUNK

    def append(
        self,
        sys_time_ns: int,
        policy_action: Sequence[float],
        sent_action: Sequence[float],
        teleop_active: bool,
    ) -> None:
        if self._datasets is None:
            self._datasets = {
                "sys_time_ns": self._group.create_dataset(
                    "sys_time_ns",
                    shape=(_ACTION_CHUNK,),
                    maxshape=(None,),
                    dtype=np.int64,
                    chunks=(_ACTION_CHUNK,),
                ),
                "policy_action": _create_dataset(self._group, "policy_action", tuple(policy_action), _ACTION_CHUNK),
                "sent_action": _create_dataset(self._group, "sent_action", tuple(sent_action), _ACTION_CHUNK),
                "teleop_active": self._group.create_dataset(
                    "teleop_active",
                    shape=(_ACTION_CHUNK,),
                    maxshape=(None,),
                    dtype=np.bool_,
                    chunks=(_ACTION_CHUNK,),
                ),
            }
        self._ensure_capacity(self._count + 1)
        index = self._count
        self._datasets["sys_time_ns"][index] = int(sys_time_ns)
        self._datasets["policy_action"][index] = _dataset_value(tuple(policy_action))
        self._datasets["sent_action"][index] = _dataset_value(tuple(sent_action))
        self._datasets["teleop_active"][index] = bool(teleop_active)
        self._count = index + 1

    def append_batch(
        self,
        sys_time_ns: Sequence[int],
        policy_actions: Sequence[Sequence[float]],
        sent_actions: Sequence[Sequence[float]],
        teleop_active: Sequence[bool],
    ) -> None:
        count = len(sys_time_ns)
        if count == 0:
            return
        if self._datasets is None:
            self._datasets = {
                "sys_time_ns": self._group.create_dataset(
                    "sys_time_ns",
                    shape=(_ACTION_CHUNK,),
                    maxshape=(None,),
                    dtype=np.int64,
                    chunks=(_ACTION_CHUNK,),
                ),
                "policy_action": _create_dataset(self._group, "policy_action", tuple(policy_actions[0]), _ACTION_CHUNK),
                "sent_action": _create_dataset(self._group, "sent_action", tuple(sent_actions[0]), _ACTION_CHUNK),
                "teleop_active": self._group.create_dataset(
                    "teleop_active",
                    shape=(_ACTION_CHUNK,),
                    maxshape=(None,),
                    dtype=np.bool_,
                    chunks=(_ACTION_CHUNK,),
                ),
            }
        start = self._count
        stop = start + count
        self._ensure_capacity(stop)
        self._datasets["sys_time_ns"][start:stop] = np.asarray(sys_time_ns, dtype=np.int64)
        self._datasets["policy_action"][start:stop] = np.asarray(policy_actions)
        self._datasets["sent_action"][start:stop] = np.asarray(sent_actions)
        self._datasets["teleop_active"][start:stop] = np.asarray(teleop_active, dtype=np.bool_)
        self._count = stop

    def close(self) -> None:
        if self._datasets is None:
            return
        for dataset in self._datasets.values():
            dataset.resize(self._count, axis=0)

    def _grow(self) -> None:
        self._ensure_capacity(self._capacity + _ACTION_CHUNK)

    def _ensure_capacity(self, needed: int) -> None:
        if needed <= self._capacity:
            return
        while needed > self._capacity:
            self._capacity += _ACTION_CHUNK
        if self._datasets is None:
            return
        for dataset in self._datasets.values():
            dataset.resize(self._capacity, axis=0)


class _CameraTimestampWriter:
    def __init__(self, h5_file: h5py.File, camera_name: str) -> None:
        self._group = h5_file.create_group(f"camera/{camera_name}", track_order=True)
        self._group.attrs["field_order"] = np.asarray(("sys_time_ns", "frame_idx"), dtype=_STRING_DTYPE)
        self._datasets = {
            "sys_time_ns": self._group.create_dataset(
                "sys_time_ns",
                shape=(_CAMERA_CHUNK,),
                maxshape=(None,),
                dtype=np.int64,
                chunks=(_CAMERA_CHUNK,),
            ),
            "frame_idx": self._group.create_dataset(
                "frame_idx",
                shape=(_CAMERA_CHUNK,),
                maxshape=(None,),
                dtype=np.int64,
                chunks=(_CAMERA_CHUNK,),
            ),
        }
        self._count = 0
        self._capacity = _CAMERA_CHUNK

    def append(self, sys_time_ns: int, frame_idx: int) -> None:
        self._ensure_capacity(self._count + 1)
        index = self._count
        self._datasets["sys_time_ns"][index] = int(sys_time_ns)
        self._datasets["frame_idx"][index] = int(frame_idx)
        self._count = index + 1

    def append_batch(self, sys_time_ns: Sequence[int], frame_idx: Sequence[int]) -> None:
        count = len(sys_time_ns)
        if count == 0:
            return
        start = self._count
        stop = start + count
        self._ensure_capacity(stop)
        self._datasets["sys_time_ns"][start:stop] = np.asarray(sys_time_ns, dtype=np.int64)
        self._datasets["frame_idx"][start:stop] = np.asarray(frame_idx, dtype=np.int64)
        self._count = stop

    def close(self) -> None:
        for dataset in self._datasets.values():
            dataset.resize(self._count, axis=0)

    def _grow(self) -> None:
        self._ensure_capacity(self._capacity + _CAMERA_CHUNK)

    def _ensure_capacity(self, needed: int) -> None:
        if needed <= self._capacity:
            return
        while needed > self._capacity:
            self._capacity += _CAMERA_CHUNK
        for dataset in self._datasets.values():
            dataset.resize(self._capacity, axis=0)


class _TactileMetadataWriter:
    def __init__(self, h5_file: h5py.File) -> None:
        self._h5_file = h5_file
        self._writers: dict[tuple[str, str], _TactileFingerMetadataWriter] = {}

    def append(self, frame: SharpaTactileFrame, raw_frame_idx: int, deform_frame_idx: int) -> None:
        key = (frame.side, frame.finger)
        if key not in self._writers:
            self._writers[key] = _TactileFingerMetadataWriter(self._h5_file, frame.side, frame.finger)
        self._writers[key].append(frame, raw_frame_idx, deform_frame_idx)

    def append_batch(
        self,
        frames: Sequence[SharpaTactileFrame],
        raw_frame_idx: Sequence[int],
        deform_frame_idx: Sequence[int],
    ) -> None:
        if not frames:
            return
        grouped: dict[tuple[str, str], list[tuple[SharpaTactileFrame, int, int]]] = {}
        for frame, raw_idx, deform_idx in zip(frames, raw_frame_idx, deform_frame_idx, strict=True):
            key = (frame.side, frame.finger)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append((frame, int(raw_idx), int(deform_idx)))
        for (side, finger), values in grouped.items():
            key = (side, finger)
            if key not in self._writers:
                self._writers[key] = _TactileFingerMetadataWriter(self._h5_file, side, finger)
            self._writers[key].append_batch(
                frames=[value[0] for value in values],
                raw_frame_idx=[value[1] for value in values],
                deform_frame_idx=[value[2] for value in values],
            )

    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()


class _TactileFingerMetadataWriter:
    def __init__(self, h5_file: h5py.File, side: str, finger: str) -> None:
        root = h5_file.require_group("tactile")
        side_group = root.require_group(side)
        self._group = side_group.create_group(finger, track_order=True)
        self._group.attrs["field_order"] = np.asarray(
            (
                "sys_time_ns",
                "sdk_timestamp_s",
                "frame_id",
                "channel",
                "raw_frame_idx",
                "deform_frame_idx",
                "raw_shape",
                "deform_shape",
                "f6",
                "f6_shape",
                "contact_point",
                "contact_point_shape",
            ),
            dtype=_STRING_DTYPE,
        )
        self._datasets = {
            "sys_time_ns": self._group.create_dataset(
                "sys_time_ns",
                shape=(_TACTILE_CHUNK,),
                maxshape=(None,),
                dtype=np.int64,
                chunks=(_TACTILE_CHUNK,),
            ),
            "sdk_timestamp_s": self._group.create_dataset(
                "sdk_timestamp_s",
                shape=(_TACTILE_CHUNK,),
                maxshape=(None,),
                dtype=np.float64,
                chunks=(_TACTILE_CHUNK,),
            ),
            "frame_id": self._group.create_dataset(
                "frame_id",
                shape=(_TACTILE_CHUNK,),
                maxshape=(None,),
                dtype=np.int64,
                chunks=(_TACTILE_CHUNK,),
            ),
            "channel": self._group.create_dataset(
                "channel",
                shape=(_TACTILE_CHUNK,),
                maxshape=(None,),
                dtype=np.int64,
                chunks=(_TACTILE_CHUNK,),
            ),
            "raw_frame_idx": self._group.create_dataset(
                "raw_frame_idx",
                shape=(_TACTILE_CHUNK,),
                maxshape=(None,),
                dtype=np.int64,
                chunks=(_TACTILE_CHUNK,),
            ),
            "deform_frame_idx": self._group.create_dataset(
                "deform_frame_idx",
                shape=(_TACTILE_CHUNK,),
                maxshape=(None,),
                dtype=np.int64,
                chunks=(_TACTILE_CHUNK,),
            ),
            "raw_shape": self._group.create_dataset(
                "raw_shape",
                shape=(_TACTILE_CHUNK,),
                maxshape=(None,),
                dtype=_VLEN_INT64_DTYPE,
                chunks=(_TACTILE_CHUNK,),
            ),
            "deform_shape": self._group.create_dataset(
                "deform_shape",
                shape=(_TACTILE_CHUNK,),
                maxshape=(None,),
                dtype=_VLEN_INT64_DTYPE,
                chunks=(_TACTILE_CHUNK,),
            ),
            "f6": self._group.create_dataset(
                "f6",
                shape=(_TACTILE_CHUNK,),
                maxshape=(None,),
                dtype=_VLEN_FLOAT32_DTYPE,
                chunks=(_TACTILE_CHUNK,),
            ),
            "f6_shape": self._group.create_dataset(
                "f6_shape",
                shape=(_TACTILE_CHUNK,),
                maxshape=(None,),
                dtype=_VLEN_INT64_DTYPE,
                chunks=(_TACTILE_CHUNK,),
            ),
            "contact_point": self._group.create_dataset(
                "contact_point",
                shape=(_TACTILE_CHUNK,),
                maxshape=(None,),
                dtype=_VLEN_FLOAT32_DTYPE,
                chunks=(_TACTILE_CHUNK,),
            ),
            "contact_point_shape": self._group.create_dataset(
                "contact_point_shape",
                shape=(_TACTILE_CHUNK,),
                maxshape=(None,),
                dtype=_VLEN_INT64_DTYPE,
                chunks=(_TACTILE_CHUNK,),
            ),
        }
        self._count = 0
        self._capacity = _TACTILE_CHUNK

    def append(self, frame: SharpaTactileFrame, raw_frame_idx: int, deform_frame_idx: int) -> None:
        self._ensure_capacity(self._count + 1)
        index = self._count
        self._datasets["sys_time_ns"][index] = int(frame.timestamp_ns)
        self._datasets["sdk_timestamp_s"][index] = float(frame.sdk_timestamp_s)
        self._datasets["frame_id"][index] = int(frame.frame_id)
        self._datasets["channel"][index] = int(frame.channel)
        self._datasets["raw_frame_idx"][index] = int(raw_frame_idx)
        self._datasets["deform_frame_idx"][index] = int(deform_frame_idx)
        self._datasets["raw_shape"][index] = _image_shape(frame.raw_image)
        self._datasets["deform_shape"][index] = _image_shape(frame.deform_image)
        self._datasets["f6"][index] = np.asarray(frame.f6, dtype=np.float32)
        self._datasets["f6_shape"][index] = np.asarray(frame.f6_shape, dtype=np.int64)
        self._datasets["contact_point"][index] = np.asarray(frame.contact_point, dtype=np.float32)
        self._datasets["contact_point_shape"][index] = np.asarray(frame.contact_point_shape, dtype=np.int64)
        self._count = index + 1

    def append_batch(
        self,
        frames: Sequence[SharpaTactileFrame],
        raw_frame_idx: Sequence[int],
        deform_frame_idx: Sequence[int],
    ) -> None:
        count = len(frames)
        if count == 0:
            return
        start = self._count
        stop = start + count
        self._ensure_capacity(stop)
        self._datasets["sys_time_ns"][start:stop] = np.asarray([int(frame.timestamp_ns) for frame in frames], dtype=np.int64)
        self._datasets["sdk_timestamp_s"][start:stop] = np.asarray([float(frame.sdk_timestamp_s) for frame in frames], dtype=np.float64)
        self._datasets["frame_id"][start:stop] = np.asarray([int(frame.frame_id) for frame in frames], dtype=np.int64)
        self._datasets["channel"][start:stop] = np.asarray([int(frame.channel) for frame in frames], dtype=np.int64)
        self._datasets["raw_frame_idx"][start:stop] = np.asarray(raw_frame_idx, dtype=np.int64)
        self._datasets["deform_frame_idx"][start:stop] = np.asarray(deform_frame_idx, dtype=np.int64)
        self._datasets["raw_shape"][start:stop] = [_image_shape(frame.raw_image) for frame in frames]
        self._datasets["deform_shape"][start:stop] = [_image_shape(frame.deform_image) for frame in frames]
        self._datasets["f6"][start:stop] = [np.asarray(frame.f6, dtype=np.float32) for frame in frames]
        self._datasets["f6_shape"][start:stop] = [np.asarray(frame.f6_shape, dtype=np.int64) for frame in frames]
        self._datasets["contact_point"][start:stop] = [np.asarray(frame.contact_point, dtype=np.float32) for frame in frames]
        self._datasets["contact_point_shape"][start:stop] = [
            np.asarray(frame.contact_point_shape, dtype=np.int64) for frame in frames
        ]
        self._count = stop

    def close(self) -> None:
        for dataset in self._datasets.values():
            dataset.resize(self._count, axis=0)

    def _grow(self) -> None:
        self._ensure_capacity(self._capacity + _TACTILE_CHUNK)

    def _ensure_capacity(self, needed: int) -> None:
        if needed <= self._capacity:
            return
        while needed > self._capacity:
            self._capacity += _TACTILE_CHUNK
        for dataset in self._datasets.values():
            dataset.resize(self._capacity, axis=0)


class TactileVideoEpisodeWriter:
    def __init__(self, path: str | Path, fps: float, modalities: Sequence[str]) -> None:
        self.path = Path(path)
        self.fps = float(fps)
        self.modalities = tuple(str(modality) for modality in modalities)
        self._writers: dict[tuple[str, str, str], _FFV1GrayVideoWriter] = {}

    def write_frame(self, side: str, finger: str, modality: str, image: np.ndarray) -> int:
        if modality not in self.modalities:
            raise RuntimeError(f"unsupported tactile video modality: {modality}")
        frame = _gray_uint8_frame(image, "tactile")
        key = (side, finger, modality)
        if key not in self._writers:
            self._writers[key] = _FFV1GrayVideoWriter(
                path=self.path / f"{side}_{finger}_{modality}.mkv",
                fps=self.fps,
                width=frame.shape[1],
                height=frame.shape[0],
            )
        return self._writers[key].write_frame(frame)

    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()


class _FFV1GrayVideoWriter:
    def __init__(self, path: str | Path, fps: float, width: int, height: int) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width = int(width)
        self.height = int(height)
        fourcc = cv2.VideoWriter_fourcc(*"FFV1")
        self._writer = cv2.VideoWriter(str(self.path), fourcc, float(fps), (self.width, self.height), False)
        if not self._writer.isOpened():
            self._writer.release()
            raise RuntimeError(f"failed to open FFV1/MKV writer: {self.path}")
        self._frame_count = 0
        logger.info("recording tactile FFV1/MKV opened: %s size=%sx%s fps=%s", self.path, width, height, fps)

    def write_frame(self, frame: np.ndarray) -> int:
        frame = _gray_uint8_frame(frame, "tactile")
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            raise RuntimeError(
                f"tactile frame shape {frame.shape} does not match FFV1/MKV size "
                f"height={self.height} width={self.width}"
            )
        frame_idx = self._frame_count
        self._writer.write(frame)
        self._frame_count = frame_idx + 1
        return frame_idx

    def close(self) -> None:
        self._writer.release()
        logger.info("recording tactile FFV1/MKV closed: %s frames=%s", self.path, self._frame_count)


class MP4EpisodeWriter:
    def __init__(self, path: str | Path, fps: int, width: int, height: int, codec: str) -> None:
        self.path = Path(path)
        self.width = int(width)
        self.height = int(height)
        fourcc = cv2.VideoWriter_fourcc(*str(codec))
        self._writer = cv2.VideoWriter(str(self.path), fourcc, int(fps), (self.width, self.height))
        if not self._writer.isOpened():
            self._writer.release()
            raise RuntimeError(f"failed to open MP4 writer: {self.path}")
        self._frame_count = 0
        logger.info("recording MP4 opened: %s size=%sx%s fps=%s codec=%s", self.path, width, height, fps, codec)

    def write_frame(self, frame_rgb: np.ndarray) -> None:
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise RuntimeError(f"camera frame must be RGB HxWx3, got shape={frame_rgb.shape}")
        if frame_rgb.shape[1] != self.width or frame_rgb.shape[0] != self.height:
            raise RuntimeError(
                f"camera frame shape {frame_rgb.shape} does not match MP4 size "
                f"height={self.height} width={self.width}"
            )
        frame_bgr = frame_rgb[..., ::-1]
        self._writer.write(frame_bgr)
        self._frame_count += 1

    def close(self) -> None:
        self._writer.release()
        logger.info("recording MP4 closed: %s frames=%s", self.path, self._frame_count)


def _create_dataset(group: h5py.Group, name: str, value: Any, chunk_size: int) -> h5py.Dataset:
    dataset_value = _dataset_value(value)
    if isinstance(dataset_value, str):
        return group.create_dataset(
            name,
            shape=(chunk_size,),
            maxshape=(None,),
            dtype=_STRING_DTYPE,
            chunks=(chunk_size,),
        )
    array = np.asarray(dataset_value)
    if array.ndim == 0:
        return group.create_dataset(
            name,
            shape=(chunk_size,),
            maxshape=(None,),
            dtype=array.dtype,
            chunks=(chunk_size,),
        )
    return group.create_dataset(
        name,
        shape=(chunk_size, *array.shape),
        maxshape=(None, *array.shape),
        dtype=array.dtype,
        chunks=(chunk_size, *array.shape),
    )


def _dataset_value(value: Any) -> Any:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return value
    if isinstance(value, tuple):
        if not value:
            return "[]"
        if all(isinstance(item, str) for item in value):
            return json.dumps(list(value), ensure_ascii=True)
        return np.asarray(value)
    if isinstance(value, list):
        if not value:
            return "[]"
        if all(isinstance(item, str) for item in value):
            return json.dumps(value, ensure_ascii=True)
        return np.asarray(value)
    if isinstance(value, Mapping):
        return json.dumps(_jsonable_recording_value(value), sort_keys=True, ensure_ascii=True)
    raise TypeError(f"unsupported recording field value type: {type(value).__name__}")


def _image_shape(image: np.ndarray | None) -> np.ndarray:
    if image is None:
        return np.asarray((), dtype=np.int64)
    return np.asarray(image.shape, dtype=np.int64)


def _gray_uint8_frame(frame: np.ndarray, name: str) -> np.ndarray:
    if frame.ndim != 2:
        raise RuntimeError(f"{name} frame must be grayscale HxW, got shape={frame.shape}")
    if frame.dtype != np.uint8:
        raise RuntimeError(f"{name} frame must be uint8, got dtype={frame.dtype}")
    return np.ascontiguousarray(frame)


def _jsonable_recording_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable_recording_value(child) for key, child in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable_recording_value(child) for child in value]
    return value
