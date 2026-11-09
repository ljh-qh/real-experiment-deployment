from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import queue
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from unirobot_infra.camera import CameraManager
from unirobot_infra.core.time import now_ns
from unirobot_infra.core.types import SharpaHandState, SharpaTactileFrame, TianjiArmState
from unirobot_infra.recorder.writers import HDF5EpisodeWriter, MP4EpisodeWriter, TactileVideoEpisodeWriter
from unirobot_infra.robots.compositional.group import RobotGroup
from unirobot_infra.utils.logging import get_logger


logger = get_logger(__name__)

_SENTINEL = object()
_ALL_STATES_TOKEN = "*"


@dataclass(frozen=True)
class _StreamSample:
    stream_name: str
    sys_time_ns: int
    fields: Mapping[str, Any]


@dataclass(frozen=True)
class _ActionSample:
    sys_time_ns: int
    policy_action: tuple[float, ...]
    sent_action: tuple[float, ...]
    teleop_active: bool


@dataclass(frozen=True)
class _CameraFrameSample:
    camera_name: str
    sys_time_ns: int
    frame_idx: int
    image: np.ndarray


@dataclass(frozen=True)
class _TactileFrameSample:
    frame: SharpaTactileFrame


@dataclass
class _StreamStats:
    count: int = 0


@dataclass
class _CameraStats:
    count: int = 0


@dataclass
class _TactileStats:
    count: int = 0


@dataclass
class _EpisodeRuntime:
    episode_dir: Path
    queue: queue.Queue
    stop_event: threading.Event
    writer: HDF5EpisodeWriter
    mp4_writers: dict[str, MP4EpisodeWriter]
    tactile_video_writer: TactileVideoEpisodeWriter | None
    stats_by_stream: dict[str, _StreamStats]
    stats_by_camera: dict[str, _CameraStats]
    stats_by_tactile: dict[str, _TactileStats]
    action_count: int = 0
    start_monotonic: float = 0.0
    writer_thread: threading.Thread | None = None
    sampler_threads: tuple[threading.Thread, ...] = ()
    camera_threads: tuple[threading.Thread, ...] = ()
    tactile_threads: tuple[threading.Thread, ...] = ()


@dataclass(frozen=True)
class _StreamSampler:
    name: str
    side: str
    role: str
    target_hz: float
    states: tuple[str, ...]
    read_state: Callable[[], TianjiArmState | SharpaHandState]


@dataclass(frozen=True)
class _TactileSampler:
    name: str
    side: str
    target_hz: float
    timeout_s: float
    channels: tuple[int, ...]
    read_tactile_frame: Callable[[int, float], SharpaTactileFrame | None]


class DataRecorder:
    def __init__(
        self,
        cfg: DictConfig,
        robot_group: RobotGroup,
        cameras: DictConfig | None = None,
        camera_manager: CameraManager | None = None,
        close_camera_manager: bool = True,
        sample_camera_threads: bool = True,
    ) -> None:
        self.cfg = cfg
        self.robot_group = robot_group
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._samplers = tuple(_build_stream_sampler(robot_group, stream_cfg) for stream_cfg in cfg.streams)
        if not self._samplers:
            raise ValueError("recording.streams must not be empty")
        self._tactile_cfg = cfg.tactile
        self._tactile_modalities = _tactile_modalities(self._tactile_cfg)
        self._tactile_samplers = _build_tactile_samplers(robot_group, self._tactile_cfg)
        self._episode_idx = self._next_episode_idx()
        self._recording = False
        self._error: BaseException | None = None
        self._camera_cfg = _select_camera_config(cfg, cameras)
        self._camera_manager = camera_manager if camera_manager is not None else _build_camera_manager(self._camera_cfg)
        self._close_camera_manager = bool(close_camera_manager)
        self._sample_camera_threads = bool(sample_camera_threads)
        self._camera_names = tuple(self._camera_manager.captures.keys()) if self._camera_manager is not None else ()
        self._runtime: _EpisodeRuntime | None = None
        self._finalizer_threads: list[threading.Thread] = []
        self._finalizer_errors: list[BaseException] = []
        self._finalizer_lock = threading.Lock()
        self._latest_camera_frames: dict[str, np.ndarray] = {}
        self._latest_camera_lock = threading.Lock()

    def start_episode(self) -> Path:
        if self._recording:
            raise RuntimeError("DataRecorder is already recording")
        self._raise_thread_error()

        episode_dir = self.output_dir / f"episode_{self._episode_idx:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        runtime = _EpisodeRuntime(
            episode_dir=episode_dir,
            queue=queue.Queue(),
            stop_event=threading.Event(),
            writer=HDF5EpisodeWriter(episode_dir / "robot_states.h5", camera_names=self._camera_names),
            mp4_writers=_build_mp4_writers(episode_dir, self.cfg, self._camera_cfg, self._camera_names),
            tactile_video_writer=_build_tactile_video_writer(
                episode_dir=episode_dir,
                tactile_cfg=self._tactile_cfg,
                modalities=self._tactile_modalities,
            ),
            stats_by_stream={sampler.name: _StreamStats() for sampler in self._samplers},
            stats_by_tactile={sampler.name: _TactileStats() for sampler in self._tactile_samplers},
            stats_by_camera={camera_name: _CameraStats() for camera_name in self._camera_names},
            start_monotonic=time.monotonic(),
        )
        self._error = None

        self._recording = True
        self._runtime = runtime
        runtime.writer_thread = threading.Thread(target=self._write_loop, args=(runtime,), name="recording-writer", daemon=True)
        runtime.writer_thread.start()
        runtime.sampler_threads = tuple(
            threading.Thread(
                target=self._sample_loop,
                args=(sampler, runtime),
                name=f"recording-{sampler.name}",
                daemon=True,
            )
            for sampler in self._samplers
        )
        for thread in runtime.sampler_threads:
            thread.start()
        runtime.camera_threads = (
            tuple(
                threading.Thread(
                    target=self._camera_loop,
                    args=(camera_name, runtime),
                    name=f"recording-camera-{camera_name}",
                    daemon=True,
                )
                for camera_name in self._camera_names
            )
            if self._sample_camera_threads
            else ()
        )
        for thread in runtime.camera_threads:
            thread.start()
        runtime.tactile_threads = tuple(
            threading.Thread(
                target=self._tactile_loop,
                args=(sampler, runtime),
                name=f"recording-tactile-{sampler.name}",
                daemon=True,
            )
            for sampler in self._tactile_samplers
        )
        for thread in runtime.tactile_threads:
            thread.start()

        logger.info("recording started: %s", episode_dir)
        return episode_dir

    def stop_episode(self) -> None:
        if not self._recording:
            raise RuntimeError("DataRecorder is not recording")
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("DataRecorder runtime is not set")
        self._finish_episode_async(runtime)
        self._episode_idx += 1
        self._runtime = None
        self._recording = False
        self._raise_thread_error()
        logger.info("recording stopped, finalizing in background: %s", runtime.episode_dir)

    def discard_episode(self) -> None:
        if not self._recording:
            raise RuntimeError("DataRecorder is not recording")
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("DataRecorder runtime is not set")
        self._finish_episode_sync(runtime)
        shutil.rmtree(runtime.episode_dir)
        self._runtime = None
        self._recording = False
        logger.info("recording discarded: %s", runtime.episode_dir)

    def record_action(
        self,
        policy_action: tuple[float, ...],
        sent_action: tuple[float, ...],
        teleop_active: bool,
    ) -> None:
        if not self._recording:
            raise RuntimeError("DataRecorder is not recording")
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("DataRecorder runtime is not set")
        runtime.queue.put(
            _ActionSample(
                sys_time_ns=now_ns(),
                policy_action=tuple(float(value) for value in policy_action),
                sent_action=tuple(float(value) for value in sent_action),
                teleop_active=bool(teleop_active),
            )
        )
        runtime.action_count += 1

    def record_camera_frames(self, camera_frames: Mapping[str, np.ndarray]) -> None:
        if not self._recording:
            raise RuntimeError("DataRecorder is not recording")
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("DataRecorder runtime is not set")
        sys_time_ns = now_ns()
        for camera_name in self._camera_names:
            image = np.asarray(camera_frames[camera_name], dtype=np.uint8)
            self._update_latest_camera_frame(camera_name, image)
            frame_idx = runtime.stats_by_camera[camera_name].count
            runtime.queue.put(
                _CameraFrameSample(
                    camera_name=camera_name,
                    sys_time_ns=sys_time_ns,
                    frame_idx=frame_idx,
                    image=image,
                )
            )
            runtime.stats_by_camera[camera_name].count += 1

    @property
    def has_error(self) -> bool:
        return self._error is not None or bool(self._finalizer_errors)

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def records_camera_frames_manually(self) -> bool:
        return bool(self._camera_names) and not self._sample_camera_threads

    @property
    def episode_index(self) -> int:
        return self._episode_idx

    @property
    def error_message(self) -> str:
        if self._error is None:
            if not self._finalizer_errors:
                return ""
            return str(self._finalizer_errors[0])
        return str(self._error)

    def get_stats(self) -> dict[str, Any]:
        runtime = self._runtime
        elapsed_s = time.monotonic() - runtime.start_monotonic if self._recording and runtime is not None else 0.0
        stream_stats = {}
        stats_by_stream = runtime.stats_by_stream if runtime is not None else {sampler.name: _StreamStats() for sampler in self._samplers}
        for name, stats in stats_by_stream.items():
            stream_stats[name] = {
                "count": stats.count,
                "hz": stats.count / elapsed_s if elapsed_s > 0.0 else 0.0,
            }
        camera_stats = {}
        stats_by_camera = runtime.stats_by_camera if runtime is not None else {camera_name: _CameraStats() for camera_name in self._camera_names}
        for name, stats in stats_by_camera.items():
            camera_stats[name] = {
                "count": stats.count,
                "hz": stats.count / elapsed_s if elapsed_s > 0.0 else 0.0,
            }
        tactile_stats = {}
        stats_by_tactile = runtime.stats_by_tactile if runtime is not None else {sampler.name: _TactileStats() for sampler in self._tactile_samplers}
        for name, stats in stats_by_tactile.items():
            tactile_stats[name] = {
                "count": stats.count,
                "hz": stats.count / elapsed_s if elapsed_s > 0.0 else 0.0,
            }
        return {
            "elapsed_s": elapsed_s,
            "streams": stream_stats,
            "cameras": camera_stats,
            "tactile": tactile_stats,
            "action_count": runtime.action_count if runtime is not None else 0,
            "action_hz": (runtime.action_count / elapsed_s) if runtime is not None and elapsed_s > 0.0 else 0.0,
            "queue_size": runtime.queue.qsize() if runtime is not None else 0,
        }

    def latest_camera_frames(self) -> dict[str, np.ndarray]:
        with self._latest_camera_lock:
            return {name: self._latest_camera_frames[name].copy() for name in self._camera_names if name in self._latest_camera_frames}

    def close(self) -> None:
        if self._recording:
            self.stop_episode()
        self.wait_for_pending_saves()
        if self._camera_manager is not None and self._close_camera_manager:
            self._camera_manager.close()

    def _sample_loop(self, sampler: _StreamSampler, runtime: _EpisodeRuntime) -> None:
        try:
            period_s = 1.0 / sampler.target_hz
            next_sample_time = time.monotonic()
            while not runtime.stop_event.is_set():
                state = sampler.read_state()
                if runtime.stop_event.is_set():
                    break
                sys_time_ns = now_ns()
                runtime.queue.put(
                    _StreamSample(
                        stream_name=sampler.name,
                        sys_time_ns=sys_time_ns,
                        fields=_recording_fields(state, sampler.states),
                    )
                )
                runtime.stats_by_stream[sampler.name].count += 1
                next_sample_time += period_s
                remaining_s = next_sample_time - time.monotonic()
                if remaining_s > 0.0:
                    runtime.stop_event.wait(remaining_s)
                else:
                    next_sample_time = time.monotonic() + period_s
                    runtime.stop_event.wait(period_s)
        except BaseException as error:
            self._error = error
            runtime.stop_event.set()

    def _camera_loop(self, camera_name: str, runtime: _EpisodeRuntime) -> None:
        try:
            frame_idx = 0
            read_timeout_s = float(self.cfg.camera_read_timeout_s)
            capture = self._camera_manager.captures[camera_name]
            while not runtime.stop_event.is_set():
                image = capture.read(timeout_s=read_timeout_s)
                if runtime.stop_event.is_set():
                    break
                self._update_latest_camera_frame(camera_name, image)
                sys_time_ns = now_ns()
                runtime.queue.put(
                    _CameraFrameSample(
                        camera_name=camera_name,
                        sys_time_ns=sys_time_ns,
                        frame_idx=frame_idx,
                        image=image,
                    )
                )
                runtime.stats_by_camera[camera_name].count += 1
                frame_idx += 1
        except BaseException as error:
            self._error = error
            runtime.stop_event.set()

    def _update_latest_camera_frame(self, camera_name: str, image: np.ndarray) -> None:
        with self._latest_camera_lock:
            self._latest_camera_frames[camera_name] = image.copy()

    def _tactile_loop(self, sampler: _TactileSampler, runtime: _EpisodeRuntime) -> None:
        try:
            period_s = 1.0 / sampler.target_hz
            next_sample_time = time.monotonic()
            while not runtime.stop_event.is_set():
                for channel in sampler.channels:
                    frame = sampler.read_tactile_frame(channel, sampler.timeout_s)
                    if runtime.stop_event.is_set():
                        break
                    if frame is None:
                        continue
                    runtime.queue.put(_TactileFrameSample(frame=frame))
                    runtime.stats_by_tactile[sampler.name].count += 1
                next_sample_time += period_s
                remaining_s = next_sample_time - time.monotonic()
                if remaining_s > 0.0:
                    runtime.stop_event.wait(remaining_s)
                else:
                    next_sample_time = time.monotonic() + period_s
                    runtime.stop_event.wait(period_s)
        except BaseException as error:
            self._error = error
            runtime.stop_event.set()

    def _write_loop(self, runtime: _EpisodeRuntime) -> None:
        try:
            while True:
                message = runtime.queue.get()
                if message is _SENTINEL:
                    break
                batch = [message]
                while True:
                    try:
                        message = runtime.queue.get_nowait()
                    except queue.Empty:
                        break
                    if message is _SENTINEL:
                        self._write_batch(runtime, batch)
                        return
                    batch.append(message)
                self._write_batch(runtime, batch)
        except BaseException as error:
            self._error = error
            runtime.stop_event.set()

    def _write_batch(
        self,
        runtime: _EpisodeRuntime,
        batch: list[_StreamSample | _ActionSample | _CameraFrameSample | _TactileFrameSample | object],
    ) -> None:
        stream_samples: dict[str, list[_StreamSample]] = {}
        action_samples = []
        camera_samples: dict[str, list[_CameraFrameSample]] = {}
        tactile_samples: list[_TactileFrameSample] = []
        for message in batch:
            if isinstance(message, _StreamSample):
                if message.stream_name not in stream_samples:
                    stream_samples[message.stream_name] = []
                stream_samples[message.stream_name].append(message)
            elif isinstance(message, _ActionSample):
                action_samples.append(message)
            elif isinstance(message, _CameraFrameSample):
                if message.camera_name not in camera_samples:
                    camera_samples[message.camera_name] = []
                camera_samples[message.camera_name].append(message)
            elif isinstance(message, _TactileFrameSample):
                tactile_samples.append(message)
            else:
                raise TypeError("unsupported recording message")

        for stream_name, samples in stream_samples.items():
            runtime.writer.append_stream_samples(
                stream_name=stream_name,
                sys_time_ns=[sample.sys_time_ns for sample in samples],
                fields=[sample.fields for sample in samples],
            )
        if action_samples:
            runtime.writer.append_action_samples(
                sys_time_ns=[sample.sys_time_ns for sample in action_samples],
                policy_actions=[sample.policy_action for sample in action_samples],
                sent_actions=[sample.sent_action for sample in action_samples],
                teleop_active=[sample.teleop_active for sample in action_samples],
            )
        for camera_name, samples in camera_samples.items():
            for sample in samples:
                runtime.mp4_writers[sample.camera_name].write_frame(sample.image)
            runtime.writer.append_camera_frame_samples(
                camera_name=camera_name,
                sys_time_ns=[sample.sys_time_ns for sample in samples],
                frame_idx=[sample.frame_idx for sample in samples],
            )
        if tactile_samples:
            self._write_tactile_batch(runtime, tactile_samples)

    def _write_tactile_batch(self, runtime: _EpisodeRuntime, samples: list[_TactileFrameSample]) -> None:
        if runtime.tactile_video_writer is None:
            raise RuntimeError("tactile video writer is not enabled")
        metadata_by_finger: dict[tuple[str, str], list[tuple[SharpaTactileFrame, int, int]]] = {}
        for sample in samples:
            raw_frame_idx = -1
            deform_frame_idx = -1
            frame = sample.frame
            if "raw" in self._tactile_modalities and frame.raw_image is not None:
                raw_frame_idx = runtime.tactile_video_writer.write_frame(
                    side=frame.side,
                    finger=frame.finger,
                    modality="raw",
                    image=frame.raw_image,
                )
            if "deform" in self._tactile_modalities and frame.deform_image is not None:
                deform_frame_idx = runtime.tactile_video_writer.write_frame(
                    side=frame.side,
                    finger=frame.finger,
                    modality="deform",
                    image=frame.deform_image,
                )
            key = (frame.side, frame.finger)
            if key not in metadata_by_finger:
                metadata_by_finger[key] = []
            metadata_by_finger[key].append((frame, raw_frame_idx, deform_frame_idx))

        for values in metadata_by_finger.values():
            runtime.writer.append_tactile_frame_samples(
                frames=[value[0] for value in values],
                raw_frame_idx=[value[1] for value in values],
                deform_frame_idx=[value[2] for value in values],
            )

    def _raise_thread_error(self) -> None:
        if self._error is not None:
            raise RuntimeError(str(self._error))
        if self._finalizer_errors:
            raise RuntimeError(str(self._finalizer_errors[0]))

    def _stop_capture_threads(self, runtime: _EpisodeRuntime) -> None:
        runtime.stop_event.set()
        for thread in runtime.sampler_threads:
            thread.join()
        for thread in runtime.camera_threads:
            thread.join()
        for thread in runtime.tactile_threads:
            thread.join()

    def _finish_episode_sync(self, runtime: _EpisodeRuntime) -> None:
        self._stop_capture_threads(runtime)
        runtime.queue.put(_SENTINEL)
        self._finalize_episode(runtime)
        self._raise_thread_error()

    def _finish_episode_async(self, runtime: _EpisodeRuntime) -> None:
        self._stop_capture_threads(runtime)
        runtime.queue.put(_SENTINEL)
        try:
            self._wait_for_finalizer_slot()
        except BaseException:
            self._finalize_episode(runtime)
            raise
        thread = threading.Thread(
            target=self._finalize_episode,
            args=(runtime,),
            name=f"recording-finalizer-{runtime.episode_dir.name}",
            daemon=True,
        )
        with self._finalizer_lock:
            self._finalizer_threads.append(thread)
        thread.start()

    def _finalize_episode(self, runtime: _EpisodeRuntime) -> None:
        try:
            if runtime.writer_thread is not None:
                runtime.writer_thread.join()
            runtime.writer.close()
            for writer in runtime.mp4_writers.values():
                writer.close()
            if runtime.tactile_video_writer is not None:
                runtime.tactile_video_writer.close()
            logger.info("recording finalized: %s", runtime.episode_dir)
        except BaseException as error:
            self._error = error
            with self._finalizer_lock:
                self._finalizer_errors.append(error)

    def wait_for_pending_saves(self) -> None:
        while True:
            with self._finalizer_lock:
                threads = list(self._finalizer_threads)
                self._finalizer_threads = [thread for thread in self._finalizer_threads if thread.is_alive()]
            if not threads:
                break
            for thread in threads:
                thread.join()
        self._raise_thread_error()

    def _wait_for_finalizer_slot(self) -> None:
        while True:
            with self._finalizer_lock:
                alive_threads = [thread for thread in self._finalizer_threads if thread.is_alive()]
                self._finalizer_threads = alive_threads
            if len(alive_threads) == 0:
                return
            alive_threads[0].join()
            self._raise_thread_error()

    def _next_episode_idx(self) -> int:
        episode_indices = [
            int(path.name.split("_")[1])
            for path in self.output_dir.iterdir()
            if path.is_dir() and path.name.startswith("episode_")
        ]
        return max(episode_indices, default=0) + 1


def _build_stream_sampler(robot_group: RobotGroup, stream_cfg: DictConfig) -> _StreamSampler:
    if float(stream_cfg.target_hz) <= 0.0:
        raise ValueError("recording stream target_hz must be positive")
    unit_by_side = {unit.side: unit for unit in robot_group.units}
    unit = unit_by_side[stream_cfg.side]
    if stream_cfg.role == "arm":
        if unit.arm is None:
            raise ValueError(f"recording stream {stream_cfg.name} requires {stream_cfg.side} arm")
        read_state = unit.arm.read_state
    elif stream_cfg.role == "end_effector":
        if unit.end_effector is None:
            raise ValueError(f"recording stream {stream_cfg.name} requires {stream_cfg.side} end_effector")
        read_state = unit.end_effector.read_state
    else:
        raise ValueError("recording stream role must be arm or end_effector")
    states = tuple(str(state_name) for state_name in stream_cfg.states)
    if not states:
        raise ValueError("recording stream states must not be empty")
    return _StreamSampler(
        name=str(stream_cfg.name),
        side=str(stream_cfg.side),
        role=str(stream_cfg.role),
        target_hz=float(stream_cfg.target_hz),
        states=states,
        read_state=read_state,
    )


def _build_tactile_samplers(robot_group: RobotGroup, tactile_cfg: DictConfig) -> tuple[_TactileSampler, ...]:
    if not bool(tactile_cfg.enabled):
        return ()
    target_hz = float(tactile_cfg.target_hz)
    if target_hz <= 0.0:
        raise ValueError("recording.tactile.target_hz must be positive")
    timeout_s = float(tactile_cfg.timeout_s)
    if timeout_s < 0.0:
        raise ValueError("recording.tactile.timeout_s must be non-negative")
    _validate_tactile_modalities(tuple(str(modality) for modality in tactile_cfg.modalities))
    unit_by_side = {unit.side: unit for unit in robot_group.units}
    samplers = []
    for hand_cfg in tactile_cfg.hands:
        side = str(hand_cfg.side)
        unit = unit_by_side[side]
        if unit.end_effector is None:
            raise ValueError(f"recording tactile requires {side} end_effector")
        channels = tuple(int(channel) for channel in hand_cfg.channels)
        if not channels:
            raise ValueError("recording.tactile hand channels must not be empty")
        _validate_tactile_channels(side, channels)
        samplers.append(
            _TactileSampler(
                name=f"{side}_tactile",
                side=side,
                target_hz=target_hz,
                timeout_s=timeout_s,
                channels=channels,
                read_tactile_frame=unit.end_effector.read_tactile_frame,
            )
        )
    if not samplers:
        raise ValueError("recording.tactile.hands must not be empty when tactile is enabled")
    return tuple(samplers)


def _recording_fields(state: TianjiArmState | SharpaHandState, state_names: tuple[str, ...]) -> dict[str, Any]:
    state_dict = state.to_dict()
    if state_names == (_ALL_STATES_TOKEN,):
        return _flatten_recording_fields("", state_dict)
    return {state_name: _recording_field_value(state_dict, state_name) for state_name in state_names}


def _recording_field_value(state_dict: Mapping[str, Any], state_name: str) -> Any:
    value: Any = state_dict
    for part in state_name.split("."):
        value = value[part]
    return _recording_leaf_value(state_name, value)


def _flatten_recording_fields(prefix: str, value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        fields: dict[str, Any] = {}
        for key, child in value.items():
            child_name = f"{prefix}.{key}" if prefix else str(key)
            fields.update(_flatten_recording_fields(child_name, child))
        return fields
    return {prefix: _recording_leaf_value(prefix, value)}


def _recording_leaf_value(field_name: str, value: Any) -> Any:
    if value is None:
        raise RuntimeError(f"recording field {field_name} is None")
    if isinstance(value, Mapping):
        return json.dumps(_jsonable_recording_value(value), sort_keys=True, ensure_ascii=True)
    return value


def _jsonable_recording_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable_recording_value(child) for key, child in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable_recording_value(child) for child in value]
    return value


def _select_camera_config(cfg: DictConfig, cameras: DictConfig | None) -> DictConfig:
    camera_names = tuple(str(name) for name in cfg.camera_names)
    if not camera_names:
        return OmegaConf.create({})
    if cameras is None:
        raise ValueError("recording.camera_names requires camera config")
    return OmegaConf.create({camera_name: cameras[camera_name] for camera_name in camera_names})


def _build_camera_manager(camera_cfg: DictConfig) -> CameraManager | None:
    if not camera_cfg:
        return None
    return CameraManager(camera_cfg)


def _build_mp4_writers(
    episode_dir: Path,
    cfg: DictConfig,
    cameras: DictConfig,
    camera_names: tuple[str, ...],
) -> dict[str, MP4EpisodeWriter]:
    writers: dict[str, MP4EpisodeWriter] = {}
    for camera_name in camera_names:
        camera_cfg = cameras[camera_name]
        width, height = _camera_video_size(camera_cfg)
        writers[camera_name] = MP4EpisodeWriter(
            path=episode_dir / f"{camera_name}.mp4",
            fps=int(camera_cfg.fps),
            width=width,
            height=height,
            codec=str(cfg.video_codec),
        )
    return writers


def _build_tactile_video_writer(
    episode_dir: Path,
    tactile_cfg: DictConfig,
    modalities: tuple[str, ...],
) -> TactileVideoEpisodeWriter | None:
    if not bool(tactile_cfg.enabled):
        return None
    return TactileVideoEpisodeWriter(
        path=episode_dir / "tactile",
        fps=float(tactile_cfg.target_hz),
        modalities=modalities,
    )


def _camera_video_size(camera_cfg: DictConfig) -> tuple[int, int]:
    if camera_cfg.type not in ("insta360_v4l2", "zed_sdk"):
        raise ValueError(f"unsupported recording camera type: {camera_cfg.type}")
    values = tuple(int(value) for value in camera_cfg.image_size)
    if len(values) != 2:
        raise ValueError("camera image_size must have length 2")
    return values[0], values[1]


def _tactile_modalities(tactile_cfg: DictConfig) -> tuple[str, ...]:
    if not bool(tactile_cfg.enabled):
        return ()
    modalities = tuple(str(modality) for modality in tactile_cfg.modalities)
    _validate_tactile_modalities(modalities)
    return modalities


def _validate_tactile_modalities(modalities: tuple[str, ...]) -> None:
    if not modalities:
        raise ValueError("recording.tactile.modalities must not be empty")
    allowed = {"raw", "deform"}
    for modality in modalities:
        if modality not in allowed:
            raise ValueError("recording.tactile.modalities only supports raw and deform")


def _validate_tactile_channels(side: str, channels: tuple[int, ...]) -> None:
    for channel in channels:
        if side == "right" and not 0 <= channel < 5:
            raise ValueError("right Sharpa tactile channels must be in [0, 4]")
        if side == "left" and not 5 <= channel < 10:
            raise ValueError("left Sharpa tactile channels must be in [5, 9]")
        if side not in ("left", "right"):
            raise ValueError("recording.tactile hand side must be left or right")
