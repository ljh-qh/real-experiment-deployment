from dataclasses import dataclass
from pathlib import Path
from threading import Lock
import time
from typing import Any

import hydra
from omegaconf import DictConfig
from rich.console import Console

from unirobot_infra.apps.teleop import (
    _hold_action_from_sent_actions,
    _robot_sides_summary,
    _sleep_remaining,
    _teleop_status_snapshot,
)
from unirobot_infra.envs.builder import build_robot_group
from unirobot_infra.recorder import DataRecorder
from unirobot_infra.utils.logging import get_logger, setup_logging
from unirobot_infra.utils.sdk_output_router import start_sdk_output_router_from_config
from unirobot_infra.utils.status_gui import StatusGui, StatusSnapshot
from unirobot_infra.wrappers.builder import build_env
from unirobot_infra.wrappers.image_display import ImageDisplayWrapper


CONFIG_PATH = str(Path(__file__).resolve().parents[3] / "configs")


@dataclass
class _TimingStats:
    count: int = 0
    total_s: float = 0.0
    max_s: float = 0.0

    def add(self, value_s: float) -> None:
        self.count += 1
        self.total_s += float(value_s)
        self.max_s = max(self.max_s, float(value_s))

    @property
    def avg_ms(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_s * 1000.0 / self.count

    @property
    def max_ms(self) -> float:
        return self.max_s * 1000.0


class _LoopDiagnostics:
    def __init__(self, cfg, logger, recorder: DataRecorder, target_hz: float) -> None:
        self.enabled = bool(cfg.enabled)
        self.log_interval_s = float(cfg.log_interval_s)
        if self.enabled and self.log_interval_s <= 0.0:
            raise ValueError("app.loop_diagnostics.log_interval_s must be positive")
        self.logger = logger
        self.recorder = recorder
        self.target_hz = float(target_hz)
        self.window_start_monotonic = time.monotonic()
        self.window_start_frame = 0
        self.stage_stats: dict[str, _TimingStats] = {}

    def observe(self, frame_index: int, stage_timings: dict[str, float]) -> None:
        if not self.enabled:
            return
        for name, value_s in stage_timings.items():
            if name not in self.stage_stats:
                self.stage_stats[name] = _TimingStats()
            self.stage_stats[name].add(float(value_s))
        if time.monotonic() - self.window_start_monotonic >= self.log_interval_s:
            self._log(frame_index)

    def flush(self, frame_index: int) -> None:
        if not self.enabled or not self.stage_stats:
            return
        self._log(frame_index)

    def _log(self, frame_index: int) -> None:
        elapsed_s = time.monotonic() - self.window_start_monotonic
        frame_delta = frame_index - self.window_start_frame
        actual_hz = frame_delta / elapsed_s if elapsed_s > 0.0 else 0.0
        recorder_stats = self.recorder.get_stats()
        self.logger.info(
            "teleop_record 频率诊断 frame=%s actual_hz=%.2f target_hz=%.2f "
            "action_hz=%.2f queue=%s stages=%s streams_hz=%s cameras_hz=%s tactile_channel_hz=%s",
            frame_index,
            actual_hz,
            self.target_hz,
            float(recorder_stats["action_hz"]),
            int(recorder_stats["queue_size"]),
            _timing_stats_summary(self.stage_stats),
            _rate_stats_summary(recorder_stats["streams"]),
            _rate_stats_summary(recorder_stats["cameras"]),
            _rate_stats_summary(recorder_stats["tactile"]),
        )
        self.window_start_monotonic = time.monotonic()
        self.window_start_frame = frame_index
        self.stage_stats = {}


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="apps/teleop_record/default")
def main(cfg: DictConfig) -> None:
    console = Console(stderr=True)
    setup_logging(cfg.logging.level, console=console)
    logger = get_logger(__name__)
    with start_sdk_output_router_from_config(cfg.app.sdk_output) as sdk_output_router:
        if sdk_output_router is not None:
            logger.info("SDK 输出重定向目录=%s", sdk_output_router.run_dir)
        logger.info(
            "teleop_record 启动 robot=%s sides=%s source=%s",
            cfg.robot.name,
            _robot_sides_summary(cfg),
            cfg.wrappers.teleop_intervention.source,
        )
        _run_teleop_record(cfg, logger, console)


def _run_teleop_record(cfg: DictConfig, logger, console: Console | None = None) -> None:
    robot_group = build_robot_group(cfg)
    env = build_env(cfg, robot_group=robot_group, console=console)
    recorder = None
    key_listener = None
    try:
        recorder = DataRecorder(cfg.recording, robot_group=robot_group, cameras=cfg.camera.cameras)
        if bool(cfg.app.camera_preview.enabled):
            env = ImageDisplayWrapper(
                env,
                cfg.app.camera_preview,
                frame_source=recorder.latest_camera_frames,
                camera_names=tuple(cfg.camera.cameras.keys()),
            )
            logger.info("相机预览已启动 window=%s", cfg.app.camera_preview.window_name)
        key_listener = _NullRecordKeyListener() if cfg.app.record_control.auto_start else _RecordKeyListener()
        _run_teleop_record_loop(cfg, env, recorder, logger, key_listener)
    finally:
        if key_listener is not None:
            key_listener.close()
        if recorder is not None:
            recorder.close()
        env.close()


def _run_teleop_record_loop(cfg: DictConfig, env, recorder: DataRecorder, logger, key_listener) -> None:
    control_keys = _record_control_keys(cfg.app.record_control)
    num_episodes = int(cfg.app.record_control.num_episodes)
    if num_episodes <= 0:
        raise ValueError("app.record_control.num_episodes must be positive")

    episode_done = 0
    _, info = env.reset()
    with StatusGui.from_config(cfg.app.status_gui) as status_gui:
        if cfg.app.status_gui.enabled:
            logger.info("teleop 状态 GUI 已启动")
        while episode_done < num_episodes:
            episode_index = recorder.episode_index
            logger.info(
                "episode %s (%s/%s) ready. Press %s to start, %s to quit.",
                episode_index,
                episode_done + 1,
                num_episodes,
                control_keys.start,
                control_keys.quit,
            )
            status_gui.update(
                _record_status_snapshot(
                    cfg=cfg,
                    frame_index=0,
                    info={"teleop_active": False, "target_actions": ()},
                    record_state="ready",
                    episode_number=episode_done + 1,
                    num_episodes=num_episodes,
                    disk_episode_index=episode_index,
                )
            )
            wait_command = _wait_for_episode_start(cfg, control_keys, key_listener)
            if wait_command == "quit":
                logger.info("teleop_record quit before recording episode %s", episode_index)
                break

            episode_dir = recorder.start_episode()
            logger.info(
                "episode %s recording. Press %s to save, %s to redo, %s to quit.",
                episode_index,
                control_keys.save,
                control_keys.redo,
                control_keys.quit,
            )
            action = env.action_layout.hold_action_from_states(tuple(info["robot_states"]))
            recording_status_info = dict(info)
            recording_status_info["teleop_active"] = False
            recording_status_info["target_actions"] = ()
            status_gui.update(
                _record_status_snapshot(
                    cfg=cfg,
                    frame_index=0,
                    info=recording_status_info,
                    record_state="recording",
                    episode_number=episode_done + 1,
                    num_episodes=num_episodes,
                    disk_episode_index=episode_index,
                )
            )
            episode_command, frame_index, info = _run_recording_episode(
                cfg=cfg,
                env=env,
                recorder=recorder,
                logger=logger,
                key_listener=key_listener,
                control_keys=control_keys,
                status_gui=status_gui,
                action=action,
                episode_number=episode_done + 1,
                num_episodes=num_episodes,
                disk_episode_index=episode_index,
            )
            terminal_status_info = dict(info)
            terminal_status_info["teleop_active"] = False
            terminal_state = _terminal_record_state(episode_command)
            if terminal_state is not None:
                status_gui.update(
                    _record_status_snapshot(
                        cfg=cfg,
                        frame_index=frame_index,
                        info=terminal_status_info,
                        record_state=terminal_state,
                        episode_number=episode_done + 1,
                        num_episodes=num_episodes,
                        disk_episode_index=episode_index,
                    )
                )

            if episode_command == "save":
                recorder.stop_episode()
                logger.info("episode %s save queued: %s frames=%s", episode_index, episode_dir, frame_index)
                episode_done += 1
                if episode_done < num_episodes:
                    _, info = env.reset()
            elif episode_command == "redo":
                recorder.discard_episode()
                logger.info("episode %s discarded, resetting for redo", episode_index)
                _, info = env.reset()
            elif episode_command == "quit":
                recorder.discard_episode()
                logger.info("episode %s discarded, quitting", episode_index)
                break
            elif episode_command == "error":
                error_message = recorder.error_message
                recorder.discard_episode()
                raise RuntimeError(error_message)
            else:
                raise RuntimeError(f"unsupported teleop_record episode command: {episode_command}")


def _run_recording_episode(
    cfg: DictConfig,
    env,
    recorder: DataRecorder,
    logger,
    key_listener,
    control_keys: "_RecordControlKeys",
    status_gui: StatusGui,
    action: tuple[float, ...],
    episode_number: int,
    num_episodes: int,
    disk_episode_index: int,
) -> tuple[str, int, dict]:
    period_s = 1.0 / float(cfg.env.runtime.rate_hz)
    frame_limit = int(cfg.env.runtime.frames)
    duration_s = float(cfg.app.record_control.episode_duration_s)
    diagnostics = _LoopDiagnostics(cfg.app.loop_diagnostics, logger, recorder, cfg.env.runtime.rate_hz)
    frame_index = 0
    key_listener.flush()
    episode_start = time.monotonic()
    info = {"teleop_active": False, "target_actions": (), "policy_action": (), "robot_states": ()}
    while frame_limit == 0 or frame_index < frame_limit:
        key_command = _recording_key_command(control_keys, key_listener.pop_key())
        if key_command is not None:
            diagnostics.flush(frame_index)
            return key_command, frame_index, info
        loop_start = time.monotonic()
        if duration_s > 0.0 and loop_start - episode_start >= duration_s:
            diagnostics.flush(frame_index)
            return "save", frame_index, info
        env_step_started = time.monotonic()
        _, _, _, _, info = env.step(action)
        env_step_finished = time.monotonic()
        sent_action = _hold_action_from_sent_actions(env, action, info)
        record_action_started = time.monotonic()
        recorder.record_action(
            policy_action=tuple(info["policy_action"]),
            sent_action=sent_action,
            teleop_active=bool(info["teleop_active"]),
        )
        record_action_finished = time.monotonic()
        action = sent_action
        frame_index += 1
        status_gui_started = time.monotonic()
        status_gui.update(
            _record_status_snapshot(
                cfg=cfg,
                frame_index=frame_index,
                info=info,
                record_state="recording",
                episode_number=episode_number,
                num_episodes=num_episodes,
                disk_episode_index=disk_episode_index,
            )
        )
        status_gui_finished = time.monotonic()
        loop_work_s = time.monotonic() - loop_start
        _sleep_remaining(loop_start, period_s)
        loop_total_s = time.monotonic() - loop_start
        diagnostics.observe(
            frame_index,
            _merged_stage_timings(
                {
                    "loop_work_s": loop_work_s,
                    "loop_total_s": loop_total_s,
                    "sleep_s": max(0.0, loop_total_s - loop_work_s),
                    "env_step_s": env_step_finished - env_step_started,
                    "record_action_s": record_action_finished - record_action_started,
                    "status_gui_update_s": status_gui_finished - status_gui_started,
                },
                info["timing"] if "timing" in info else {},
            ),
        )
        if recorder.has_error:
            diagnostics.flush(frame_index)
            return "error", frame_index, info
    diagnostics.flush(frame_index)
    return "save", frame_index, info


def _wait_for_episode_start(cfg: DictConfig, control_keys: "_RecordControlKeys", key_listener) -> str:
    import time

    if cfg.app.record_control.auto_start:
        return "start"
    key_listener.flush()
    while True:
        key_command = _waiting_key_command(control_keys, key_listener.pop_key())
        if key_command is not None:
            return key_command
        time.sleep(float(cfg.app.record_control.wait_poll_s))


def _waiting_key_command(control_keys: "_RecordControlKeys", key_name: str | None) -> str | None:
    if key_name is None:
        return None
    if key_name == control_keys.start:
        return "start"
    if key_name == control_keys.quit:
        return "quit"
    return None


def _recording_key_command(control_keys: "_RecordControlKeys", key_name: str | None) -> str | None:
    if key_name is None:
        return None
    if key_name == control_keys.save:
        return "save"
    if key_name == control_keys.redo:
        return "redo"
    if key_name == control_keys.quit:
        return "quit"
    return None


def _terminal_record_state(command: str) -> str | None:
    if command == "save":
        return "saving"
    if command == "redo":
        return "discarding"
    if command == "quit":
        return "discarding"
    return None


def _merged_stage_timings(*timings: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for timing in timings:
        for key, value in timing.items():
            merged[str(key)] = float(value)
    return merged


def _timing_stats_summary(stats_by_name: dict[str, _TimingStats]) -> str:
    if not stats_by_name:
        return "-"
    ordered_names = sorted(stats_by_name)
    return ", ".join(
        f"{_timing_stat_label(name)}={stats_by_name[name].avg_ms:.1f}/{stats_by_name[name].max_ms:.1f}ms"
        for name in ordered_names
    )


def _timing_stat_label(name: str) -> str:
    if name.endswith("_s"):
        return name[:-2]
    return name


def _rate_stats_summary(stats_by_name: dict[str, dict[str, float]]) -> str:
    if not stats_by_name:
        return "-"
    ordered_names = sorted(stats_by_name)
    return ", ".join(f"{name}={float(stats_by_name[name]['hz']):.1f}" for name in ordered_names)


def _record_status_snapshot(
    cfg: DictConfig,
    frame_index: int,
    info: dict,
    record_state: str,
    episode_number: int,
    num_episodes: int,
    disk_episode_index: int,
) -> StatusSnapshot:
    snapshot = _teleop_status_snapshot(cfg, frame_index, info)
    usage = (
        f"{snapshot.usage}\n\n"
        "录制控制\n"
        f"{str(cfg.app.record_control.start_key).upper()}: start\n"
        f"{str(cfg.app.record_control.save_key).upper()}: save\n"
        f"{str(cfg.app.record_control.redo_key).upper()}: redo\n"
        f"{str(cfg.app.record_control.quit_key).upper()}: quit"
    )
    return StatusSnapshot(
        robot=snapshot.robot,
        source=snapshot.source,
        mapping=snapshot.mapping,
        rate=snapshot.rate,
        frame=f"{frame_index} | episode {episode_number}/{num_episodes} | disk {disk_episode_index}",
        teleop=record_state if record_state != "recording" else snapshot.teleop,
        action=snapshot.action,
        usage=usage,
    )


@dataclass(frozen=True)
class _RecordControlKeys:
    start: str
    save: str
    redo: str
    quit: str


def _record_control_keys(cfg: DictConfig) -> _RecordControlKeys:
    return _RecordControlKeys(
        start=_normalize_key_name(str(cfg.start_key)),
        save=_normalize_key_name(str(cfg.save_key)),
        redo=_normalize_key_name(str(cfg.redo_key)),
        quit=_normalize_key_name(str(cfg.quit_key)),
    )


class _RecordKeyListener:
    def __init__(self) -> None:
        from pynput import keyboard

        self._lock = Lock()
        self._keys: list[str] = []
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.start()

    def pop_key(self) -> str | None:
        with self._lock:
            if not self._keys:
                return None
            return self._keys.pop(0)

    def flush(self) -> None:
        with self._lock:
            self._keys.clear()

    def close(self) -> None:
        self._listener.stop()

    def _on_press(self, key: Any) -> None:
        key_name = _key_name(key)
        with self._lock:
            self._keys.append(key_name)


class _NullRecordKeyListener:
    def pop_key(self) -> str | None:
        return None

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


def _key_name(key: Any) -> str:
    if hasattr(key, "char") and key.char is not None:
        return _normalize_key_name(str(key.char))
    return _normalize_key_name(str(key))


def _normalize_key_name(value: str) -> str:
    key_name = value.strip().lower()
    if key_name.startswith("key."):
        return key_name[4:]
    return key_name


if __name__ == "__main__":
    main()
