from dataclasses import dataclass
from pathlib import Path
import time

import hydra
from omegaconf import DictConfig
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from unirobot_infra.core.types import RobotAction
from unirobot_infra.utils.logging import get_logger, setup_logging
from unirobot_infra.utils.sdk_output_router import start_sdk_output_router_from_config
from unirobot_infra.utils.status_gui import StatusGui, StatusSnapshot
from unirobot_infra.wrappers.base import EnvLike
from unirobot_infra.wrappers.builder import build_env


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
    def __init__(self, cfg, logger, target_hz: float) -> None:
        self.enabled = bool(cfg.enabled)
        self.log_interval_s = float(cfg.log_interval_s)
        if self.enabled and self.log_interval_s <= 0.0:
            raise ValueError("app.loop_diagnostics.log_interval_s must be positive")
        self.logger = logger
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
        self.logger.info(
            "teleop 频率诊断 frame=%s actual_hz=%.2f target_hz=%.2f stages=%s",
            frame_index,
            actual_hz,
            self.target_hz,
            _timing_stats_summary(self.stage_stats),
        )
        self.window_start_monotonic = time.monotonic()
        self.window_start_frame = frame_index
        self.stage_stats = {}


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="apps/teleop/default")
def main(cfg: DictConfig) -> None:
    console = Console(stderr=True)
    setup_logging(cfg.logging.level, console=console)
    logger = get_logger(__name__)
    with start_sdk_output_router_from_config(cfg.app.sdk_output) as sdk_output_router:
        if sdk_output_router is not None:
            logger.info("SDK 输出重定向目录=%s", sdk_output_router.run_dir)
        logger.info(
            "teleop 启动 robot=%s sides=%s source=%s",
            cfg.robot.name,
            _robot_sides_summary(cfg),
            cfg.wrappers.teleop_intervention.source,
        )
        _run_teleop(cfg, logger, console)


def _run_teleop(
    cfg: DictConfig,
    logger,
    console: Console | None = None,
) -> None:
    logger.info("env wrappers=%s rate_hz=%s", ",".join(cfg.wrappers), cfg.env.runtime.rate_hz)
    env = build_env(cfg, console=console)
    try:
        _run_teleop_loop(cfg, env, logger)
    finally:
        env.close()


def _run_teleop_loop(
    cfg: DictConfig,
    env: EnvLike,
    logger,
) -> None:
    period_s = 1.0 / float(cfg.env.runtime.rate_hz)
    diagnostics = _LoopDiagnostics(cfg.app.loop_diagnostics, logger, cfg.env.runtime.rate_hz)
    frame_limit = int(cfg.env.runtime.frames)
    frame_index = 0
    _, info = env.reset()
    action = env.action_layout.hold_action_from_states(tuple(info["robot_states"]))
    status_info = {"teleop_active": False, "target_actions": ()}
    with StatusGui.from_config(cfg.app.status_gui) as status_gui:
        if cfg.app.status_gui.enabled:
            logger.info("teleop 状态 GUI 已启动")
        status_gui.update(_teleop_status_snapshot(cfg, frame_index, status_info))
        while frame_limit == 0 or frame_index < frame_limit:
            loop_start = time.monotonic()
            env_step_started = time.monotonic()
            _, _, _, _, info = env.step(action)
            env_step_finished = time.monotonic()
            action = _hold_action_from_sent_actions(env, action, info)
            frame_index += 1
            status_gui_started = time.monotonic()
            status_gui.update(_teleop_status_snapshot(cfg, frame_index, info))
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
                        "status_gui_update_s": status_gui_finished - status_gui_started,
                    },
                    info["timing"] if "timing" in info else {},
                ),
            )
    diagnostics.flush(frame_index)


def _sleep_remaining(loop_start: float, period_s: float) -> None:
    import time

    elapsed_s = time.monotonic() - loop_start
    if elapsed_s < period_s:
        time.sleep(period_s - elapsed_s)


def _hold_action_from_sent_actions(env: EnvLike, previous_action: tuple[float, ...], info: dict) -> tuple[float, ...]:
    return env.action_layout.overlay_actions(previous_action, tuple(info["sent_actions"]))


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


def _robot_sides_summary(cfg: DictConfig) -> str:
    return ", ".join(f"{side}:{'+'.join(cfg.robot.sides[side].parts)}" for side in cfg.robot.sides)


def _actions_summary(actions: tuple[RobotAction, ...]) -> str:
    if not actions:
        return "none"
    return ", ".join(_action_summary(action) for action in actions)


def _action_summary(action: RobotAction) -> str:
    parts = []
    if action.arm_q_rad is not None:
        parts.append(f"arm({len(action.arm_q_rad)})")
    if action.end_effector_q_rad is not None:
        parts.append(f"hand({len(action.end_effector_q_rad)})")
    return f"{action.side}:{'+'.join(parts)}"


def _teleop_status_panel(
    cfg: DictConfig,
    frame_index: int,
    info: dict,
) -> Panel:
    snapshot = _teleop_status_snapshot(cfg, frame_index, info)
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="cyan", no_wrap=True)
    summary.add_column()
    summary.add_row("robot", snapshot.robot)
    summary.add_row("source", snapshot.source)
    summary.add_row("mapping", snapshot.mapping)
    summary.add_row("rate", snapshot.rate)
    summary.add_row("frame", snapshot.frame)
    summary.add_row("teleop", snapshot.teleop)
    summary.add_row("action", snapshot.action)

    return Panel(
        Group(summary, Text(""), Text(snapshot.usage)),
        title="unirobot_infra Teleop",
        border_style="blue",
    )


def _teleop_status_snapshot(cfg: DictConfig, frame_index: int, info: dict) -> StatusSnapshot:
    return StatusSnapshot(
        robot=f"{cfg.robot.name} ({_robot_sides_summary(cfg)})",
        source=str(cfg.wrappers.teleop_intervention.source),
        mapping=str(cfg.wrappers.teleop_intervention.mapping.type),
        rate=f"{float(cfg.env.runtime.rate_hz):.1f} Hz",
        frame=str(frame_index),
        teleop="active" if bool(info["teleop_active"]) else "idle",
        action=_actions_summary(tuple(info["target_actions"])) if bool(info["teleop_active"]) else "none",
        usage=_teleop_usage_text(cfg).plain,
    )


def _teleop_usage_text(cfg: DictConfig) -> Text:
    source = str(cfg.wrappers.teleop_intervention.source)
    if source == "keyboard":
        return _keyboard_usage_text(
            cfg.wrappers.teleop_intervention.device.keyboard.key_bindings,
            cfg.robot.hardware.arm.robot_world_frame,
        )
    if source == "vive":
        text = Text()
        text.append("使用说明\n", style="bold")
        if cfg.wrappers.teleop_intervention.mapping.type != "tianji_vive_local_se3_ik":
            raise ValueError("unsupported vive mapping.type")
        text.append("按住 I 同时同步左/右 Vive tracker local 6D 增量到 Tianji TCP local 6D 增量；松开停止。")
        return text
    if source == "manus":
        text = Text()
        text.append("使用说明\n", style="bold")
        text.append("佩戴 Manus 手套并保持 Manus client 输出，手势会 retarget 到 Sharpa。")
        return text
    if source == "manus_vive":
        text = Text()
        text.append("使用说明\n", style="bold")
        if cfg.wrappers.teleop_intervention.mapping.type != "manus_vive":
            raise ValueError("unsupported manus_vive mapping.type")
        text.append("Manus 手套控制 Sharpa；按住 I 同时同步左/右 Vive tracker local 6D 增量到 Tianji TCP local 6D 增量，松开停止 Tianji 同步。")
        return text
    if source == "joint_gui":
        text = Text()
        text.append("使用说明\n", style="bold")
        text.append("打开启动时输出的 Joint GUI URL；拖动滑块会通过 env.step 发送到机器人。")
        return text
    raise ValueError("unsupported teleop source")


def _keyboard_usage_text(key_bindings, robot_world_frame) -> Text:
    command_names = {
        "x_pos": "+x",
        "x_neg": "-x",
        "y_pos": "+y",
        "y_neg": "-y",
        "z_pos": "+z",
        "z_neg": "-z",
        "roll_pos": "+roll",
        "roll_neg": "-roll",
        "pitch_pos": "+pitch",
        "pitch_neg": "-pitch",
        "yaw_pos": "+yaw",
        "yaw_neg": "-yaw",
    }
    lines = [
        "使用说明",
        (
            f"xyz 平移坐标系: robot_world "
            f"(+x {robot_world_frame.x}, +y {robot_world_frame.y}, +z {robot_world_frame.z})"
        ),
        "abc 旋转仍使用 Tianji SDK xyzabc 姿态参数",
    ]
    for command in key_bindings:
        lines.append(f"{str(key_bindings[command]).upper()}: {command_names[command]}")
    return Text("\n".join(lines))


if __name__ == "__main__":
    main()
