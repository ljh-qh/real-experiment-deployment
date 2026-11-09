from dataclasses import dataclass, field
from pathlib import Path
import statistics
import time

import hydra
from omegaconf import DictConfig

from unirobot_infra.devices.vive import ViveTrackerDevice, ViveTrackerFrame
from unirobot_infra.utils.logging import get_logger, setup_logging


CONFIG_PATH = str(Path(__file__).resolve().parents[3] / "configs")


@dataclass
class ViveProbeStats:
    frames: int = 0
    valid_tracker_counts: list[int] = field(default_factory=list)
    latency_ms: list[float] = field(default_factory=list)
    inter_arrival_ms: list[float] = field(default_factory=list)
    last_received_timestamp_ns: int | None = None

    def add_frame(self, frame: ViveTrackerFrame) -> None:
        self.frames += 1
        self.valid_tracker_counts.append(len(frame.valid_trackers))
        self.latency_ms.append(frame.latency_ms)
        if self.last_received_timestamp_ns is not None:
            self.inter_arrival_ms.append((frame.received_timestamp_ns - self.last_received_timestamp_ns) / 1_000_000.0)
        self.last_received_timestamp_ns = frame.received_timestamp_ns


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="apps/vive_probe")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.logging.level)
    logger = get_logger(__name__)
    stats = probe_vive(cfg, logger)
    if stats.frames == 0:
        raise TimeoutError(f"Vive tracker no frames from {cfg.device.vive.endpoint} topic={cfg.device.vive.topic}")


def probe_vive(cfg: DictConfig, logger) -> ViveProbeStats:
    logger.info("Vive probe endpoint=%s topic=%s", cfg.device.vive.endpoint, cfg.device.vive.topic)
    stats = ViveProbeStats()
    device = ViveTrackerDevice(cfg.device.vive)
    device.connect()
    try:
        deadline = time.monotonic() + float(cfg.probe.duration_s)
        max_frames = int(cfg.probe.max_frames)
        while time.monotonic() < deadline and (max_frames == 0 or stats.frames < max_frames):
            frame = device.recv_latest(timeout_s=float(cfg.probe.timeout_s))
            if frame is None:
                logger.info("Vive frame timeout")
                continue
            stats.add_frame(frame)
            logger.info(
                "frame=%s trackers=%s valid=%s latency_ms=%.2f ids=%s",
                stats.frames,
                len(frame.trackers),
                len(frame.valid_trackers),
                frame.latency_ms,
                ",".join(tracker.tracker_id for tracker in frame.valid_trackers),
            )
    finally:
        device.close()

    logger.info("summary %s", vive_probe_summary(stats))
    return stats


def vive_probe_summary(stats: ViveProbeStats) -> str:
    if stats.frames == 0:
        return "frames=0"
    return (
        f"frames={stats.frames} "
        f"valid_trackers_min={min(stats.valid_tracker_counts)} "
        f"valid_trackers_max={max(stats.valid_tracker_counts)} "
        f"latency_ms_min={min(stats.latency_ms):.2f} "
        f"latency_ms_p50={statistics.median(stats.latency_ms):.2f} "
        f"latency_ms_max={max(stats.latency_ms):.2f} "
        f"inter_arrival_ms_p50={_median_or_zero(stats.inter_arrival_ms):.2f}"
    )


def _median_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


if __name__ == "__main__":
    main()
