from __future__ import annotations

from pathlib import Path
import time

import hydra
from omegaconf import DictConfig

from unirobot_infra.camera.insta360_capture import Insta360V4L2Capture
from unirobot_infra.camera.zed_capture import ZedCapture
from unirobot_infra.perception.types import Detection2D
from unirobot_infra.perception.yoloe import YoloEDetector
from unirobot_infra.utils.logging import get_logger, setup_logging


CONFIG_PATH = str(Path(__file__).resolve().parents[4] / "configs")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="apps/perception/detection_loop")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.logging.level)
    run_detection_loop(cfg, get_logger(__name__))


def run_detection_loop(cfg: DictConfig, logger) -> None:
    detector = YoloEDetector(cfg.detection.detector)
    camera = _build_rgb_camera(cfg)
    visualizer = _build_rerun_visualizer(cfg)
    frame_index = 0
    try:
        while _should_continue(frame_index, int(cfg.app.max_frames)):
            frame_start = time.monotonic()
            ok, image_rgb = camera.read()
            if not ok or image_rgb is None:
                raise RuntimeError("RGB camera read failed")
            detect_start = time.monotonic()
            detections = detector.detect(image_rgb)
            latency_ms = (time.monotonic() - detect_start) * 1000.0
            if visualizer is not None:
                _log_detection_frame(visualizer, cfg, frame_index, image_rgb, detections)
            if _should_log_frame(frame_index, int(cfg.app.log_every_n_frames)):
                logger.info(
                    "检测帧完成 frame=%s detections=%s latency_ms=%.1f",
                    frame_index,
                    len(detections),
                    latency_ms,
                )
            frame_index += 1
            _sleep_remaining(frame_start, 1.0 / float(cfg.app.rate_hz))
    finally:
        camera.close()


def _build_rgb_camera(cfg: DictConfig):
    camera_name = str(cfg.detection.camera_name)
    camera_cfg = cfg.camera.cameras[camera_name]
    if camera_cfg.type == "zed_sdk":
        return ZedCapture(camera_name, camera_cfg)
    if camera_cfg.type == "insta360_v4l2":
        return Insta360V4L2Capture(camera_name, camera_cfg)
    raise ValueError(f"unsupported camera type for detection_loop: {camera_cfg.type}")


def _build_rerun_visualizer(cfg: DictConfig):
    if not bool(cfg.visualize.rerun.enabled):
        return None
    import rerun as rr

    rr.init(str(cfg.visualize.rerun.app_id), spawn=bool(cfg.visualize.rerun.spawn))
    return {"rr": rr, "image_entity": str(cfg.visualize.image_entity)}


def _log_detection_frame(
    visualizer,
    cfg: DictConfig,
    frame_index: int,
    image_rgb,
    detections: tuple[Detection2D, ...],
) -> None:
    rr = visualizer["rr"]
    image_entity = visualizer["image_entity"]
    rr.set_time_sequence("frame", frame_index)
    rr.log(image_entity, rr.Image(image_rgb))
    rr.log(
        f"{image_entity}/detections",
        rr.Boxes2D(
            mins=_bbox_mins(detections),
            sizes=_bbox_sizes(detections),
            labels=[f"{detection.label} {detection.score:.2f}" for detection in detections],
            colors=[tuple(int(value) for value in cfg.visualize.box_color_rgb)] * len(detections),
            show_labels=bool(cfg.visualize.show_labels),
        ),
    )


def _bbox_mins(detections: tuple[Detection2D, ...]) -> list[tuple[float, float]]:
    return [(detection.bbox.x1, detection.bbox.y1) for detection in detections]


def _bbox_sizes(detections: tuple[Detection2D, ...]) -> list[tuple[float, float]]:
    return [
        (detection.bbox.x2 - detection.bbox.x1, detection.bbox.y2 - detection.bbox.y1)
        for detection in detections
    ]


def _should_continue(frame_index: int, max_frames: int) -> bool:
    return max_frames <= 0 or frame_index < max_frames


def _should_log_frame(frame_index: int, log_every_n_frames: int) -> bool:
    return log_every_n_frames > 0 and frame_index % log_every_n_frames == 0


def _sleep_remaining(loop_start: float, period_s: float) -> None:
    remaining_s = period_s - (time.monotonic() - loop_start)
    if remaining_s > 0.0:
        time.sleep(remaining_s)


if __name__ == "__main__":
    main()
