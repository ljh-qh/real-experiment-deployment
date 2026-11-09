from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from unirobot_infra.camera import CameraManager
from unirobot_infra.utils.logging import get_logger, setup_logging


CONFIG_PATH = str(Path(__file__).resolve().parents[3] / "configs")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="apps/realtime_capture/default")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.logging.level)
    logger = get_logger(__name__)
    logger.info("realtime_capture 启动 cameras=%s", ",".join(str(name) for name in cfg.realtime_capture.camera_names))
    _run_realtime_capture(cfg, logger)


def _run_realtime_capture(cfg: DictConfig, logger) -> None:
    import cv2

    save_dir = Path(cfg.realtime_capture.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    display_enabled = bool(cfg.realtime_capture.display)
    camera_cfg = _select_camera_config(cfg)
    manager = CameraManager(camera_cfg)
    frame_index = 0
    try:
        while int(cfg.realtime_capture.max_frames) == 0 or frame_index < int(cfg.realtime_capture.max_frames):
            images = manager.read(timeout_s=float(cfg.realtime_capture.read_timeout_s))
            display = _display_image(
                images=images,
                panel_width=int(cfg.realtime_capture.panel_width),
            )
            if display_enabled:
                cv2.imshow(str(cfg.realtime_capture.window_name), display)
                key = cv2.waitKey(int(cfg.realtime_capture.wait_key_ms)) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    path = save_dir / f"capture_{frame_index:06d}.jpg"
                    cv2.imwrite(str(path), display)
                    logger.info("保存当前画面 %s", path)
            frame_index += 1
    finally:
        manager.close()
        if display_enabled:
            cv2.destroyAllWindows()


def _display_image(images: dict[str, np.ndarray], panel_width: int) -> np.ndarray:
    panels = [
        _labeled_panel(name=camera_name, rgb_image=image, panel_width=panel_width)
        for camera_name, image in images.items()
    ]
    return np.hstack(panels)


def _select_camera_config(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(
        {
            str(camera_name): cfg.camera.cameras[camera_name]
            for camera_name in cfg.realtime_capture.camera_names
        }
    )


def _labeled_panel(name: str, rgb_image: np.ndarray, panel_width: int) -> np.ndarray:
    import cv2

    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("camera image must have shape HxWx3")
    scale = panel_width / float(rgb_image.shape[1])
    panel_height = max(1, int(round(rgb_image.shape[0] * scale)))
    panel_rgb = cv2.resize(rgb_image, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
    panel_bgr = panel_rgb[..., ::-1].copy()
    cv2.rectangle(panel_bgr, (0, 0), (panel_bgr.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(
        panel_bgr,
        name,
        (10, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel_bgr


if __name__ == "__main__":
    main()
