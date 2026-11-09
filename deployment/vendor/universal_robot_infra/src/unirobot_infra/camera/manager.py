from collections import OrderedDict

import numpy as np
from omegaconf import DictConfig

from unirobot_infra.camera.insta360_capture import Insta360V4L2Capture
from unirobot_infra.camera.video_capture import VideoCapture
from unirobot_infra.camera.zed_capture import ZedCapture
from unirobot_infra.utils.logging import get_logger


logger = get_logger(__name__)


class CameraManager:
    def __init__(self, cameras: DictConfig) -> None:
        if not cameras:
            raise ValueError("camera.cameras must not be empty")
        self.captures: OrderedDict[str, VideoCapture] = OrderedDict()
        logger.info("初始化相机管理器 cameras=%s", ",".join(str(name) for name in cameras))
        try:
            for camera_name in cameras:
                camera_cfg = cameras[camera_name]
                capture = _build_capture(str(camera_name), camera_cfg)
                self.captures[str(camera_name)] = VideoCapture(capture, name=str(camera_name))
                logger.info(
                    "相机已启动 name=%s type=%s fps=%s",
                    camera_name,
                    camera_cfg.type,
                    camera_cfg.fps,
                )
        except Exception:
            self.close()
            raise

    def read(self, timeout_s: float) -> dict[str, np.ndarray]:
        return {
            camera_name: capture.read(timeout_s=timeout_s)
            for camera_name, capture in self.captures.items()
        }

    def close(self) -> None:
        logger.info("关闭相机管理器")
        for capture in self.captures.values():
            capture.close()
        self.captures.clear()


def _build_capture(camera_name: str, camera_cfg: DictConfig):
    if camera_cfg.type == "insta360_v4l2":
        return Insta360V4L2Capture(camera_name, camera_cfg)
    if camera_cfg.type == "zed_sdk":
        return ZedCapture(camera_name, camera_cfg)
    raise ValueError(f"unsupported camera type: {camera_cfg.type}")
