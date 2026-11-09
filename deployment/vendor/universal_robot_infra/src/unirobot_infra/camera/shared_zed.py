from collections import OrderedDict
from dataclasses import dataclass
import multiprocessing as mp
import queue
import threading
import time

import numpy as np
from omegaconf import DictConfig

from unirobot_infra.camera.rgbd import RgbdFrame
from unirobot_infra.camera.zed_capture import ZedCapture
from unirobot_infra.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class SharedZedRgbdFrameBus:
    queue: mp.Queue
    enabled_event: mp.Event

    def start(self) -> None:
        self.enabled_event.set()

    def stop(self) -> None:
        self.enabled_event.clear()
        self.drain()

    def drain(self) -> None:
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                return

    def close(self) -> None:
        self.stop()
        self.queue.cancel_join_thread()
        self.queue.close()


class SharedZedCameraManager:
    def __init__(
        self,
        camera_name: str,
        camera_cfg: DictConfig,
        rgbd_frame_bus: SharedZedRgbdFrameBus,
        rgbd_target_hz: float,
    ) -> None:
        self.source = SharedZedRgbdFrameSource(
            camera_name=camera_name,
            camera_cfg=camera_cfg,
            rgbd_frame_bus=rgbd_frame_bus,
            rgbd_target_hz=rgbd_target_hz,
        )
        self.captures = OrderedDict([(camera_name, SharedZedRgbCapture(self.source))])

    def close(self) -> None:
        self.source.close()
        self.captures.clear()


class SharedZedRgbCapture:
    def __init__(self, source: "SharedZedRgbdFrameSource") -> None:
        self.source = source

    def read(self, timeout_s: float) -> np.ndarray:
        return self.source.read_rgb(timeout_s)

    def close(self) -> None:
        return None


class SharedZedRgbdFrameSource:
    def __init__(
        self,
        camera_name: str,
        camera_cfg: DictConfig,
        rgbd_frame_bus: SharedZedRgbdFrameBus,
        rgbd_target_hz: float,
    ) -> None:
        self.camera_name = str(camera_name)
        self.capture = ZedCapture(self.camera_name, camera_cfg)
        self.rgbd_frame_bus = rgbd_frame_bus
        self.rgbd_target_hz = float(rgbd_target_hz)
        if self.rgbd_target_hz <= 0.0:
            raise ValueError("shared ZED rgbd_target_hz must be positive")
        self.rgb_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._running = True
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._reader, name=f"shared-zed-{self.camera_name}", daemon=True)
        self._thread.start()
        logger.info(
            "共享 ZED 相机已启动 name=%s rgbd_target_hz=%.3f",
            self.camera_name,
            self.rgbd_target_hz,
        )

    def read_rgb(self, timeout_s: float) -> np.ndarray:
        if self._error is not None:
            raise RuntimeError(f"shared ZED camera {self.camera_name} failed: {self._error}")
        try:
            return self.rgb_queue.get(timeout=timeout_s)
        except queue.Empty:
            if self._error is not None:
                raise RuntimeError(f"shared ZED camera {self.camera_name} failed: {self._error}")
            raise TimeoutError(f"shared ZED camera {self.camera_name} RGB frame timeout")

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self.capture.close()
        self.rgbd_frame_bus.close()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        logger.info("共享 ZED 相机已关闭 name=%s", self.camera_name)

    def _reader(self) -> None:
        try:
            period_s = 1.0 / self.rgbd_target_hz
            next_rgbd_time_s = 0.0
            while self._running:
                now_s = time.monotonic()
                if self.rgbd_frame_bus.enabled_event.is_set() and now_s >= next_rgbd_time_s:
                    ok, rgbd = self.capture.read_rgbd()
                    if ok and rgbd is not None:
                        self._publish_rgb(rgbd.rgb)
                        self._publish_rgbd(rgbd)
                        next_rgbd_time_s = time.monotonic() + period_s
                    continue

                ok, rgb = self.capture.read()
                if ok and rgb is not None:
                    self._publish_rgb(rgb)
        except BaseException as error:
            self._error = error
            self._running = False

    def _publish_rgb(self, rgb: np.ndarray) -> None:
        _put_latest(self.rgb_queue, np.asarray(rgb, dtype=np.uint8).copy())

    def _publish_rgbd(self, rgbd: RgbdFrame) -> None:
        if self.rgbd_frame_bus.enabled_event.is_set():
            _put_latest(self.rgbd_frame_bus.queue, rgbd)


def _put_latest(target_queue, value) -> None:
    while True:
        try:
            target_queue.put_nowait(value)
            return
        except queue.Full:
            try:
                target_queue.get_nowait()
            except queue.Empty:
                return
