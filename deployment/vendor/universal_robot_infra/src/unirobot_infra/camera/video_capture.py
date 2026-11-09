import queue
import threading

import numpy as np

from unirobot_infra.utils.logging import get_logger


logger = get_logger(__name__)


class VideoCapture:
    def __init__(self, capture, name: str) -> None:
        self.capture = capture
        self.name = name
        self.queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._running = True
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._reader, name=f"camera-{name}", daemon=True)
        self._thread.start()
        logger.debug("启动相机读帧线程 name=%s", self.name)

    def read(self, timeout_s: float) -> np.ndarray:
        if self._error is not None:
            raise RuntimeError(f"camera {self.name} reader failed: {self._error}")
        try:
            return self.queue.get(timeout=timeout_s)
        except queue.Empty:
            if self._error is not None:
                raise RuntimeError(f"camera {self.name} reader failed: {self._error}")
            raise TimeoutError(f"camera {self.name} frame timeout")

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self.capture.close()
        logger.debug("停止相机读帧线程 name=%s", self.name)

    def _reader(self) -> None:
        try:
            while self._running:
                ok, frame = self.capture.read()
                if not ok:
                    continue
                if self.queue.full():
                    self.queue.get_nowait()
                self.queue.put_nowait(frame)
        except BaseException as error:
            self._error = error
            self._running = False
