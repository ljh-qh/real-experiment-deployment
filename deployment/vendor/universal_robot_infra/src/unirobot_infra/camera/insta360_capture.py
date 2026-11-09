import re
import subprocess
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from unirobot_infra.utils.logging import get_logger


logger = get_logger(__name__)

USB_PORT_RE = re.compile(r"^\d+-\d+(?:\.\d+)*$")
INSTA360_WARMUP_FRAMES = 3


class Insta360V4L2Capture:
    def __init__(self, name: str, cfg: DictConfig) -> None:
        self.name = name
        self.device_path = str(cfg.device_path)
        self.image_size = _image_size_tuple(cfg.image_size)
        _validate_device_path(self.device_path)
        self.video_index = _video_index_from_path(self.device_path)
        _validate_v4l_identity(self.device_path, cfg)
        self._frame_size_bytes = self.image_size[0] * self.image_size[1] * 3

        logger.info(
            "启动 Insta360 V4L2 相机 name=%s path=%s video_index=%s size=%sx%s fps=%s fourcc=%s",
            self.name,
            self.device_path,
            self.video_index,
            self.image_size[0],
            self.image_size[1],
            cfg.fps,
            cfg.fourcc,
        )
        self._process = _start_ffmpeg_process(self.device_path, self.image_size, cfg)
        try:
            self._warmup(INSTA360_WARMUP_FRAMES)
        except Exception as exc:
            logger.error("Insta360 V4L2 相机 %s 初始化失败: %s", self.name, exc)
            self.close()
            raise
        logger.info("Insta360 V4L2 相机已就绪 name=%s", self.name)

    def read(self) -> tuple[bool, np.ndarray | None]:
        frame_bytes = self._process.stdout.read(self._frame_size_bytes)
        if len(frame_bytes) != self._frame_size_bytes:
            raise RuntimeError(
                f"Insta360 V4L2 camera {self.name} failed to read frame: {_ffmpeg_error_message(self._process)}"
            )
        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((self.image_size[1], self.image_size[0], 3)).copy()
        _validate_frame_shape(self.name, frame, self.image_size)
        return True, frame

    def close(self) -> None:
        if self._process.poll() is None:
            self._process.kill()
            self._process.wait()
        self._process.stdout.close()
        self._process.stderr.close()

    def _warmup(self, warmup_frames: int) -> None:
        last_frame: np.ndarray | None = None
        for _ in range(warmup_frames):
            ok, frame = self.read()
            if ok and frame is not None:
                last_frame = frame
        if warmup_frames > 0 and last_frame is None:
            raise RuntimeError(f"Insta360 V4L2 camera {self.name} did not produce a warmup frame")
        if last_frame is not None:
            _validate_frame_shape(self.name, last_frame, self.image_size)


def _start_ffmpeg_process(device_path: str, image_size: tuple[int, int], cfg: DictConfig) -> subprocess.Popen:
    fourcc = str(cfg.fourcc)
    if len(fourcc) != 4:
        raise ValueError(f"Insta360 V4L2 camera fourcc must have 4 characters: {fourcc}")
    resolved_path = str(Path(device_path).resolve())
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-fflags",
        "nobuffer",
        "-f",
        "v4l2",
        "-input_format",
        _ffmpeg_input_format(fourcc),
        "-video_size",
        f"{image_size[0]}x{image_size[1]}",
        "-framerate",
        str(int(cfg.fps)),
        "-i",
        resolved_path,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=image_size[0] * image_size[1] * 3 * 2,
    )
    if process.stdout is None or process.stderr is None:
        raise RuntimeError(f"Insta360 V4L2 camera failed to start ffmpeg: {resolved_path}")
    return process


def _ffmpeg_input_format(fourcc: str) -> str:
    if fourcc == "MJPG":
        return "mjpeg"
    return fourcc.lower()


def _ffmpeg_error_message(process: subprocess.Popen) -> str:
    stderr = process.stderr.read().decode("utf-8", errors="ignore").strip()
    if stderr:
        return stderr
    if process.returncode is not None:
        return f"ffmpeg exited with returncode={process.returncode}"
    return "ffmpeg stream ended unexpectedly"


def _validate_device_path(device_path: str) -> None:
    if not device_path:
        raise ValueError("Insta360 V4L2 camera device_path must not be empty")
    if not Path(device_path).exists():
        raise FileNotFoundError(f"Insta360 V4L2 camera device_path does not exist: {device_path}")


def _validate_v4l_identity(device_path: str, cfg: DictConfig) -> None:
    video_index = _video_index_from_path(device_path)
    expected_name = str(cfg.device_name_contains).strip()
    if expected_name:
        device_name = _v4l_device_name(video_index)
        if expected_name not in device_name:
            raise RuntimeError(
                f"Insta360 V4L2 camera path={device_path} name mismatch: "
                f"expected contains {expected_name!r}, actual {device_name!r}"
            )

    expected_usb_port = str(cfg.usb_port).strip()
    if expected_usb_port:
        usb_port = _v4l_usb_port(video_index)
        if expected_usb_port != usb_port:
            raise RuntimeError(
                f"Insta360 V4L2 camera path={device_path} usb_port mismatch: "
                f"expected {expected_usb_port!r}, actual {usb_port!r}"
            )


def _video_index_from_path(device_path: str) -> int:
    resolved = Path(device_path).resolve()
    match = re.fullmatch(r"video(\d+)", resolved.name)
    if match is None:
        raise RuntimeError(f"Insta360 V4L2 camera path must resolve to /dev/videoN: {device_path}")
    return int(match.group(1))


def _v4l_device_name(video_index: int) -> str:
    return Path(f"/sys/class/video4linux/video{video_index}/name").read_text(encoding="utf-8").strip()


def _v4l_usb_port(video_index: int) -> str:
    device_path = Path(f"/sys/class/video4linux/video{video_index}/device").resolve()
    for part in device_path.parts:
        if USB_PORT_RE.fullmatch(part):
            return part
    raise RuntimeError(f"cannot resolve USB port for /dev/video{video_index}: {device_path}")


def _validate_frame_shape(name: str, frame: np.ndarray, image_size: tuple[int, int]) -> None:
    width, height = image_size
    if frame.ndim != 3:
        raise RuntimeError(f"Insta360 V4L2 camera {name} produced invalid frame shape: {frame.shape}")
    if frame.shape[1] != width or frame.shape[0] != height:
        raise RuntimeError(
            f"Insta360 V4L2 camera {name} produced frame shape {frame.shape}, "
            f"expected height={height} width={width}"
        )


def _image_size_tuple(image_size) -> tuple[int, int]:
    values = tuple(int(value) for value in image_size)
    if len(values) != 2:
        raise ValueError("camera image_size must have length 2")
    return values[0], values[1]
