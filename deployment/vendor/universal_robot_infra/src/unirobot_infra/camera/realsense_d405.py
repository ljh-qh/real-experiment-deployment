"""Wrist camera (Intel RealSense D405) capture and recording.

Two layers so the recording logic is testable without hardware:

- ``CameraRecorder`` — sink: color frames go to an mp4 (imageio-ffmpeg),
  depth frames (uint16, mm) are throttled to ``depth_hz`` and written as
  16-bit PNGs, and every frame is indexed in ``timestamps.jsonl`` with both
  wall-clock and monotonic timestamps (teleop JSONL rows carry the same
  monotonic clock, so camera and teleop streams align offline).
- ``WristCamera`` — pyrealsense2 capture on a background thread. ``rotate``
  straightens the camera mounting orientation once at capture time, so live
  view AND recordings share the same upright frame.

The Tk live view (``_CameraViewManager`` / ``_VIEW``) and the unified
``toggle_cameras`` entry live in ``unirobot_infra.visualization.realsense_view``;
they are imported here as well because capture registers each started camera
with the shared view.
"""

import json
import math
from pathlib import Path
import threading
import time

import numpy as np

from unirobot_infra.visualization.realsense_view import _VIEW, toggle_cameras


class CameraRecorder:
    def __init__(self, out_dir, depth_hz=10.0, fps=30, recording=True):
        if not math.isfinite(depth_hz) or depth_hz <= 0:
            raise ValueError("depth_hz must be positive and finite")
        if not isinstance(fps, int) or fps < 1:
            raise ValueError("fps must be a positive integer")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "depth").mkdir(exist_ok=True)
        self.depth_interval = 1.0 / depth_hz
        self.fps = fps
        self._seq = 0
        self._depth_seq = 0
        self._seg = 0
        self._closed = False
        self._lock = threading.Lock()
        self._video = None
        self._index = None
        self._last_depth = None
        self._recording = False
        if recording:
            self._begin_segment()

    def _begin_segment(self):
        self._seg += 1
        import imageio.v2 as imageio
        self._video = imageio.get_writer(
            str(self.out_dir / f"color_take{self._seg:03d}.mp4"),
            fps=self.fps, macro_block_size=1)
        self._index = (self.out_dir / f"timestamps_take{self._seg:03d}.jsonl").open(
            "x", encoding="utf-8", buffering=1)
        self._index.write(json.dumps({"event": "start", "take": self._seg,
                                      "wall": round(time.time(), 6),
                                      "mono": round(time.monotonic(), 6)}) + "\n")
        self._last_depth = None
        self._recording = True

    def _end_segment(self):
        if self._index is not None:
            self._index.write(json.dumps({"event": "stop", "take": self._seg,
                                          "wall": round(time.time(), 6),
                                          "mono": round(time.monotonic(), 6)}) + "\n")
            self._index.close()
            self._index = None
        if self._video is not None:
            self._video.close()
            self._video = None
        self._recording = False

    @property
    def recording(self):
        return self._recording

    def set_recording(self, on):
        with self._lock:
            if self._closed or on == self._recording:
                return self._recording
            self._begin_segment() if on else self._end_segment()
        return self._recording

    def add_frame(self, color, depth, wall, mono, now=None):
        """One captured frame: color (H,W,3) uint8 RGB, depth (H,W) uint16 mm."""
        with self._lock:
            if self._closed or not self._recording:
                return
            self._seq += 1
            self._video.append_data(np.asarray(color, dtype=np.uint8))
            depth_name = None
            if depth is not None and (self._last_depth is None
                                      or mono - self._last_depth >= self.depth_interval):
                from PIL import Image
                self._depth_seq += 1
                depth_name = f"depth/depth_{self._depth_seq:06d}.png"
                Image.fromarray(np.asarray(depth, dtype=np.uint16), mode="I;16").save(
                    self.out_dir / depth_name)
                self._last_depth = mono
            self._index.write(json.dumps({"i": self._seq, "take": self._seg,
                                          "wall": round(wall, 6),
                                          "mono": round(mono, 6),
                                          "depth": depth_name}) + "\n")

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._end_segment()

    @property
    def frames(self):
        return self._seq


class WristCamera:
    def __init__(self, serial=None, width=640, height=480, fps=30, rotate=0,
                 recorder=None, live=True, role="wrist"):
        if rotate not in (0, 90, 180, 270):
            raise ValueError("rotate must be one of 0/90/180/270")
        self.serial = serial
        self.width, self.height, self.fps = width, height, fps
        self.rotate = rotate
        self.recorder = recorder
        self.live = live
        self.role = role
        self._thread = None
        self._stop = threading.Event()
        self._pipe = None
        self._latest = None  # (PIL Image for display, wall, mono)
        self._latest_lock = threading.Lock()
        self.frames = 0
        self.recoveries = 0
        self._last_frame_mono = None
        self._last_error_mono = 0.
        self._pipeline_failures = 0

    def start(self):
        import pyrealsense2 as rs
        self._rs = rs
        self._open_pipeline()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True,
                                        name=f"{self.role}-camera-capture")
        self._thread.start()
        if self.live:
            _VIEW.register(self)

    def _open_pipeline(self):
        rs = self._rs
        devices = rs.context().query_devices()
        dev = (devices[0] if self.serial is None and len(devices) else
               next((d for d in devices
                     if d.get_info(rs.camera_info.serial_number) == self.serial), None))
        if dev is None:
            raise RuntimeError(f"{self.role} RealSense SN {self.serial} not found")
        usb = dev.get_info(rs.camera_info.usb_type_descriptor)
        if float(usb) < 3.0:
            raise RuntimeError(
                f"{self.role} camera [{self.serial}] is on USB {usb}; "
                "640x480 RGB + depth at 30 Hz requires a stable USB 3.x connection")
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(dev.get_info(rs.camera_info.serial_number))
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        pipe.start(cfg)
        self._pipe = pipe

    def _log_capture_error(self, exc):
        now = time.monotonic()
        if now - self._last_error_mono >= 5.:
            print(f"CAMERA ERROR {self.role} [{self.serial}]: {exc}", flush=True)
            self._last_error_mono = now

    def _capture_loop(self):
        while not self._stop.is_set():
            try:
                frames = self._pipe.wait_for_frames(1000)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    raise RuntimeError("color or depth frame missing")
                color = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data())
                if self.rotate:
                    # np.rot90 逆时针，与 PIL rotate 一致；录制和显示同用摆正后的朝向
                    k = self.rotate // 90
                    color = np.rot90(color, k).copy()
                    depth = np.rot90(depth, k).copy()
                wall, mono = time.time(), time.monotonic()
                self.frames += 1
                self._last_frame_mono = mono
                self._pipeline_failures = 0
                if self.recorder is not None:
                    self.recorder.add_frame(color, depth, wall, mono)
                if self.live:
                    from PIL import Image
                    with self._latest_lock:
                        self._latest = (Image.fromarray(color), wall, mono)
            except Exception as exc:
                if self._stop.is_set():
                    break
                self._pipeline_failures += 1
                self._log_capture_error(exc)
                if self._pipeline_failures < 3:
                    continue
                try:
                    if self._pipe is not None:
                        self._pipe.stop()
                    self._pipe = None
                    self._open_pipeline()
                    self.recoveries += 1
                    self._pipeline_failures = 0
                    print(f"CAMERA RECOVERED {self.role} [{self.serial}]", flush=True)
                except Exception as restart_exc:
                    self._log_capture_error(restart_exc)
                    self._stop.wait(1.)

    def latest(self):
        with self._latest_lock:
            return self._latest

    def frame_age(self):
        if self._last_frame_mono is None:
            return float("inf")
        return max(0., time.monotonic() - self._last_frame_mono)

    @property
    def recording(self):
        return self.recorder.recording if self.recorder is not None else False

    def set_recording(self, on):
        if self.recorder is None:
            return False
        state = self.recorder.set_recording(on)
        print(f"Wrist camera recording {'ON (new take)' if state else 'OFF'}", flush=True)
        return state

    def toggle_recording(self):
        return self.set_recording(not self.recording)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._pipe is not None:
            try:
                self._pipe.stop()
            except Exception:
                pass
            self._pipe = None
        if self.recorder is not None:
            self.recorder.close()


def default_camera_dir(record_path=None, role="wrist"):
    """Next to the --record file when present, else .local/<role>-camera/<ts>."""
    if record_path is not None:
        record_path = Path(record_path)
        return record_path.parent / (record_path.stem + f".{role}")
    return Path.cwd() / ".local" / f"{role}-camera-{time.strftime('%Y%m%d-%H%M%S')}"
