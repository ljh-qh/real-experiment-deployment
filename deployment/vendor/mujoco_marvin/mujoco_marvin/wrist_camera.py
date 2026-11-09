"""Wrist camera (Intel RealSense D405) capture, live view and recording.

Two layers so the recording logic is testable without hardware:

- ``CameraRecorder`` — sink: color frames go to an mp4 (imageio-ffmpeg),
  depth frames (uint16, mm) are throttled to ``depth_hz`` and written as
  16-bit PNGs, and every frame is indexed in ``timestamps.jsonl`` with both
  wall-clock and monotonic timestamps (teleop JSONL rows carry the same
  monotonic clock, so camera and teleop streams align offline).
- ``WristCamera`` — pyrealsense2 capture on a background thread, optional
  tkinter live view on its own thread. ``rotate`` straightens the camera
  mounting orientation once at capture time, so live view AND recordings
  share the same upright frame.
"""

import json
import math
import queue
from pathlib import Path
import threading
import time

import numpy as np


class CameraRecordingError(RuntimeError):
    """A disk/encoder failure, independent of the RealSense capture pipeline."""


class CameraRecorder:
    """Bounded, asynchronous writer; acquisition never waits for FFmpeg or PNG."""

    def __init__(self, out_dir, depth_hz=10.0, fps=30, recording=True,
                 queue_capacity=32, encoder_threads=2):
        if not math.isfinite(depth_hz) or depth_hz <= 0:
            raise ValueError("depth_hz must be positive and finite")
        if not isinstance(fps, int) or fps < 1:
            raise ValueError("fps must be a positive integer")
        if not isinstance(queue_capacity, int) or queue_capacity < 1:
            raise ValueError("queue_capacity must be a positive integer")
        if not isinstance(encoder_threads, int) or encoder_threads < 1:
            raise ValueError("encoder_threads must be a positive integer")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "depth").mkdir(exist_ok=True)
        self.depth_interval = 1.0 / depth_hz
        self.fps = fps
        self.queue_capacity = queue_capacity
        self.encoder_threads = encoder_threads
        self._seq = 0
        self._depth_seq = 0
        self._seg = 0
        self._closed = False
        self._lock = threading.Lock()
        self._video = None
        self._index = None
        self._last_depth = None
        self._recording = False
        self._error = None
        self._writer_thread = None
        self._queue = None
        self._writer_stop = threading.Event()
        self._max_write_ms = 0.
        self._max_queue_depth = 0
        self._enqueued_frames = 0
        if recording:
            self._begin_segment()

    @property
    def settings(self):
        return dict(depth_hz=1. / self.depth_interval, fps=self.fps,
                    queue_capacity=self.queue_capacity, encoder_threads=self.encoder_threads)

    @property
    def error(self):
        return self._error

    @property
    def diagnostics(self):
        return {"error": self._error, "frames_written": self._seq,
                "frames_enqueued": self._enqueued_frames,
                "queue_capacity": self.queue_capacity,
                "max_queue_depth": self._max_queue_depth,
                "queued_frames": self._queue.qsize() if self._queue is not None else 0,
                "writer_alive": bool(self._writer_thread and self._writer_thread.is_alive()),
                "max_write_ms": round(self._max_write_ms, 3),
                "depth_hz": 1. / self.depth_interval, "fps": self.fps,
                "encoder_threads": self.encoder_threads}

    def _fail(self, exc):
        if self._error is None:
            self._error = f"{type(exc).__name__}: {exc}"
            print(f"CAMERA RECORDING ERROR {self.out_dir.name}: {self._error}", flush=True)
        self._recording = False
        self._writer_stop.set()

    def _begin_segment(self):
        if self._error:
            raise CameraRecordingError(self._error)
        self._seg += 1
        import imageio.v2 as imageio
        self._video = imageio.get_writer(
            str(self.out_dir / f"color_take{self._seg:03d}.mp4"),
            fps=self.fps, macro_block_size=1,
            ffmpeg_params=["-threads", str(self.encoder_threads)])
        try:
            self._index = (self.out_dir / f"timestamps_take{self._seg:03d}.jsonl").open(
                "x", encoding="utf-8", buffering=1)
            self._index.write(json.dumps({"event": "start", "take": self._seg,
                                          "wall": round(time.time(), 6),
                                          "mono": round(time.monotonic(), 6)}) + "\n")
        except BaseException:
            self._video.close()
            self._video = None
            if self._index is not None:
                self._index.close()
                self._index = None
            raise
        self._last_depth = None
        self._queue = queue.Queue(maxsize=self.queue_capacity)
        self._writer_stop.clear()
        self._recording = True
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True,
                                                name=f"{self.out_dir.name}-camera-writer")
        self._writer_thread.start()

    def _writer_loop(self):
        try:
            while not self._writer_stop.is_set() or not self._queue.empty():
                if self._error is not None:
                    break
                try:
                    item = self._queue.get(timeout=.05)
                except queue.Empty:
                    continue
                started = time.monotonic()
                try:
                    self._write_frame(*item)
                finally:
                    self._max_write_ms = max(self._max_write_ms,
                                             (time.monotonic() - started) * 1000.)
                    self._queue.task_done()
        except Exception as exc:
            self._fail(exc)
        finally:
            # All writer resources belong to this thread, including on failure.
            # A failed/overflowed queue is discarded only with an explicit error.
            while True:
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except queue.Empty:
                    break
            try:
                if self._video is not None:
                    self._video.close()
            except Exception as exc:
                self._fail(exc)
            finally:
                self._video = None
            try:
                if self._index is not None:
                    self._index.write(json.dumps({"event": "stop", "take": self._seg,
                                                  "wall": round(time.time(), 6),
                                                  "mono": round(time.monotonic(), 6),
                                                  "error": self._error}) + "\n")
                    self._index.close()
            except Exception as exc:
                self._fail(exc)
            finally:
                self._index = None

    def _end_segment(self):
        self._recording = False
        self._writer_stop.set()
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=5.)
            if self._writer_thread.is_alive():
                self._fail(CameraRecordingError("encoder did not finalize within 5 seconds"))

    @property
    def recording(self):
        return self._recording

    def set_recording(self, on):
        with self._lock:
            if self._error and on:
                raise CameraRecordingError(self._error)
            if self._closed or on == self._recording:
                return self._recording
            self._begin_segment() if on else self._end_segment()
        return self._recording

    def add_frame(self, color, depth, wall, mono, now=None):
        """Copy SDK-backed buffers and enqueue, without waiting for encoder I/O."""
        with self._lock:
            if self._error:
                raise CameraRecordingError(self._error)
            if self._closed or not self._recording:
                return
            item = (np.array(color, dtype=np.uint8, order="C", copy=True),
                    np.array(depth, dtype=np.uint16, order="C", copy=True) if depth is not None else None,
                    wall, mono)
            try:
                self._queue.put_nowait(item)
                self._enqueued_frames += 1
                self._max_queue_depth = max(self._max_queue_depth, self._queue.qsize())
            except queue.Full:
                exc = CameraRecordingError("writer queue full; episode must be marked incomplete")
                self._fail(exc)
                raise exc

    def _write_frame(self, color, depth, wall, mono):
        self._video.append_data(color)
        depth_name = None
        if depth is not None and (self._last_depth is None
                                  or mono - self._last_depth >= self.depth_interval):
            from PIL import Image
            depth_name = f"depth/depth_{self._depth_seq + 1:06d}.png"
            Image.fromarray(depth).save(self.out_dir / depth_name)
            self._depth_seq += 1
            self._last_depth = mono
        sequence = self._seq + 1
        self._index.write(json.dumps({"i": sequence, "take": self._seg,
                                      "wall": round(wall, 6),
                                      "mono": round(mono, 6),
                                      "depth": depth_name}) + "\n")
        self._seq = sequence

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._end_segment()

    @property
    def frames(self):
        return self._seq


class _CameraViewManager:
    """Single window with all camera views and ONE record toggle.

    Tk runs entirely on the MAIN thread: `update()` + `sync()` are pumped
    from the teleop's viewer/headless loop. No GUI thread -> no cross-thread
    Tcl deletion on shutdown. `shutdown()` clears StringVar/PhotoImage refs
    and destroys the root on the same thread that created it."""

    def __init__(self):
        self._cameras = []
        self._root = None
        self._views = {}
        self._state_text = None
        self._lock = threading.Lock()
        self._closed = False

    def register(self, camera):
        with self._lock:
            self._cameras.append(camera)

    def toggle_all(self):
        with self._lock:
            cams = list(self._cameras)
        toggle_cameras(cams)

    def _ensure_root(self):
        if self._root is not None or self._closed:
            return
        import tkinter as tk
        from PIL import ImageTk
        self._tk = tk
        self._ImageTk = ImageTk
        self._root = tk.Tk()
        self._root.title("相机视角（主视角 + 双腕）——关窗只关显示")
        self._root.protocol("WM_DELETE_WINDOW", self.shutdown)
        self._grid = tk.Frame(self._root)
        self._grid.pack()
        self._state_text = tk.StringVar()
        tk.Button(self._root, textvariable=self._state_text,
                  font=("Sans", 14, "bold"),
                  command=self.toggle_all).pack(fill="x", padx=10, pady=8)

    def update(self):
        """Pump pending Tk events once (main thread's loop)."""
        if self._closed:
            return
        if self._root is None and self._cameras:
            self._ensure_root()
        if self._root is None:
            return
        try:
            self._root.update()
        except Exception:
            self.shutdown()

    def sync(self):
        """Create/refresh camera images (main thread's loop)."""
        if self._closed:
            return
        if self._root is None and self._cameras:
            self._ensure_root()
        if self._root is None:
            return
        try:
            self._sync_views()
        except Exception:
            # 窗口被外部销毁等情况：只停显示，不影响遥操/录制主循环
            self.shutdown()

    def _sync_views(self):
        with self._lock:
            cams = list(self._cameras)
        for index, cam in enumerate(cams):
            if cam not in self._views:
                column = self._tk.Frame(self._grid)
                column.grid(row=0, column=index, padx=4, pady=4)
                title = self._tk.Label(column, text=f"{cam.role} [{cam.serial[-4:] if cam.serial else ''}]",
                                       font=("Sans", 11, "bold"))
                title.pack()
                label = self._tk.Label(column)
                label.pack()
                self._views[cam] = {"title": title, "label": label,
                                    "photo": self._tk.PhotoImage(width=1, height=1)}
        for cam, view in self._views.items():
            item = cam.latest()
            if item is not None:
                # 主线程架构下按帧新建 PhotoImage 也安全（Tcl 同线程 GC）
                view["photo"] = self._ImageTk.PhotoImage(item[0])
                view["label"].configure(image=view["photo"])
            state = " — 录制失败（采集仍在运行）" if cam.recording_error else (
                " — 无新帧，正在重连" if cam.frame_age() > 2. else "")
            view["title"].configure(
                text=f"{cam.role} [{cam.serial[-4:] if cam.serial else ''}]{state}")
        any_recording = any(cam.recording for cam in cams)
        self._state_text.set("■ 停止录制（全部机位，每段一个文件）" if any_recording
                             else "● 开始录制（全部机位）")

    def shutdown(self):
        """Clear Tk object refs, then destroy the root — all on the caller's thread."""
        self._closed = True
        if self._root is None:
            return
        for view in self._views.values():
            view["photo"] = None
        self._views.clear()
        self._state_text = None
        try:
            self._root.destroy()
        except Exception:
            pass
        self._root = None


def toggle_cameras(cameras):
    """Start/stop recording: the unified episode manager when configured,
    otherwise each camera's own recorder (legacy per-camera takes)."""
    from .episode import EPISODES
    if EPISODES.active:
        EPISODES.toggle()
        return
    target = not any(cam.recording for cam in cameras)
    for cam in cameras:
        cam.set_recording(target)


_VIEW = _CameraViewManager()


class WristCamera:
    def __init__(self, serial=None, width=640, height=480, fps=30, rotate=0,
                 recorder=None, live=True, role="wrist"):
        if rotate not in (0, 90, 180, 270):
            raise ValueError("rotate must be one of 0/90/180/270")
        self.serial = serial
        self.width, self.height, self.fps = width, height, fps
        self.rotate = rotate
        self.recorder = recorder
        self.recorder_settings = recorder.settings if recorder is not None else {"fps": fps}
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
        self.capture_errors = 0
        self._failed_recorder = None
        self._recording_error = None
        self._last_preview_error_mono = float("-inf")

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
        try:
            pipe.start(cfg)
            if self._stop.is_set():
                raise RuntimeError("camera stopped while opening pipeline")
        except BaseException:
            try:
                pipe.stop()
            except Exception:
                pass
            raise
        self.serial = dev.get_info(rs.camera_info.serial_number)
        self._pipe = pipe

    def _close_pipeline(self):
        pipe, self._pipe = self._pipe, None
        if pipe is not None:
            try:
                pipe.stop()
            except Exception as exc:
                self._log_capture_error(f"pipeline stop failed: {exc}")

    @property
    def recording_error(self):
        recorder = self.recorder
        if recorder is None:
            return None
        return getattr(recorder, "error", None) or (
            self._recording_error if recorder is self._failed_recorder else None)

    def _log_capture_error(self, exc):
        now = time.monotonic()
        if now - self._last_error_mono >= 5.:
            print(f"CAMERA ERROR {self.role} [{self.serial}]: {exc} "
                  f"(mono={now:.3f}, capture_frames={self.frames})", flush=True)
            self._last_error_mono = now

    def _capture_loop(self):
        while not self._stop.is_set():
            if self._pipe is None:
                try:
                    self._open_pipeline()
                    self.recoveries += 1
                    self._pipeline_failures = 0
                    print(f"CAMERA RECOVERED {self.role} [{self.serial}]", flush=True)
                except Exception as exc:
                    if self._stop.is_set():
                        break
                    self._log_capture_error(f"pipeline reopen failed: {exc}")
                    self._stop.wait(1.)
                    continue
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
            except Exception as exc:
                if self._stop.is_set():
                    break
                self._pipeline_failures += 1
                self.capture_errors += 1
                self._log_capture_error(exc)
                if self._pipeline_failures < 3:
                    continue
                self._close_pipeline()
                continue
            # Writer failures must not be interpreted as camera/USB failures.
            # Snapshot the attachment: the episode thread can detach it while
            # stopping without racing a second self.recorder dereference.
            recorder = self.recorder
            if recorder is not None and recorder is not self._failed_recorder:
                try:
                    recorder.add_frame(color, depth, wall, mono)
                except Exception as exc:
                    self._failed_recorder = recorder
                    self._recording_error = f"{type(exc).__name__}: {exc}"
                    print(f"CAMERA RECORDING ERROR {self.role}: {self._recording_error}; "
                          "capture continues", flush=True)
            if self.live:
                try:
                    from PIL import Image
                    with self._latest_lock:
                        self._latest = (Image.fromarray(color), wall, mono)
                except Exception as exc:
                    now = time.monotonic()
                    if now - self._last_preview_error_mono > 5.:
                        print(f"CAMERA PREVIEW ERROR {self.role}: {exc}", flush=True)
                        self._last_preview_error_mono = now
        self._close_pipeline()

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
        self._close_pipeline()
        if self.recorder is not None:
            self.recorder.close()


def default_camera_dir(record_path=None, role="wrist"):
    """Next to the --record file when present, else .local/<role>-camera/<ts>."""
    if record_path is not None:
        record_path = Path(record_path)
        return record_path.parent / (record_path.stem + f".{role}")
    return Path.cwd() / ".local" / f"{role}-camera-{time.strftime('%Y%m%d-%H%M%S')}"
