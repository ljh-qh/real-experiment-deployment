"""Atomic PICO Body payloads, recording playback and an explicit synthetic demo.

Read the public PXREA JSON callback directly. The installed Pybind getters
expose fixed-size, separately locked arrays and cannot prove Body freshness.
This adapter never initializes robot hardware or the Pybind singleton.

The native library is resolved with the priority ``library argument >
PICO_SDK_DIR environment variable > ./vendor`` and is only loaded by
:meth:`PicoBodySource.start`; demo and replay sources never touch it.
"""

from collections import deque
import ctypes
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation


JOINT_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hand", "right_hand",
)
PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)
ARM_ROLES = {name: JOINT_NAMES.index(name) for name in (
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist")}

_LIBRARY_HELP = (
    "PICO Body SDK library not found. Install the PICO PC Service and the "
    "xrobotoolkit prebuilt SDK (libPXREARobotSDK.so), then set PICO_SDK_DIR "
    "or place the library in ./vendor. Demo/replay sources need no SDK."
)


def default_body_library():
    return Path(os.environ.get("PICO_SDK_DIR") or Path.cwd() / "vendor") / "libPXREARobotSDK.so"


def timestamp(value):
    try:
        return max(0, int(value))
    except (ValueError, TypeError, OverflowError):
        return 0


@dataclass(frozen=True)
class BodyFrame:
    sequence: int
    received_at: float
    source_timestamp_ns: int
    poses: np.ndarray
    valid: np.ndarray
    joint_count: int
    quality: str = "UNKNOWN"
    quality_origin: str = "NOT_PROVIDED"
    raw: dict = field(default_factory=dict)
    source: str = "pico_body"
    error: str = ""

    @property
    def complete(self):
        return self.joint_count == 24 and bool(self.valid.all()) and not self.error

    def record(self):
        return {
            "sequence": self.sequence, "source_timestamp_ns": self.source_timestamp_ns,
            "joint_count": self.joint_count, "poses": self.poses.tolist(),
            "valid": self.valid.tolist(), "quality": self.quality,
            "quality_origin": self.quality_origin, "raw_body": self.raw,
            "source": self.source, "error": self.error,
        }


def parse_body(body, sequence, received_at, source="pico_body"):
    """Missing/malformed joints become invalid, never values from an old frame.

    Native Body payloads currently have no documented validity field. Only
    our explicitly versioned extension is interpreted; arbitrary status codes
    are preserved in raw data and remain UNKNOWN.
    """
    poses = np.zeros((24, 7), dtype=float)
    valid = np.zeros(24, dtype=bool)
    error = ""
    if not isinstance(body, dict):
        body = {}
        error = "Body must be an object"
    joints = body.get("joints", [])
    if not isinstance(joints, list):
        joints = []
        error = "Body.joints must be an array"
    for index, joint in enumerate(joints[:24]):
        try:
            value = joint["p"]
            pose = np.asarray(value.split(",") if isinstance(value, str) else value, dtype=float)
            if pose.shape != (7,) or not np.isfinite(pose).all():
                continue
            norm = np.linalg.norm(pose[3:])
            if not .95 <= norm <= 1.05:
                continue
            poses[index] = pose
            poses[index, 3:] /= norm
            valid[index] = True
        except (TypeError, ValueError, KeyError):
            continue
    if "len" in body and timestamp(body["len"]) != len(joints):
        error = "Body.len does not match Body.joints"
    quality, origin = "UNKNOWN", "NOT_PROVIDED"
    extension = body.get("teleop_tracking", {})
    if isinstance(extension, dict) and extension.get("schema_version") == 1:
        value = extension.get("state")
        if value == "INVALID" or (value in ("TRACKING", "PREDICTED") and extension.get("calibrated") is True):
            quality, origin = value, "EXPLICIT_EXTENSION_V1"
        if extension.get("calibrated") is False:
            quality, origin = "INVALID", "EXPLICIT_EXTENSION_V1"
    poses.setflags(write=False)
    valid.setflags(write=False)
    return BodyFrame(sequence, received_at, timestamp(body.get("timeStampNs")),
                     poses, valid, len(joints), quality, origin, body, source, error)


class _StateJson(ctypes.Structure):
    _fields_ = [("devID", ctypes.c_char * 32), ("stateJson", ctypes.c_char * 16352)]


class PicoBodySource:
    """Keep only the latest coherent Body JSON, independently of Head packets."""

    def __init__(self, library=None):
        self.library = Path(library or default_body_library())
        self._lock = threading.Lock()
        self._frame = None
        self._sequence = 0
        self._body_times = deque(maxlen=240)
        self._diagnostic = {"connected": False, "packet_at": None, "motion_count": 0, "error": ""}
        self._sdk = None
        self._callback_ref = None

    def start(self):
        if self._sdk is not None:
            return
        if not self.library.is_file():
            raise FileNotFoundError(f"{_LIBRARY_HELP} (looked for {self.library})")
        callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int,
                                         ctypes.c_int, ctypes.c_void_p)
        self._callback_ref = callback_type(self._callback)
        try:
            sdk = ctypes.CDLL(str(self.library))
        except OSError as exc:
            raise OSError(f"{_LIBRARY_HELP} (failed to load {self.library}: {exc})") from exc
        sdk.PXREAInit.argtypes = [ctypes.c_void_p, callback_type, ctypes.c_uint]
        sdk.PXREAInit.restype = ctypes.c_int
        sdk.PXREADeinit.argtypes = []
        sdk.PXREADeinit.restype = ctypes.c_int
        result = sdk.PXREAInit(None, self._callback_ref, 0xffffffff)
        if result != 0:
            sdk.PXREADeinit()
            raise RuntimeError(f"PXREAInit failed: {result}")
        self._sdk = sdk

    def _callback(self, _context, event, _status, payload):
        try:
            now = time.monotonic()
            if event in (1 << 2, 1 << 3, 1 << 5):
                with self._lock:
                    self._diagnostic["connected"] = event == 1 << 2
                    if event != 1 << 2:
                        self._frame = None
                        self._body_times.clear()
                return
            if event != 1 << 25 or not payload:
                return
            message = ctypes.cast(payload, ctypes.POINTER(_StateJson)).contents
            wrapper = json.loads(bytes(message.stateJson).decode("utf-8"))
            value = wrapper.get("value", wrapper)
            if isinstance(value, str):
                value = json.loads(value)
            if not isinstance(value, dict):
                return
            motion = value.get("Motion", {})
            count = len(motion.get("joints", [])) if isinstance(motion, dict) else 0
            with self._lock:
                self._diagnostic.update(packet_at=now, motion_count=count, error="")
                if "Body" in value:
                    self._sequence += 1
                    self._frame = parse_body(value["Body"], self._sequence, now)
                    self._body_times.append(now)
        except Exception as exc:
            # Exceptions must never unwind through the native callback.
            with self._lock:
                self._diagnostic["error"] = f"Body receive: {type(exc).__name__}: {exc}"
                self._frame = None

    def snapshot(self):
        with self._lock:
            return self._frame

    def diagnostics(self):
        with self._lock:
            result = dict(self._diagnostic)
            times = tuple(self._body_times)
        result["body_hz"] = ((len(times) - 1) / (times[-1] - times[0])
                             if len(times) > 1 and times[-1] > times[0] and
                             time.monotonic() - times[-1] < .25 else 0.)
        return result

    def close(self):
        if self._sdk is not None:
            self._sdk.PXREADeinit()
            self._sdk = None
        self._callback_ref = None


class ReplayBodySource:
    """Playback explicitly recorded Body payloads at their original receive times."""

    def __init__(self, path):
        self.records = []
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                value = json.loads(line)
                if value.get("type") == "body_frame":
                    self.records.append(value)
        if not self.records:
            raise ValueError("Replay contains no body_frame records")
        times = [float(v["elapsed_s"]) for v in self.records]
        if not np.isfinite(times).all() or any(b < a for a, b in zip(times, times[1:])):
            raise ValueError("Replay timestamps must be finite and nondecreasing")
        self._times = np.asarray(times) - times[0]
        self._index = -1
        self._frame = None
        self.done = False

    def start(self):
        self._started = time.monotonic()

    def snapshot(self):
        elapsed = time.monotonic() - self._started
        while self._index + 1 < len(self.records) and self._times[self._index + 1] <= elapsed:
            self._index += 1
            record = self.records[self._index]
            data = record["frame"]
            self._frame = parse_body(data["raw_body"], self._index + 1,
                                     self._started + self._times[self._index], source="replay")
        self.done = elapsed > self._times[-1] + .3
        return self._frame

    def diagnostics(self):
        return {"source": "replay", "body_hz": 0., "error": ""}

    def close(self):
        pass


class DemoBodySource:
    """Synthetic skeleton for offline use; never eligible for hardware output."""

    def start(self):
        self._started = time.monotonic()
        self._frame = None

    def snapshot(self):
        now = time.monotonic()
        elapsed = now - self._started
        sequence = int(elapsed * 90) + 1
        if self._frame is not None and self._frame.sequence == sequence:
            return self._frame
        # SDK-like right-handed frame: +Y up, -Z forward, -X left.
        p = np.array([[0, .95, 0], [-.10,.95,0], [.10,.95,0], [0,1.05,0],
                      [-.10,.52,0], [.10,.52,0], [0,1.2,0], [-.10,.10,0], [.10,.10,0],
                      [0,1.38,0], [-.10,.06,-.12], [.10,.06,-.12], [0,1.53,0],
                      [-.10,1.46,0], [.10,1.46,0], [0,1.68,0],
                      [-.20,1.45,0], [.20,1.45,0], [-.25,1.13,-.08], [.25,1.13,-.08],
                      [-.25,1.15,-.35], [.25,1.15,-.35], [-.25,1.15,-.43], [.25,1.15,-.43]])
        q = np.tile([0., 0., 0., 1.], (24, 1))
        t = max(0., elapsed - 6.)
        for indices, sign in (((20,22), -1), ((21,23), 1)):
            wrist = indices[0]
            elbow = 18 if wrist == 20 else 19
            rest = p[wrist] - p[elbow]
            turn = Rotation.from_euler("xy", [.28*np.sin(.7*t), .25*sign*np.sin(.5*t)])
            displacement = turn.apply(rest) - rest
            for index in indices:
                p[index] += displacement
                q[index] = Rotation.from_rotvec([0, 0, sign * .8 * np.sin(.4*t)]).as_quat()
        body = {"timeStampNs": sequence * 11111111, "len": 24,
                "joints": [{"p": list(pos) + list(quat)} for pos, quat in zip(p, q)]}
        self._frame = parse_body(body, sequence, now, source="demo")
        return self._frame

    def diagnostics(self):
        return {"source": "demo", "body_hz": 90., "error": ""}

    def close(self):
        pass
