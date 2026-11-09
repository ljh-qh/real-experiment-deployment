from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import socket
import subprocess
import time
from typing import Any
from urllib.parse import urlparse

from omegaconf import DictConfig

from unirobot_infra.core.time import now_ns
from unirobot_infra.core.types import DeviceFrame, Pose, Side
from unirobot_infra.utils.sdk_paths import prepend_sys_paths, sharpa_manus_sdk_paths


MANUS_POSE_WIDTH = 7


@dataclass(frozen=True)
class ManusKeypoints:
    timestamp_ns: int
    left: tuple[tuple[float, ...], ...]
    right: tuple[tuple[float, ...], ...]

    def side_points(self, side: Side) -> tuple[tuple[float, ...], ...]:
        if side == "left":
            return self.left
        if side == "right":
            return self.right
        raise ValueError("side must be left or right")

    def side_has_data(self, side: Side) -> bool:
        return any(value != 0.0 for point in self.side_points(side) for value in point)

    def to_device_frame(self, side: Side, frame: str) -> DeviceFrame:
        poses = {
            f"keypoint_{index:02d}": _point_to_pose(point, frame, self.timestamp_ns)
            for index, point in enumerate(self.side_points(side))
        }
        return DeviceFrame(
            device_name="manus",
            side=side,
            timestamp_ns=self.timestamp_ns,
            poses=poses,
            scalars={"has_data": float(self.side_has_data(side))},
        )


@dataclass
class ManusDevice:
    cfg: DictConfig
    sharpa_cfg: DictConfig
    name: str = "manus"
    connected: bool = False
    _proto: Any = None
    _ctx: Any = None
    _socket: Any = None
    _managed_client: subprocess.Popen | None = None
    _managed_client_log: Any = None
    _pending_keypoints: ManusKeypoints | None = None

    def connect(self) -> None:
        sdk_paths = sharpa_manus_sdk_paths(Path(self.sharpa_cfg.sdk.vendor_root))
        if not sdk_paths.proto_dir.exists():
            raise FileNotFoundError(str(sdk_paths.proto_dir))
        if bool(self.cfg.client.auto_start):
            self._start_client_if_needed(sdk_paths)
        prepend_sys_paths((sdk_paths.proto_dir,))
        self._proto = importlib.import_module("sharpa_hand_pb2")

        import zmq

        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt(zmq.RCVHWM, int(self.cfg.receive_high_water_mark))
        self._socket.setsockopt(zmq.CONFLATE, int(self.cfg.conflate))
        self._socket.setsockopt_string(zmq.SUBSCRIBE, self.cfg.topic)
        self._socket.connect(self.cfg.mocap_endpoint)
        self.connected = True
        self._pending_keypoints = self.recv_latest(timeout_s=float(self.cfg.startup_timeout_s))
        if self._pending_keypoints is None:
            raise TimeoutError(
                f"Manus keypoints timeout from {self.cfg.mocap_endpoint}; "
                f"client log: {self.cfg.client.log_path}"
            )

    def _start_client_if_needed(self, sdk_paths) -> None:
        if _tcp_endpoint_listening(self.cfg.mocap_endpoint):
            return
        client_path = sdk_paths.client_dir / "SharpaManusClient.out"
        client_lib_dir = sdk_paths.client_dir / "ManusSDK" / "lib"
        if not client_path.exists():
            raise FileNotFoundError(str(client_path))
        if not client_lib_dir.exists():
            raise FileNotFoundError(str(client_lib_dir))
        client_working_dir = Path(str(self.cfg.client.working_dir))
        if not client_working_dir.exists():
            raise FileNotFoundError(str(client_working_dir))
        for calibration_name in ("Calibration_left.mcal", "Calibration_right.mcal"):
            calibration_path = client_working_dir / calibration_name
            if not calibration_path.exists():
                raise FileNotFoundError(str(calibration_path))
        log_path = Path(self.cfg.client.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._managed_client_log = log_path.open("w", encoding="utf-8")
        env = dict(os.environ)
        current_ld_library_path = env["LD_LIBRARY_PATH"] if "LD_LIBRARY_PATH" in env else ""
        env["LD_LIBRARY_PATH"] = (
            f"{client_lib_dir}:{current_ld_library_path}" if current_ld_library_path else str(client_lib_dir)
        )
        self._managed_client = subprocess.Popen(
            [str(client_path)],
            cwd=str(client_working_dir),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=self._managed_client_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + float(self.cfg.client.wait_s)
        while not _tcp_endpoint_listening(self.cfg.mocap_endpoint):
            if self._managed_client.poll() is not None:
                raise RuntimeError(f"Manus client exited; log: {log_path}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Manus client did not listen on {self.cfg.mocap_endpoint}; log: {log_path}")
            time.sleep(float(self.cfg.poll_sleep_s))

    def recv_latest(self, timeout_s: float) -> ManusKeypoints | None:
        self._ensure_connected()
        if self._pending_keypoints is not None:
            keypoints = self._pending_keypoints
            self._pending_keypoints = None
            return keypoints
        import zmq

        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            try:
                payload = self._socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                if time.monotonic() >= deadline:
                    return None
                time.sleep(float(self.cfg.poll_sleep_s))
                continue

            message = self._proto.MocapKeypoints()
            message.ParseFromString(payload)
            return mocap_message_to_keypoints(
                message=message,
                timestamp_ns=now_ns(),
                expected_keypoints=int(self.cfg.expected_keypoints),
                allow_empty_side=bool(self.cfg.allow_empty_side),
            )

    def read_keypoints(self) -> ManusKeypoints:
        keypoints = self.recv_latest(timeout_s=float(self.cfg.recv_timeout_s))
        if keypoints is None:
            raise TimeoutError("Manus keypoints timeout")
        return keypoints

    def read_frame(self) -> DeviceFrame:
        return self.read_keypoints().to_device_frame(side=self.cfg.frame_side, frame=self.cfg.frame_name)

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = None
        self._proto = None
        self.connected = False
        if self._managed_client is not None and bool(self.cfg.client.stop_on_close):
            self._managed_client.terminate()
            deadline = time.monotonic() + float(self.cfg.client.stop_timeout_s)
            while self._managed_client.poll() is None and time.monotonic() < deadline:
                time.sleep(float(self.cfg.poll_sleep_s))
            if self._managed_client.poll() is None:
                self._managed_client.kill()
            self._managed_client = None
        if self._managed_client_log is not None:
            self._managed_client_log.close()
            self._managed_client_log = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _ensure_connected(self) -> None:
        if not self.connected or self._socket is None or self._proto is None:
            raise RuntimeError("ManusDevice is not connected")


def _tcp_endpoint_listening(endpoint: str) -> bool:
    parsed = urlparse(endpoint)
    if parsed.scheme != "tcp":
        raise ValueError("Manus endpoint must use tcp://")
    if parsed.hostname is None or parsed.port is None:
        raise ValueError("Manus endpoint must include host and port")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((parsed.hostname, int(parsed.port))) == 0


def mocap_message_to_keypoints(
    message: Any,
    timestamp_ns: int,
    expected_keypoints: int,
    allow_empty_side: bool,
) -> ManusKeypoints:
    return ManusKeypoints(
        timestamp_ns=timestamp_ns,
        left=_poses_to_points(
            poses=message.left_mocap_pose,
            side="left",
            expected_keypoints=expected_keypoints,
            allow_empty_side=allow_empty_side,
        ),
        right=_poses_to_points(
            poses=message.right_mocap_pose,
            side="right",
            expected_keypoints=expected_keypoints,
            allow_empty_side=allow_empty_side,
        ),
    )


def _poses_to_points(
    poses: Any,
    side: Side,
    expected_keypoints: int,
    allow_empty_side: bool,
) -> tuple[tuple[float, ...], ...]:
    if len(poses) == 0 and allow_empty_side:
        return tuple((0.0,) * MANUS_POSE_WIDTH for _ in range(expected_keypoints))
    if len(poses) != expected_keypoints:
        raise RuntimeError(f"Manus {side} keypoint count is not {expected_keypoints}")
    return tuple(_pose_to_point(pose) for pose in poses)


def _pose_to_point(pose: Any) -> tuple[float, ...]:
    return (
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
        float(pose.orientation.w),
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
    )


def _point_to_pose(point: tuple[float, ...], frame: str, timestamp_ns: int) -> Pose:
    if len(point) != MANUS_POSE_WIDTH:
        raise ValueError("Manus keypoint width is not 7")
    return Pose(
        xyz_m=(point[0], point[1], point[2]),
        quat_xyzw=(point[4], point[5], point[6], point[3]),
        frame=frame,
        timestamp_ns=timestamp_ns,
    )
