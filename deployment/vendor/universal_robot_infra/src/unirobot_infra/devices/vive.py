from dataclasses import dataclass
import json
import time
from typing import Any

from omegaconf import DictConfig

from unirobot_infra.core.types import DeviceFrame, Pose


@dataclass(frozen=True)
class ViveTrackerPose:
    tracker_id: str
    pose_valid: bool
    position_m: tuple[float, float, float] | None
    orientation_xyzw: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class ViveTrackerFrame:
    topic: str
    timestamp_ns: int
    received_timestamp_ns: int
    frame: str
    trackers: tuple[ViveTrackerPose, ...]

    @property
    def latency_ms(self) -> float:
        return (self.received_timestamp_ns - self.timestamp_ns) / 1_000_000.0

    @property
    def valid_trackers(self) -> tuple[ViveTrackerPose, ...]:
        return tuple(tracker for tracker in self.trackers if tracker.pose_valid)

    def to_device_frame(self, side: str = "right") -> DeviceFrame:
        poses = {
            tracker.tracker_id: Pose(
                xyz_m=tracker.position_m,
                quat_xyzw=tracker.orientation_xyzw,
                frame=self.frame,
                timestamp_ns=self.timestamp_ns,
            )
            for tracker in self.valid_trackers
            if tracker.position_m is not None and tracker.orientation_xyzw is not None
        }
        return DeviceFrame(
            device_name="vive",
            side=side,
            timestamp_ns=self.timestamp_ns,
            poses=poses,
            scalars={
                "latency_ms": self.latency_ms,
                "tracker_count": float(len(self.trackers)),
                "valid_tracker_count": float(len(self.valid_trackers)),
            },
        )


@dataclass
class ViveTrackerDevice:
    cfg: DictConfig
    connected: bool = False
    _ctx: Any = None
    _socket: Any = None

    def connect(self) -> None:
        import zmq

        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.BINDTODEVICE, self.cfg.network_interface)
        self._socket.setsockopt(zmq.RCVHWM, int(self.cfg.receive_high_water_mark))
        self._socket.setsockopt(zmq.CONFLATE, int(self.cfg.conflate))
        self._socket.setsockopt_string(zmq.SUBSCRIBE, self.cfg.topic)
        self._socket.connect(self.cfg.endpoint)
        self.connected = True

    def recv_latest(self, timeout_s: float) -> ViveTrackerFrame | None:
        self._ensure_connected()
        import zmq

        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            try:
                message = self._socket.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                if time.monotonic() >= deadline:
                    return None
                time.sleep(float(self.cfg.poll_sleep_s))
                continue
            return vive_tracker_message_to_frame(
                message=message,
                expected_topic=str(self.cfg.topic),
                received_timestamp_ns=time.time_ns(),
            )

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = None
        self.connected = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _ensure_connected(self) -> None:
        if not self.connected or self._socket is None:
            raise RuntimeError("ViveTrackerDevice is not connected")


def vive_tracker_message_to_frame(
    message: str,
    expected_topic: str,
    received_timestamp_ns: int,
) -> ViveTrackerFrame:
    topic, raw_payload = _split_topic_payload(message)
    if expected_topic and topic != expected_topic:
        raise ValueError("Vive tracker topic mismatch")
    payload = json.loads(raw_payload)
    if payload["type"] != "vive_trackers_pose":
        raise ValueError("unsupported Vive tracker payload type")
    return ViveTrackerFrame(
        topic=topic,
        timestamp_ns=int(payload["timestamp_ns"]),
        received_timestamp_ns=int(received_timestamp_ns),
        frame=str(payload["frame"]),
        trackers=tuple(_tracker_pose_from_payload(tracker) for tracker in payload["trackers"]),
    )


def _split_topic_payload(message: str) -> tuple[str, str]:
    parts = message.split(" ", 1)
    if len(parts) != 2:
        raise ValueError("Vive tracker ZMQ message must be '<topic> <json>'")
    return parts[0], parts[1]


def _tracker_pose_from_payload(payload: dict[str, Any]) -> ViveTrackerPose:
    pose_valid = bool(payload["pose_valid"])
    position_m = tuple(float(value) for value in payload["position_m"]) if pose_valid else None
    orientation_xyzw = tuple(float(value) for value in payload["orientation_xyzw"]) if pose_valid else None
    if position_m is not None and len(position_m) != 3:
        raise ValueError("Vive tracker position_m must have 3 values")
    if orientation_xyzw is not None and len(orientation_xyzw) != 4:
        raise ValueError("Vive tracker orientation_xyzw must have 4 values")
    return ViveTrackerPose(
        tracker_id=str(payload["id"]),
        pose_valid=pose_valid,
        position_m=position_m,
        orientation_xyzw=orientation_xyzw,
    )
