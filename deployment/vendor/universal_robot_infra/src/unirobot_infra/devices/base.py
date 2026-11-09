from typing import Protocol

from unirobot_infra.core.types import DeviceFrame, Pose


class Device(Protocol):
    name: str

    def connect(self) -> None:
        ...

    def read_frame(self) -> DeviceFrame:
        ...

    def close(self) -> None:
        ...


class PoseSource(Device, Protocol):
    def read_pose(self, name: str) -> Pose:
        ...


class HandKeypointSource(Device, Protocol):
    def read_keypoints(self) -> DeviceFrame:
        ...
