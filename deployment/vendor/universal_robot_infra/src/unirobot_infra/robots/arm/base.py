from typing import Protocol

from unirobot_infra.core.types import TianjiArmState


class Arm(Protocol):
    name: str
    dof: int

    def connect(self) -> None: ...

    def read_state(self) -> TianjiArmState: ...

    def send_joint_position_rad(self, q_rad: tuple[float, ...], blocking: bool = True) -> None: ...

    def close(self) -> None: ...
