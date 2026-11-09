from typing import Protocol

from unirobot_infra.core.types import SharpaHandState, SharpaTactileFrame


class EndEffector(Protocol):
    name: str
    dof: int

    def connect(self) -> None: ...

    def read_state(self) -> SharpaHandState: ...

    def read_tactile_frame(self, channel: int, timeout_s: float) -> SharpaTactileFrame | None: ...

    def send_joint_position_rad(self, q_rad: tuple[float, ...]) -> None: ...

    def close(self) -> None: ...
