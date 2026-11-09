from dataclasses import dataclass

from unirobot_infra.core.types import SharpaHandState, Side
from unirobot_infra.robots.end_effector.base import EndEffector


@dataclass
class HandJointController:
    side: Side
    hand: EndEffector

    def connect(self) -> None:
        self.hand.connect()

    def read_state(self) -> SharpaHandState:
        return self.hand.read_state()

    def send(self, q_rad: tuple[float, ...]) -> None:
        if len(q_rad) != self.hand.dof:
            raise ValueError("end effector q_rad length does not match dof")
        self.hand.send_joint_position_rad(tuple(float(value) for value in q_rad))

    def close(self) -> None:
        self.hand.close()
