from dataclasses import dataclass

from unirobot_infra.core.types import Side, TianjiArmState
from unirobot_infra.robots.arm.base import Arm


@dataclass
class ArmJointController:
    side: Side
    arm: Arm

    def connect(self) -> None:
        self.arm.connect()

    def read_state(self) -> TianjiArmState:
        return self.arm.read_state()

    def send(self, q_rad: tuple[float, ...], blocking: bool = True) -> None:
        if len(q_rad) != self.arm.dof:
            raise ValueError("arm q_rad length does not match dof")
        self.arm.send_joint_position_rad(tuple(float(value) for value in q_rad), blocking=blocking)

    def close(self) -> None:
        self.arm.close()
