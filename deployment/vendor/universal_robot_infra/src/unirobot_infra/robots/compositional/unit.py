from dataclasses import dataclass

from unirobot_infra.core.types import RobotAction, RobotState, Side
from unirobot_infra.robots.arm.base import Arm
from unirobot_infra.robots.end_effector.base import EndEffector


@dataclass
class RobotUnit:
    side: Side
    arm: Arm | None = None
    end_effector: EndEffector | None = None

    def connect(self) -> None:
        if self.arm is not None:
            self.arm.connect()
        if self.end_effector is not None:
            self.end_effector.connect()

    def read_state(self) -> RobotState:
        tianji = self.arm.read_state() if self.arm is not None else None
        sharpa = self.end_effector.read_state() if self.end_effector is not None else None
        if tianji is None and sharpa is None:
            raise RuntimeError("RobotUnit has no hardware to read")
        timestamp_ns = tianji.timestamp_ns if tianji is not None else sharpa.timestamp_ns
        return RobotState(
            side=self.side,
            timestamp_ns=timestamp_ns,
            tianji=tianji,
            sharpa=sharpa,
        )

    def send_action(self, action: RobotAction) -> None:
        if action.side != self.side:
            raise ValueError("action side does not match RobotUnit")
        if action.arm_q_rad is not None:
            if self.arm is None:
                raise RuntimeError("RobotUnit has no arm")
            self.arm.send_joint_position_rad(action.arm_q_rad)
        if action.end_effector_q_rad is not None:
            if self.end_effector is None:
                raise RuntimeError("RobotUnit has no end_effector")
            self.end_effector.send_joint_position_rad(action.end_effector_q_rad)

    def close(self) -> None:
        if self.arm is not None:
            self.arm.close()
        if self.end_effector is not None:
            self.end_effector.close()
