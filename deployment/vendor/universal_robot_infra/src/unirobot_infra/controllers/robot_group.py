from dataclasses import dataclass

from unirobot_infra.controllers.arm_joint import ArmJointController
from unirobot_infra.controllers.hand_joint import HandJointController
from unirobot_infra.core.types import RobotAction, RobotState, Side
from unirobot_infra.robots.arm.tianji import TianjiArm
from unirobot_infra.robots.compositional.group import RobotGroup
from unirobot_infra.robots.compositional.unit import RobotUnit


@dataclass
class RobotUnitJointController:
    side: Side
    arm: ArmJointController | None = None
    hand: HandJointController | None = None

    @classmethod
    def from_unit(cls, unit: RobotUnit) -> "RobotUnitJointController":
        arm = ArmJointController(side=unit.side, arm=unit.arm) if unit.arm is not None else None
        hand = HandJointController(side=unit.side, hand=unit.end_effector) if unit.end_effector is not None else None
        return cls(side=unit.side, arm=arm, hand=hand)

    def connect(self) -> None:
        if self.arm is not None:
            self.arm.connect()
        if self.hand is not None:
            self.hand.connect()

    def read_state(self) -> RobotState:
        tianji = self.arm.read_state() if self.arm is not None else None
        sharpa = self.hand.read_state() if self.hand is not None else None
        if tianji is None and sharpa is None:
            raise RuntimeError("RobotUnitJointController has no hardware to read")
        timestamp_ns = tianji.timestamp_ns if tianji is not None else sharpa.timestamp_ns
        return RobotState(
            side=self.side,
            timestamp_ns=timestamp_ns,
            tianji=tianji,
            sharpa=sharpa,
        )

    def send_action(self, action: RobotAction, blocking: bool = True) -> None:
        if action.side != self.side:
            raise ValueError("action side does not match RobotUnitJointController")
        if action.arm_q_rad is not None:
            if self.arm is None:
                raise RuntimeError("RobotUnitJointController has no arm")
            self.arm.send(action.arm_q_rad, blocking=blocking)
        if action.end_effector_q_rad is not None:
            if self.hand is None:
                raise RuntimeError("RobotUnitJointController has no end_effector")
            self.hand.send(action.end_effector_q_rad)

    def close(self) -> None:
        if self.arm is not None:
            self.arm.close()
        if self.hand is not None:
            self.hand.close()


@dataclass
class RobotGroupController:
    name: str
    units: tuple[RobotUnitJointController, ...]

    @classmethod
    def from_group(cls, group: RobotGroup) -> "RobotGroupController":
        return cls(
            name=group.name,
            units=tuple(RobotUnitJointController.from_unit(unit) for unit in group.units),
        )

    def connect(self) -> None:
        tianji_arms_by_robot: dict[int, dict] = {}
        for unit in self.units:
            if unit.arm is not None:
                if isinstance(unit.arm.arm, TianjiArm):
                    robot_key = id(unit.arm.arm.robot)
                    if robot_key not in tianji_arms_by_robot:
                        tianji_arms_by_robot[robot_key] = {
                            "robot": unit.arm.arm.robot,
                            "arm_ids": [],
                        }
                    tianji_arms_by_robot[robot_key]["arm_ids"].append(unit.arm.arm.arm_id)
                else:
                    unit.arm.connect()
        for robot_key in tianji_arms_by_robot:
            batched = tianji_arms_by_robot[robot_key]
            if not batched["robot"].connected:
                batched["robot"].connect()
            batched["robot"].enter_position_modes(tuple(batched["arm_ids"]))
        for unit in self.units:
            if unit.hand is not None:
                unit.hand.connect()

    def read_state(self) -> tuple[RobotState, ...]:
        return tuple(unit.read_state() for unit in self.units)

    def send_actions(self, actions: tuple[RobotAction, ...], blocking: bool = True) -> None:
        unit_by_side = {unit.side: unit for unit in self.units}
        tianji_commands_by_robot: dict[int, dict] = {}
        for action in actions:
            unit = unit_by_side[action.side]
            if action.arm_q_rad is not None:
                if unit.arm is None:
                    raise RuntimeError("RobotUnitJointController has no arm")
                if isinstance(unit.arm.arm, TianjiArm):
                    robot_key = id(unit.arm.arm.robot)
                    if robot_key not in tianji_commands_by_robot:
                        tianji_commands_by_robot[robot_key] = {
                            "robot": unit.arm.arm.robot,
                            "commands_by_arm": {},
                        }
                    tianji_commands_by_robot[robot_key]["commands_by_arm"][unit.arm.arm.arm_id] = tuple(
                        float(value) for value in action.arm_q_rad
                    )
                else:
                    unit.arm.send(action.arm_q_rad, blocking=blocking)
            if action.end_effector_q_rad is not None:
                if unit.hand is None:
                    raise RuntimeError("RobotUnitJointController has no end_effector")
                unit.hand.send(action.end_effector_q_rad)
        for robot_key in tianji_commands_by_robot:
            batched = tianji_commands_by_robot[robot_key]
            batched["robot"].send_joint_positions_rad(batched["commands_by_arm"], blocking=blocking)

    def close(self) -> None:
        for unit in self.units:
            unit.close()
