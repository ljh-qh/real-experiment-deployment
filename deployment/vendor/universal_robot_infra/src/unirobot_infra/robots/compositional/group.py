from dataclasses import dataclass

from unirobot_infra.core.types import RobotAction, RobotState
from unirobot_infra.robots.compositional.unit import RobotUnit


@dataclass
class RobotGroup:
    name: str
    units: tuple[RobotUnit, ...]

    def connect(self) -> None:
        for unit in self.units:
            unit.connect()

    def read_state(self) -> tuple[RobotState, ...]:
        return tuple(unit.read_state() for unit in self.units)

    def send_actions(self, actions: tuple[RobotAction, ...]) -> None:
        unit_by_side = {unit.side: unit for unit in self.units}
        for action in actions:
            unit_by_side[action.side].send_action(action)

    def close(self) -> None:
        for unit in self.units:
            unit.close()
