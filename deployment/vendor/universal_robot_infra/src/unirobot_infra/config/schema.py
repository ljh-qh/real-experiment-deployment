from dataclasses import dataclass
from typing import Literal


SideName = Literal["left", "right"]


@dataclass(frozen=True)
class RobotPartConfig:
    role: str
    name: str


@dataclass(frozen=True)
class RobotSideConfig:
    parts: tuple[str, ...]


@dataclass(frozen=True)
class CompositionalRobotConfig:
    name: str
    sides: dict[SideName, RobotSideConfig]
