from collections.abc import Sequence
from typing import Protocol

from unirobot_infra.envs.layout import RobotActionLayout
from unirobot_infra.envs.spaces import BoxSpace


class EnvLike(Protocol):
    action_layout: RobotActionLayout
    action_space: BoxSpace
    observation_space: BoxSpace

    def reset(self) -> tuple[tuple[float, ...], dict]: ...

    def step(self, action: Sequence[float]) -> tuple[tuple[float, ...], float, bool, bool, dict]: ...

    def close(self) -> None: ...


class EnvWrapper:
    def __init__(self, env: EnvLike) -> None:
        self.env = env
        self.action_layout = env.action_layout
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def reset(self) -> tuple[tuple[float, ...], dict]:
        return self.env.reset()

    def step(self, action: Sequence[float]) -> tuple[tuple[float, ...], float, bool, bool, dict]:
        return self.env.step(action)

    def close(self) -> None:
        self.env.close()


class ActionWrapper(EnvWrapper):
    def action(self, action: Sequence[float]) -> tuple[float, ...]:
        return tuple(float(value) for value in action)

    def step(self, action: Sequence[float]) -> tuple[tuple[float, ...], float, bool, bool, dict]:
        return self.env.step(self.action(action))
