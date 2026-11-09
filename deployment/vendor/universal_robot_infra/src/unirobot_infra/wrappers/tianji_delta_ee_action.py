from collections.abc import Sequence

from omegaconf import DictConfig

from unirobot_infra.controllers.delta_ee import DeltaEeActionExpander
from unirobot_infra.core.types import RobotState
from unirobot_infra.envs.layout import RobotActionLayout
from unirobot_infra.wrappers.base import EnvLike, EnvWrapper


class TianjiDeltaEeActionWrapper(EnvWrapper):
    def __init__(self, env: EnvLike, cfg: DictConfig, arm_cfg: DictConfig) -> None:
        super().__init__(env)
        self.cfg = cfg
        self._latest_robot_states: tuple[RobotState, ...] | None = None
        self._base_action_layout = env.action_layout
        self._action_expander: DeltaEeActionExpander | None = None
        if bool(cfg.enabled):
            self._action_expander = DeltaEeActionExpander(
                base_action_layout=self._base_action_layout,
                cfg=cfg,
                arm_cfg=arm_cfg,
            )
            self.action_layout = self._action_expander.action_layout
            self.action_space = self.action_layout.action_space()

    @property
    def base_action_layout(self) -> RobotActionLayout:
        return self._base_action_layout

    @property
    def action_expander(self) -> DeltaEeActionExpander:
        if self._action_expander is None:
            raise RuntimeError("tianji_delta_ee_action is disabled")
        return self._action_expander

    def reset(self) -> tuple[tuple[float, ...], dict]:
        observation, info = self.env.reset()
        if bool(self.cfg.enabled):
            self._latest_robot_states = tuple(info["robot_states"])
        return observation, info

    def step(self, action: Sequence[float]) -> tuple[tuple[float, ...], float, bool, bool, dict]:
        if not bool(self.cfg.enabled):
            return self.env.step(action)
        if self._latest_robot_states is None:
            raise RuntimeError("tianji_delta_ee_action requires reset before step")
        expansion = self.action_expander.expand(action, self._latest_robot_states)
        observation, reward, terminated, truncated, info = self.env.step(expansion.action)
        self._latest_robot_states = tuple(info["robot_states"])
        return observation, reward, terminated, truncated, info
