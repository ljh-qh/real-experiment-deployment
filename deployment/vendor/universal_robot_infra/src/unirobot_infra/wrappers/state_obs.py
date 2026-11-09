from collections.abc import Sequence

from omegaconf import DictConfig

from unirobot_infra.envs.layout import ObservationItem, SUPPORTED_OBSERVATION_STATES
from unirobot_infra.wrappers.base import EnvLike, EnvWrapper


class StateObservationWrapper(EnvWrapper):
    def __init__(self, env: EnvLike, cfg: DictConfig) -> None:
        super().__init__(env)
        self.state_items = _configured_observation_items(cfg, self.action_layout)
        self.observation_space = self.action_layout.observation_space_from_items(self.state_items)

    def reset(self) -> tuple[tuple[float, ...], dict]:
        _, info = self.env.reset()
        return self._state_observation(info), info

    def step(self, action: Sequence[float]) -> tuple[tuple[float, ...], float, bool, bool, dict]:
        _, reward, terminated, truncated, info = self.env.step(action)
        return self._state_observation(info), reward, terminated, truncated, info

    def _state_observation(self, info: dict) -> tuple[float, ...]:
        return self.action_layout.observation_from_state_items(info["robot_states"], self.state_items)


def _configured_observation_items(cfg: DictConfig, action_layout) -> tuple[ObservationItem, ...]:
    tokens = tuple(str(state_token) for state_token in cfg.states)
    if not tokens:
        raise ValueError("wrappers.state_obs.states must not be empty")
    missing_segments = str(cfg.missing_segments)
    if missing_segments not in {"skip", "error"}:
        raise ValueError("wrappers.state_obs.missing_segments must be skip or error")
    seen = set()
    items = []
    for token in tokens:
        if token in seen:
            raise ValueError(f"duplicate state observation: {token}")
        item = _parse_observation_token(token)
        if action_layout.has_segment(item.side, item.role):
            items.append(item)
        elif missing_segments == "error":
            raise ValueError(f"robot config missing segment {item.side}_{item.role}")
        seen.add(token)
    if not items:
        raise ValueError("wrappers.state_obs.states did not match any robot segment")
    return tuple(items)


def _parse_observation_token(token: str) -> ObservationItem:
    for side in ("left", "right"):
        for role in ("end_effector", "arm"):
            prefix = f"{side}_{role}_"
            if token.startswith(prefix):
                state_name = token[len(prefix) :]
                if state_name not in SUPPORTED_OBSERVATION_STATES:
                    raise ValueError(f"unsupported state observation: {state_name}")
                return ObservationItem(side=side, role=role, state_name=state_name, token=token)
    raise ValueError(f"invalid state observation token: {token}")
