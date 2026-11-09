from collections.abc import Sequence

from omegaconf import DictConfig

from unirobot_infra.core.types import RobotAction, RobotState
from unirobot_infra.visualization.rerun_robot_scene import RerunRobotSceneLogger
from unirobot_infra.wrappers.base import EnvLike, EnvWrapper


class RerunVisualizationWrapper(EnvWrapper):
    def __init__(self, env: EnvLike, cfg: DictConfig) -> None:
        super().__init__(env)
        self.cfg = cfg
        self.frame_index = 0
        if bool(cfg.enabled):
            import rerun as rr

            self.rr = rr
            rr.init(str(cfg.app_id), spawn=bool(cfg.spawn))
            self.robot_scene_logger = RerunRobotSceneLogger(rr, cfg.scene) if bool(cfg.scene.enabled) else None
            if bool(cfg.timeseries.enabled):
                self._log_action_series_styles()
                self._log_state_series_styles()
        else:
            self.rr = None
            self.robot_scene_logger = None

    def reset(self) -> tuple[tuple[float, ...], dict]:
        observation, info = self.env.reset()
        self.frame_index = 0
        if bool(self.cfg.enabled):
            self._log_frame(observation, info, ())
        return observation, info

    def step(self, action: Sequence[float]) -> tuple[tuple[float, ...], float, bool, bool, dict]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        if bool(self.cfg.enabled):
            self.frame_index += 1
            self._log_frame(observation, info, tuple(float(value) for value in action))
        return observation, reward, terminated, truncated, info

    def _log_frame(self, observation: Sequence[float], info: dict, action: tuple[float, ...]) -> None:
        if self.rr is None:
            raise RuntimeError("Rerun is not initialized")
        self.rr.set_time("frame_index", sequence=self.frame_index)
        robot_states = tuple(info["robot_states"])
        if self.robot_scene_logger is not None:
            self.robot_scene_logger.log_robot_states(robot_states)
        if bool(self.cfg.timeseries.enabled):
            self._log_vector("observation/state", tuple(float(value) for value in observation))
            if action:
                self._log_vector("policy_action", action)
            self._log_robot_states(robot_states)
            self._log_robot_actions("sent_actions", tuple(info["sent_actions"]))

    def _log_robot_states(self, states: tuple[RobotState, ...]) -> None:
        for state in states:
            if state.arm_joint is not None:
                self._log_vector(f"robot_state/{state.side}/arm", state.arm_joint.q_rad)
            if state.end_effector_joint is not None:
                self._log_vector(f"robot_state/{state.side}/end_effector", state.end_effector_joint.q_rad)

    def _log_robot_actions(self, entity: str, actions: tuple[RobotAction, ...]) -> None:
        for action in actions:
            if action.arm_q_rad is not None:
                self._log_vector(f"{entity}/{action.side}/arm", action.arm_q_rad)
            if action.end_effector_q_rad is not None:
                self._log_vector(f"{entity}/{action.side}/end_effector", action.end_effector_q_rad)

    def _log_vector(self, entity: str, values: Sequence[float]) -> None:
        if self.rr is None:
            raise RuntimeError("Rerun is not initialized")
        for index, value in enumerate(values):
            self.rr.log(f"{entity}/{index}", self.rr.Scalars(float(value)))

    def _log_action_series_styles(self) -> None:
        if self.rr is None:
            raise RuntimeError("Rerun is not initialized")
        labels = tuple(name for segment in self.action_layout.segments for name in segment.joint_names)
        for index, label in enumerate(labels):
            self.rr.log(
                f"policy_action/{index}",
                self.rr.SeriesLines(names=label, widths=2.0),
                static=True,
            )

    def _log_state_series_styles(self) -> None:
        if self.rr is None:
            raise RuntimeError("Rerun is not initialized")
        for index in range(len(self.observation_space.low)):
            self.rr.log(
                f"observation/state/{index}",
                self.rr.SeriesLines(names=f"obs_{index}", widths=2.0),
                static=True,
            )
