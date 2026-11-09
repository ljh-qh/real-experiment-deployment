from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
import time

from omegaconf import DictConfig

from unirobot_infra.core.time import now_ns
from unirobot_infra.core.types import DeviceFrame, JointState, RobotAction, RobotState
from unirobot_infra.envs.layout import ActionSegment, RobotActionLayout
from unirobot_infra.envs.spaces import BoxSpace
from unirobot_infra.envs.target_action import target_flat_action_from_targets


@dataclass
class TianjiSharpaMujocoEnv:
    action_layout: RobotActionLayout
    cfg: DictConfig
    connected: bool = False
    action_space: BoxSpace = field(init=False)
    observation_space: BoxSpace = field(init=False)
    model: object | None = None
    data: object | None = None
    _mujoco: object | None = None
    _viewer: object | None = None
    _qpos_addr_by_segment: dict[tuple[str, str], tuple[int, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.action_space = self.action_layout.action_space()
        self.observation_space = self.action_layout.observation_space()

    def reset(self) -> tuple[tuple[float, ...], dict]:
        self.connect()
        initial_actions = self.initial_state_actions()
        self._apply_actions(initial_actions)
        sent_actions = initial_actions if bool(self.cfg.reset.move_to_initial_state) else ()
        states = self._read_states()
        observation = self.action_layout.observation_from_states(states)
        return observation, _env_info(
            robot_states=states,
            target_actions=sent_actions,
            sent_actions=sent_actions,
            device_frames=(),
        )

    def connect(self) -> None:
        if self.connected:
            return
        self._load_mujoco()
        self.connected = True

    def initial_state_actions(self) -> tuple[RobotAction, ...]:
        target_action = target_flat_action_from_targets(self.cfg.reset.targets, self.action_layout.segments)
        return self.action_layout.actions_from_flat(target_action, timestamp_ns=now_ns())

    def move_to_actions_until_reached(
        self,
        actions: tuple[RobotAction, ...],
        log_label: str,
    ) -> tuple[RobotState, ...]:
        self._ensure_connected()
        self._apply_actions(actions)
        return self._read_states()

    def wait_until_actions_reached(
        self,
        actions: tuple[RobotAction, ...],
        log_label: str,
    ) -> tuple[RobotState, ...]:
        self._ensure_connected()
        self._apply_actions(actions)
        return self._read_states()

    def step(self, action: Sequence[float]) -> tuple[tuple[float, ...], float, bool, bool, dict]:
        self._ensure_connected()
        started = time.monotonic()
        actions = self.action_layout.actions_from_flat(action, timestamp_ns=now_ns())
        self._apply_actions(actions)
        states = self._read_states()
        finished = time.monotonic()
        observation = self.action_layout.observation_from_states(states)
        info = _env_info(
            robot_states=states,
            target_actions=actions,
            sent_actions=actions,
            device_frames=(),
        )
        info["timing"] = {
            "mujoco_step_s": finished - started,
        }
        return observation, 0.0, False, False, info

    def close(self) -> None:
        self._close_viewer()
        self.connected = False
        self.model = None
        self.data = None
        self._mujoco = None
        self._qpos_addr_by_segment = {}

    def _load_mujoco(self) -> None:
        if str(self.cfg.mujoco.apply_mode) != "qpos":
            raise ValueError("env.mujoco.apply_mode must be qpos")
        import mujoco
        import mujoco.viewer
        if float(self.cfg.mujoco.viewer.sync_hz) <= 0.0:
            raise ValueError("env.mujoco.viewer.sync_hz must be positive")
        if float(self.cfg.mujoco.viewer.keep_open_after_close_s) < 0.0:
            raise ValueError("env.mujoco.viewer.keep_open_after_close_s must be non-negative")

        model_path = Path(str(self.cfg.mujoco.model_path))
        if not model_path.exists():
            raise FileNotFoundError(f"MuJoCo model_path does not exist: {model_path}")
        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self._qpos_addr_by_segment = {
            (segment.side, segment.role): self._segment_qpos_addrs(segment)
            for segment in self.action_layout.segments
        }
        mujoco.mj_forward(self.model, self.data)
        self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._sync_viewer()

    def _segment_qpos_addrs(self, segment: ActionSegment) -> tuple[int, ...]:
        if self._mujoco is None or self.model is None:
            raise RuntimeError("MuJoCo model is not loaded")
        joint_names = tuple(str(name) for name in self.cfg.mujoco.joint_names[segment.side][segment.role])
        if len(joint_names) != segment.size:
            raise ValueError(
                f"env.mujoco.joint_names.{segment.side}.{segment.role} length does not match action segment"
            )
        addrs = []
        for joint_name in joint_names:
            joint_id = int(self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, joint_name))
            if joint_id < 0:
                raise ValueError(f"MuJoCo joint not found: {joint_name}")
            addrs.append(int(self.model.jnt_qposadr[joint_id]))
        return tuple(addrs)

    def _apply_actions(self, actions: tuple[RobotAction, ...]) -> None:
        self._ensure_connected()
        if self._mujoco is None or self.model is None or self.data is None:
            raise RuntimeError("MuJoCo model is not loaded")
        for action in actions:
            if action.arm_q_rad is not None:
                self._write_segment_qpos(action.side, "arm", action.arm_q_rad)
            if action.end_effector_q_rad is not None:
                self._write_segment_qpos(action.side, "end_effector", action.end_effector_q_rad)
        self._mujoco.mj_forward(self.model, self.data)
        self._sync_viewer()

    def _write_segment_qpos(self, side: str, role: str, values: Sequence[float]) -> None:
        if self.data is None:
            raise RuntimeError("MuJoCo data is not loaded")
        addrs = self._qpos_addr_by_segment[(side, role)]
        if len(values) != len(addrs):
            raise ValueError(f"{side}.{role} action length does not match MuJoCo qpos mapping")
        for addr, value in zip(addrs, values, strict=True):
            self.data.qpos[addr] = float(value)

    def _read_states(self) -> tuple[RobotState, ...]:
        if self.data is None:
            raise RuntimeError("MuJoCo data is not loaded")
        timestamp_ns = now_ns()
        states = []
        for side in self.action_layout.sides:
            arm_joint = None
            end_effector_joint = None
            if self.action_layout.has_segment(side, "arm"):
                arm_joint = self._read_segment_joint(side, "arm", timestamp_ns)
            if self.action_layout.has_segment(side, "end_effector"):
                end_effector_joint = self._read_segment_joint(side, "end_effector", timestamp_ns)
            states.append(
                RobotState(
                    side=side,
                    timestamp_ns=timestamp_ns,
                    arm_joint=arm_joint,
                    end_effector_joint=end_effector_joint,
                )
            )
        return tuple(states)

    def _read_segment_joint(self, side: str, role: str, timestamp_ns: int) -> JointState:
        if self.data is None:
            raise RuntimeError("MuJoCo data is not loaded")
        addrs = self._qpos_addr_by_segment[(side, role)]
        return JointState(
            name=f"{side}_{role}",
            q_rad=tuple(float(self.data.qpos[addr]) for addr in addrs),
            dq_rad_s=(0.0,) * len(addrs),
            timestamp_ns=timestamp_ns,
        )

    def _ensure_connected(self) -> None:
        if not self.connected:
            raise RuntimeError("TianjiSharpaMujocoEnv is not connected")

    def _sync_viewer(self) -> None:
        if self._viewer is None:
            raise RuntimeError("MuJoCo viewer is not open")
        if not bool(self._viewer.is_running()):
            raise RuntimeError("MuJoCo viewer was closed")
        self._viewer.sync()

    def _close_viewer(self) -> None:
        if self._viewer is None:
            return
        self._keep_viewer_open_before_close()
        self._viewer.close()
        self._viewer = None

    def _keep_viewer_open_before_close(self) -> None:
        if self._viewer is None:
            return
        hold_s = float(self.cfg.mujoco.viewer.keep_open_after_close_s)
        if hold_s <= 0.0:
            return
        period_s = 1.0 / float(self.cfg.mujoco.viewer.sync_hz)
        deadline_s = time.monotonic() + hold_s
        while time.monotonic() < deadline_s and bool(self._viewer.is_running()):
            self._viewer.sync()
            remaining_s = deadline_s - time.monotonic()
            time.sleep(min(period_s, max(0.0, remaining_s)))


def _env_info(
    robot_states: tuple[RobotState, ...],
    target_actions: tuple[RobotAction, ...],
    sent_actions: tuple[RobotAction, ...],
    device_frames: tuple[DeviceFrame, ...],
) -> dict:
    return {
        "robot_states": robot_states,
        "target_actions": target_actions,
        "sent_actions": sent_actions,
        "device_frames": device_frames,
    }
