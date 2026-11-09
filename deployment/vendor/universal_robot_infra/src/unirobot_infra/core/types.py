from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Side = Literal["left", "right"]
RobotPart = Literal["arm", "end_effector"]


@dataclass(frozen=True)
class Pose:
    xyz_m: tuple[float, float, float]
    quat_xyzw: tuple[float, float, float, float]
    frame: str
    timestamp_ns: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JointState:
    name: str
    q_rad: tuple[float, ...]
    dq_rad_s: tuple[float, ...]
    timestamp_ns: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TianjiArmState:
    name: str
    timestamp_ns: int
    cur_state: int
    cmd_state: int
    err_code: int
    frame_serial: int
    traj_state: int
    tip_di: int
    low_speed_flag: int
    joint_pos: tuple[float, ...]
    joint_vel: tuple[float, ...]
    joint_pos_external: tuple[float, ...]
    joint_pos_commanded: tuple[float, ...]
    joint_current: tuple[float, ...]
    joint_torque: tuple[float, ...]
    joint_temperature: tuple[float, ...]
    est_joint_friction: tuple[float, ...]
    est_joint_friction_derivative: tuple[float, ...]
    est_joint_force: tuple[float, ...]
    est_cart_force: tuple[float, ...]
    rt_in_switch: int
    imp_type: int
    in_frame_serial: int
    frame_miss_cnt: int
    max_frame_miss_cnt: int
    sys_cyc: int
    sys_cyc_miss_cnt: int
    max_sys_cyc_miss_cnt: int
    joint_cmd_pos: tuple[float, ...]
    joint_vel_ratio: int
    joint_acc_ratio: int
    joint_k: tuple[float, ...]
    joint_d: tuple[float, ...]
    drag_sp_type: int
    drag_sp_para: tuple[float, ...]
    cart_kd_type: int
    cart_k: tuple[float, ...]
    cart_d: tuple[float, ...]
    cart_kn: float
    cart_dn: float
    force_fb_type: int
    force_type: int
    force_dir: tuple[float, ...]
    force_pidul: tuple[float, ...]
    force_adj_lmt: float
    force_cmd: float
    set_tags: tuple[int, ...]
    update_tags: tuple[int, ...]
    pvt_id: int
    pvt_id_update: int
    pvt_run_id: int
    pvt_run_state: int

    @property
    def joint(self) -> JointState:
        return JointState(
            name=self.name,
            q_rad=self.joint_pos,
            dq_rad_s=self.joint_vel,
            timestamp_ns=self.timestamp_ns,
        )

    def observation_values(self, state_name: str) -> tuple[float, ...]:
        if state_name == "joint_pos":
            return self.joint_pos
        if state_name == "joint_vel":
            return self.joint_vel
        if state_name == "joint_torque":
            return self.joint_torque
        if state_name == "joint_pos_external":
            return self.joint_pos_external
        if state_name == "joint_pos_commanded":
            return self.joint_pos_commanded
        if state_name == "joint_current":
            return self.joint_current
        if state_name == "joint_temperature":
            return self.joint_temperature
        if state_name == "est_joint_friction":
            return self.est_joint_friction
        if state_name == "est_joint_friction_derivative":
            return self.est_joint_friction_derivative
        if state_name == "est_joint_force":
            return self.est_joint_force
        raise ValueError(f"Tianji state does not support observation {state_name}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SharpaTactileFrame:
    side: Side
    finger: str
    channel: int
    timestamp_ns: int
    sdk_timestamp_s: float
    frame_id: int
    raw_image: Any | None = None
    deform_image: Any | None = None
    f6: tuple[float, ...] = ()
    f6_shape: tuple[int, ...] = ()
    contact_point: tuple[float, ...] = ()
    contact_point_shape: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SharpaHandState:
    name: str
    timestamp_ns: int
    sdk_timestamp: int
    sequence: int
    joint_pos: tuple[float, ...]
    joint_vel: tuple[float, ...]
    joint_torque: tuple[float, ...]

    @property
    def joint(self) -> JointState:
        return JointState(
            name=self.name,
            q_rad=self.joint_pos,
            dq_rad_s=self.joint_vel,
            timestamp_ns=self.timestamp_ns,
        )

    def observation_values(self, state_name: str) -> tuple[float, ...]:
        if state_name == "joint_pos":
            return self.joint_pos
        if state_name == "joint_vel":
            return self.joint_vel
        if state_name == "joint_torque":
            return self.joint_torque
        raise ValueError(f"Sharpa state does not support observation {state_name}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeviceFrame:
    device_name: str
    side: Side
    timestamp_ns: int
    poses: dict[str, Pose] = field(default_factory=dict)
    scalars: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RobotState:
    side: Side
    timestamp_ns: int
    arm_joint: JointState | None = None
    end_effector_joint: JointState | None = None
    ee_pose: Pose | None = None
    tianji: TianjiArmState | None = None
    sharpa: SharpaHandState | None = None

    def __post_init__(self) -> None:
        if self.arm_joint is None and self.tianji is not None:
            object.__setattr__(self, "arm_joint", self.tianji.joint)
        if self.end_effector_joint is None and self.sharpa is not None:
            object.__setattr__(self, "end_effector_joint", self.sharpa.joint)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RobotAction:
    side: Side
    timestamp_ns: int
    arm_q_rad: tuple[float, ...] | None = None
    end_effector_q_rad: tuple[float, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Snapshot:
    timestamp_ns: int
    device_frames: tuple[DeviceFrame, ...] = ()
    target_actions: tuple[RobotAction, ...] = ()
    sent_actions: tuple[RobotAction, ...] = ()
    robot_states: tuple[RobotState, ...] = ()
    safety: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
