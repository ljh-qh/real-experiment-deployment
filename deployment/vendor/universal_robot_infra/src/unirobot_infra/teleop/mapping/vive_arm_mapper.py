from collections.abc import Sequence
from dataclasses import dataclass
import math
import numpy as np
from scipy.spatial.transform import Rotation

from unirobot_infra.core.time import now_ns
from unirobot_infra.core.types import DeviceFrame, Pose, RobotAction, RobotState, Side
from unirobot_infra.core.units import deg_to_rad, m_to_mm, rad_to_deg
from unirobot_infra.robots.arm.tianji_tcp_pose import TianjiIkSolver
from unirobot_infra.teleop.mapping.tianji_kinematics import TianjiKinematics


@dataclass(frozen=True)
class TianjiLocalSe3Anchor:
    tracker_mat_m: tuple[tuple[float, float, float, float], ...]
    robot_tcp_mat_mm: tuple[tuple[float, float, float, float], ...]
    reference_q_deg: tuple[float, float, float, float, float, float, float]


class ViveTianjiLocalSe3Mapper:
    def __init__(
        self,
        side: Side,
        pose_name: str,
        mount_rotation_from_tracker_local: Sequence[Sequence[float]],
        position_scale: float,
        orientation_filter_enabled: bool,
        orientation_filter_tau_s: float,
        orientation_filter_fallback_dt_s: float,
        solver: TianjiIkSolver,
        robot_state: RobotState,
    ) -> None:
        if robot_state.arm_joint is None:
            raise ValueError("Vive Tianji local SE(3) mapper requires arm_joint state")
        self.side = side
        self.pose_name = pose_name
        self.mount_rotation_from_tracker_local = _rotation_matrix3x3(
            mount_rotation_from_tracker_local,
            "mount_rotation_from_tracker_local",
        )
        self.position_scale = float(position_scale)
        self.orientation_filter = QuaternionSlerpLowPassFilter(
            enabled=bool(orientation_filter_enabled),
            tau_s=float(orientation_filter_tau_s),
            fallback_dt_s=float(orientation_filter_fallback_dt_s),
        )
        self.solver = solver
        self.last_q_deg: tuple[float, float, float, float, float, float, float] = _rad_tuple_to_deg7(
            robot_state.arm_joint.q_rad
        )
        self.anchor: TianjiLocalSe3Anchor | None = None

    def anchor_from_pose(self, pose: Pose) -> None:
        self.orientation_filter.reset(quat_xyzw=pose.quat_xyzw, timestamp_ns=pose.timestamp_ns)
        tcp_xyzabc = self.solver.kinematics.fk_xyzabc_mm_deg(self.last_q_deg)
        self.anchor = TianjiLocalSe3Anchor(
            tracker_mat_m=_pose_to_mat4x4_m(pose),
            robot_tcp_mat_mm=self.solver.kinematics.xyzabc_to_mat4x4(tcp_xyzabc),
            reference_q_deg=self.last_q_deg,
        )

    def clear_anchor(self) -> None:
        self.anchor = None
        self.orientation_filter.reset()

    def map_frame(self, frame: DeviceFrame) -> tuple[RobotAction, ...]:
        if self.anchor is None:
            raise RuntimeError("Vive Tianji local SE(3) mapper is not anchored")
        pose = frame.poses[self.pose_name]
        target_xyzabc = self.target_xyzabc_mm_deg(pose)
        q_deg = self.solver.solve(
            target_xyzabc_mm_deg=target_xyzabc,
            reference_q_deg=self.anchor.reference_q_deg,
            last_q_deg=self.last_q_deg,
        )
        self.last_q_deg = q_deg
        return (
            RobotAction(
                side=self.side,
                timestamp_ns=now_ns(),
                arm_q_rad=tuple(deg_to_rad(value) for value in q_deg),
            ),
        )

    def target_xyzabc_mm_deg(self, pose: Pose) -> tuple[float, float, float, float, float, float]:
        if self.anchor is None:
            raise RuntimeError("Vive Tianji local SE(3) mapper is not anchored")
        tracker_now = _pose_to_mat4x4_m(self._filtered_orientation_pose(pose))
        delta_tracker = _mat4x4_mul(_invert_rigid_mat4x4(self.anchor.tracker_mat_m), tracker_now)
        delta_tracker_mm = _scale_transform_translation(delta_tracker, m_to_mm(1.0) * self.position_scale)
        mount = _rotation3_to_mat4x4(self.mount_rotation_from_tracker_local)
        delta_tcp = _mat4x4_mul(_mat4x4_mul(mount, delta_tracker_mm), _invert_rigid_mat4x4(mount))
        target_tcp_mat = _mat4x4_mul(self.anchor.robot_tcp_mat_mm, delta_tcp)
        return self.solver.kinematics.mat4x4_to_xyzabc_mm_deg(target_tcp_mat)

    def _filtered_orientation_pose(self, pose: Pose) -> Pose:
        return Pose(
            xyz_m=pose.xyz_m,
            quat_xyzw=self.orientation_filter.update(pose.quat_xyzw, timestamp_ns=pose.timestamp_ns),
            frame=pose.frame,
            timestamp_ns=pose.timestamp_ns,
        )


class QuaternionSlerpLowPassFilter:
    def __init__(self, enabled: bool, tau_s: float, fallback_dt_s: float) -> None:
        self.enabled = bool(enabled)
        self.tau_s = float(tau_s)
        self.fallback_dt_s = float(fallback_dt_s)
        if not math.isfinite(self.tau_s) or self.tau_s <= 0.0:
            raise ValueError("orientation_filter.tau_s must be positive finite")
        if not math.isfinite(self.fallback_dt_s) or self.fallback_dt_s <= 0.0:
            raise ValueError("orientation_filter.fallback_dt_s must be positive finite")
        self.prev_quat_xyzw: tuple[float, float, float, float] | None = None
        self.prev_timestamp_ns: int | None = None

    def reset(self, quat_xyzw: Sequence[float] | None = None, timestamp_ns: int | None = None) -> None:
        self.prev_quat_xyzw = None
        self.prev_timestamp_ns = None
        if quat_xyzw is not None:
            self.prev_quat_xyzw = _normalize_quat_xyzw(quat_xyzw)
        if timestamp_ns is not None:
            self.prev_timestamp_ns = int(timestamp_ns)

    def update(self, quat_xyzw: Sequence[float], timestamp_ns: int | None) -> tuple[float, float, float, float]:
        raw_quat = _normalize_quat_xyzw(quat_xyzw)
        if not self.enabled:
            self.prev_quat_xyzw = raw_quat
            if timestamp_ns is not None:
                self.prev_timestamp_ns = int(timestamp_ns)
            return raw_quat

        if self.prev_quat_xyzw is None:
            self.prev_quat_xyzw = raw_quat
            if timestamp_ns is not None:
                self.prev_timestamp_ns = int(timestamp_ns)
            return raw_quat

        dt_s = self._dt_s(timestamp_ns)
        alpha = 1.0 - math.exp(-dt_s / self.tau_s)
        filtered_quat = _quat_slerp_xyzw(
            self.prev_quat_xyzw,
            _same_quat_hemisphere(raw_quat, self.prev_quat_xyzw),
            alpha,
        )
        self.prev_quat_xyzw = filtered_quat
        if timestamp_ns is not None:
            self.prev_timestamp_ns = int(timestamp_ns)
        return filtered_quat

    def _dt_s(self, timestamp_ns: int | None) -> float:
        if timestamp_ns is not None and self.prev_timestamp_ns is not None:
            dt_s = (int(timestamp_ns) - self.prev_timestamp_ns) * 1e-9
            if math.isfinite(dt_s) and dt_s > 0.0:
                return float(dt_s)
        return self.fallback_dt_s


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _normalize_quat_xyzw(values: Sequence[float]) -> tuple[float, float, float, float]:
    quat = tuple(float(value) for value in values)
    if len(quat) != 4:
        raise ValueError("quat_xyzw must have length 4")
    if any(not math.isfinite(value) for value in quat):
        raise ValueError("quat_xyzw must contain finite values")
    norm = math.sqrt(sum(value * value for value in quat))
    if norm <= 0.0:
        raise ValueError("quat_xyzw norm must be positive")
    return (
        quat[0] / norm,
        quat[1] / norm,
        quat[2] / norm,
        quat[3] / norm,
    )


def _same_quat_hemisphere(
    quat_xyzw: Sequence[float],
    reference_xyzw: Sequence[float],
) -> tuple[float, float, float, float]:
    quat = _normalize_quat_xyzw(quat_xyzw)
    reference = _normalize_quat_xyzw(reference_xyzw)
    dot = sum(quat[index] * reference[index] for index in range(4))
    if dot < 0.0:
        return (-quat[0], -quat[1], -quat[2], -quat[3])
    return quat


def _quat_slerp_xyzw(
    start_xyzw: Sequence[float],
    end_xyzw: Sequence[float],
    alpha: float,
) -> tuple[float, float, float, float]:
    start = np.asarray(_normalize_quat_xyzw(start_xyzw), dtype=float)
    end = np.asarray(_same_quat_hemisphere(end_xyzw, start), dtype=float)
    t = _clip(float(alpha), 0.0, 1.0)
    dot = _clip(float(start @ end), -1.0, 1.0)
    if dot > 0.9995:
        return _normalize_quat_xyzw(((1.0 - t) * start + t * end).tolist())
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    if abs(sin_theta) <= 1e-12:
        return _normalize_quat_xyzw(((1.0 - t) * start + t * end).tolist())
    weight_start = math.sin((1.0 - t) * theta) / sin_theta
    weight_end = math.sin(t * theta) / sin_theta
    return _normalize_quat_xyzw((weight_start * start + weight_end * end).tolist())


def _matrix3x3(matrix: Sequence[Sequence[float]], name: str) -> tuple[tuple[float, float, float], ...]:
    rows = tuple(tuple(float(value) for value in row) for row in matrix)
    if len(rows) != 3:
        raise ValueError(f"{name} must have 3 rows")
    if any(len(row) != 3 for row in rows):
        raise ValueError(f"{name} must have 3 columns")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError(f"{name} must contain finite values")
    return rows


def _rotation_matrix3x3(matrix: Sequence[Sequence[float]], name: str) -> tuple[tuple[float, float, float], ...]:
    rows = _matrix3x3(matrix, name)
    rotation = Rotation.from_matrix(rows).as_matrix()
    return (
        (float(rotation[0][0]), float(rotation[0][1]), float(rotation[0][2])),
        (float(rotation[1][0]), float(rotation[1][1]), float(rotation[1][2])),
        (float(rotation[2][0]), float(rotation[2][1]), float(rotation[2][2])),
    )


def _pose_to_mat4x4_m(pose: Pose) -> tuple[tuple[float, float, float, float], ...]:
    rotation = Rotation.from_quat(pose.quat_xyzw).as_matrix()
    return _mat4x4_from_rotation_translation(rotation, pose.xyz_m)


def _rotation3_to_mat4x4(
    rotation: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float, float], ...]:
    rotation_matrix = Rotation.from_matrix(rotation).as_matrix()
    return _mat4x4_from_rotation_translation(rotation_matrix, (0.0, 0.0, 0.0))


def _mat4x4_from_rotation_translation(
    rotation: np.ndarray,
    translation: Sequence[float],
) -> tuple[tuple[float, float, float, float], ...]:
    return (
        (float(rotation[0][0]), float(rotation[0][1]), float(rotation[0][2]), float(translation[0])),
        (float(rotation[1][0]), float(rotation[1][1]), float(rotation[1][2]), float(translation[1])),
        (float(rotation[2][0]), float(rotation[2][1]), float(rotation[2][2]), float(translation[2])),
        (0.0, 0.0, 0.0, 1.0),
    )


def _mat4x4_mul(
    left: tuple[tuple[float, float, float, float], ...],
    right: tuple[tuple[float, float, float, float], ...],
) -> tuple[tuple[float, float, float, float], ...]:
    result = np.asarray(left, dtype=float) @ np.asarray(right, dtype=float)
    return _array_to_mat4x4(result)


def _invert_rigid_mat4x4(
    mat: tuple[tuple[float, float, float, float], ...],
) -> tuple[tuple[float, float, float, float], ...]:
    rotation = Rotation.from_matrix(tuple(row[:3] for row in mat[:3]))
    inv_rotation = rotation.inv().as_matrix()
    translation = np.asarray((mat[0][3], mat[1][3], mat[2][3]), dtype=float)
    inv_translation = -(inv_rotation @ translation)
    return _mat4x4_from_rotation_translation(inv_rotation, inv_translation)


def _scale_transform_translation(
    mat: tuple[tuple[float, float, float, float], ...],
    scale: float,
) -> tuple[tuple[float, float, float, float], ...]:
    return (
        (mat[0][0], mat[0][1], mat[0][2], mat[0][3] * scale),
        (mat[1][0], mat[1][1], mat[1][2], mat[1][3] * scale),
        (mat[2][0], mat[2][1], mat[2][2], mat[2][3] * scale),
        (mat[3][0], mat[3][1], mat[3][2], mat[3][3]),
    )


def _array_to_mat4x4(array: np.ndarray) -> tuple[tuple[float, float, float, float], ...]:
    return (
        (float(array[0][0]), float(array[0][1]), float(array[0][2]), float(array[0][3])),
        (float(array[1][0]), float(array[1][1]), float(array[1][2]), float(array[1][3])),
        (float(array[2][0]), float(array[2][1]), float(array[2][2]), float(array[2][3])),
        (float(array[3][0]), float(array[3][1]), float(array[3][2]), float(array[3][3])),
    )


def _rad_tuple_to_deg7(q_rad: Sequence[float]) -> tuple[float, float, float, float, float, float, float]:
    q = tuple(rad_to_deg(value) for value in q_rad)
    if len(q) != 7:
        raise ValueError("q_rad must have length 7")
    return (q[0], q[1], q[2], q[3], q[4], q[5], q[6])
