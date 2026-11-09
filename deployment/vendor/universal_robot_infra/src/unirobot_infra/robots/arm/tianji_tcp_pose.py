from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from omegaconf import DictConfig

from unirobot_infra.core.units import deg_to_rad, rad_to_deg
from unirobot_infra.perception.frame_transform import matrix3x3_tuple
from unirobot_infra.teleop.mapping.tianji_kinematics import TianjiKinematics


class TianjiIkSolver:
    def __init__(
        self,
        kinematics: TianjiKinematics,
        max_joint_step_deg: float,
        joint_limit_mode: str,
    ) -> None:
        self.kinematics = kinematics
        self.max_joint_step_deg = float(max_joint_step_deg)
        self.joint_limit_mode = str(joint_limit_mode)

    def solve(
        self,
        target_xyzabc_mm_deg: tuple[float, float, float, float, float, float],
        reference_q_deg: tuple[float, float, float, float, float, float, float],
        last_q_deg: tuple[float, float, float, float, float, float, float] | None,
    ) -> tuple[float, float, float, float, float, float, float]:
        ik_reference_q_deg = last_q_deg if last_q_deg is not None else reference_q_deg
        candidate_q_deg = self.kinematics.ik_xyzabc_mm_deg(
            xyzabc_mm_deg=target_xyzabc_mm_deg,
            reference_q_deg=ik_reference_q_deg,
        )
        if last_q_deg is None:
            return candidate_q_deg
        return _limit_joint_step(
            candidate_q_deg=candidate_q_deg,
            last_q_deg=last_q_deg,
            max_joint_step_deg=self.max_joint_step_deg,
            joint_limit_mode=self.joint_limit_mode,
        )


class TianjiTcpPoseMapper:
    def __init__(
        self,
        side: str,
        kinematics: TianjiKinematics,
        solver: TianjiIkSolver | None,
        sdk_xyz_from_robot_world: Sequence[Sequence[float]],
        base_origin_world_m: Sequence[float],
    ) -> None:
        self.side = str(side)
        self.kinematics = kinematics
        self.solver = solver
        self.sdk_from_world = matrix3x3_tuple(
            sdk_xyz_from_robot_world,
            f"robot.hardware.arm.arms.{self.side}.sdk_xyz_from_robot_world",
        )
        self.world_from_sdk = _transpose3(self.sdk_from_world)
        self.base_origin_world_m = _vec3(
            base_origin_world_m,
            f"robot.hardware.arm.arms.{self.side}.base_origin_world_m",
        )

    @classmethod
    def from_config(
        cls,
        arm_cfg: DictConfig,
        side: str,
        max_joint_step_deg: float,
        joint_limit_mode: str,
    ) -> "TianjiTcpPoseMapper":
        arm_id = str(arm_cfg.arms[side].arm_id)
        kinematics = TianjiKinematics(arm_cfg, arm_id=arm_id)
        kinematics.initialize()
        solver = TianjiIkSolver(
            kinematics=kinematics,
            max_joint_step_deg=float(max_joint_step_deg),
            joint_limit_mode=str(joint_limit_mode),
        )
        return cls(
            side=side,
            kinematics=kinematics,
            solver=solver,
            sdk_xyz_from_robot_world=arm_cfg.arms[side].sdk_xyz_from_robot_world,
            base_origin_world_m=_base_origin_world_m(arm_cfg, side),
        )

    @classmethod
    def fk_from_config(cls, arm_cfg: DictConfig, side: str) -> "TianjiTcpPoseMapper":
        arm_id = str(arm_cfg.arms[side].arm_id)
        kinematics = TianjiKinematics(arm_cfg, arm_id=arm_id)
        kinematics.initialize()
        return cls(
            side=side,
            kinematics=kinematics,
            solver=None,
            sdk_xyz_from_robot_world=arm_cfg.arms[side].sdk_xyz_from_robot_world,
            base_origin_world_m=_base_origin_world_m(arm_cfg, side),
        )

    def tcp_pose_world_from_q_rad(self, q_rad: Sequence[float]) -> tuple[tuple[float, float, float, float], ...]:
        pose_base = self.tcp_pose_base_from_q_rad(q_rad)
        return tcp_pose_world_from_base(pose_base, self.sdk_from_world, self.base_origin_world_m)

    def tcp_pose_base_from_q_rad(self, q_rad: Sequence[float]) -> tuple[tuple[float, float, float, float], ...]:
        q_deg = rad_tuple_to_deg7(q_rad)
        xyzabc = self.kinematics.fk_xyzabc_mm_deg(q_deg)
        return self.kinematics.xyzabc_to_mat4x4(xyzabc)

    def delta_world_from_q_rad(
        self,
        current_q_rad: Sequence[float],
        target_q_rad: Sequence[float],
    ) -> tuple[float, ...]:
        current_pose = self.tcp_pose_world_from_q_rad(current_q_rad)
        target_pose = self.tcp_pose_world_from_q_rad(target_q_rad)
        return delta_tcp_pose_world(current_pose, target_pose)

    def q_rad_from_delta_world(
        self,
        current_q_rad: Sequence[float],
        delta_world: Sequence[float],
        last_q_rad: Sequence[float] | None,
    ) -> tuple[float, float, float, float, float, float, float]:
        current_pose_world = self.tcp_pose_world_from_q_rad(current_q_rad)
        target_pose_world = apply_delta_tcp_pose_world(current_pose_world, delta_world)
        target_pose_base = tcp_pose_base_from_world(target_pose_world, self.sdk_from_world, self.base_origin_world_m)
        target_xyzabc = self.kinematics.mat4x4_to_xyzabc_mm_deg(target_pose_base)
        reference_q_deg = rad_tuple_to_deg7(current_q_rad)
        last_q_deg = None if last_q_rad is None else rad_tuple_to_deg7(last_q_rad)
        if self.solver is None:
            raise RuntimeError("TianjiTcpPoseMapper q_rad_from_delta_world requires IK solver")
        target_q_deg = self.solver.solve(
            target_xyzabc_mm_deg=target_xyzabc,
            reference_q_deg=reference_q_deg,
            last_q_deg=last_q_deg,
        )
        return tuple(deg_to_rad(value) for value in target_q_deg)


def tcp_pose_world_from_base(
    pose_base_mm: Sequence[Sequence[float]],
    sdk_xyz_from_robot_world: Sequence[Sequence[float]],
    base_origin_world_m: Sequence[float],
) -> tuple[tuple[float, float, float, float], ...]:
    pose_base = _mat4x4_array(pose_base_mm, "pose_base_mm")
    sdk_from_world = np.asarray(matrix3x3_tuple(sdk_xyz_from_robot_world, "sdk_xyz_from_robot_world"), dtype=np.float64)
    base_origin = np.asarray(_vec3(base_origin_world_m, "base_origin_world_m"), dtype=np.float64)
    world_from_sdk = sdk_from_world.T
    rotation_world_from_tcp = world_from_sdk @ pose_base[:3, :3]
    xyz_world_m = base_origin + world_from_sdk @ (pose_base[:3, 3] * 0.001)
    return _pose_tuple(rotation_world_from_tcp, xyz_world_m)


def tcp_pose_base_from_world(
    pose_world_m: Sequence[Sequence[float]],
    sdk_xyz_from_robot_world: Sequence[Sequence[float]],
    base_origin_world_m: Sequence[float],
) -> tuple[tuple[float, float, float, float], ...]:
    pose_world = _mat4x4_array(pose_world_m, "pose_world_m")
    sdk_from_world = np.asarray(matrix3x3_tuple(sdk_xyz_from_robot_world, "sdk_xyz_from_robot_world"), dtype=np.float64)
    base_origin = np.asarray(_vec3(base_origin_world_m, "base_origin_world_m"), dtype=np.float64)
    rotation_sdk_from_tcp = sdk_from_world @ pose_world[:3, :3]
    xyz_sdk_mm = (sdk_from_world @ (pose_world[:3, 3] - base_origin)) * 1000.0
    return _pose_tuple(rotation_sdk_from_tcp, xyz_sdk_mm)


def delta_tcp_pose_world(
    current_pose_world_m: Sequence[Sequence[float]],
    target_pose_world_m: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    current = _mat4x4_array(current_pose_world_m, "current_pose_world_m")
    target = _mat4x4_array(target_pose_world_m, "target_pose_world_m")
    delta_xyz = target[:3, 3] - current[:3, 3]
    delta_rotation = target[:3, :3] @ current[:3, :3].T
    delta_rotation_6d = rotation_6d_from_matrix(delta_rotation)
    return (
        float(delta_xyz[0]),
        float(delta_xyz[1]),
        float(delta_xyz[2]),
        *delta_rotation_6d,
    )


def apply_delta_tcp_pose_world(
    current_pose_world_m: Sequence[Sequence[float]],
    delta_world: Sequence[float],
) -> tuple[tuple[float, float, float, float], ...]:
    current = _mat4x4_array(current_pose_world_m, "current_pose_world_m")
    delta = _vec9(delta_world, "delta_world")
    delta_xyz = np.asarray(delta[:3], dtype=np.float64)
    delta_rotation = rotation_matrix_from_6d(delta[3:])
    target_xyz = current[:3, 3] + delta_xyz
    target_rotation = delta_rotation @ current[:3, :3]
    return _pose_tuple(target_rotation, target_xyz)


def rotation_6d_from_matrix(rotation: Sequence[Sequence[float]]) -> tuple[float, float, float, float, float, float]:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must have shape 3x3")
    return (
        float(matrix[0, 0]),
        float(matrix[1, 0]),
        float(matrix[2, 0]),
        float(matrix[0, 1]),
        float(matrix[1, 1]),
        float(matrix[2, 1]),
    )


def rotation_matrix_from_6d(values: Sequence[float]) -> np.ndarray:
    vector = _vec6(values, "rotation_6d")
    c1 = np.asarray(vector[:3], dtype=np.float64)
    c2 = np.asarray(vector[3:], dtype=np.float64)
    b1 = _normalized(c1, "rotation_6d first column")
    c2_orthogonal = c2 - float(np.dot(b1, c2)) * b1
    b2 = _normalized(c2_orthogonal, "rotation_6d second column")
    b3 = np.cross(b1, b2)
    return np.column_stack((b1, b2, b3))


def rad_tuple_to_deg7(q_rad: Sequence[float]) -> tuple[float, float, float, float, float, float, float]:
    q = tuple(rad_to_deg(value) for value in q_rad)
    if len(q) != 7:
        raise ValueError("q_rad must have length 7")
    return (q[0], q[1], q[2], q[3], q[4], q[5], q[6])


def _base_origin_world_m(arm_cfg: DictConfig, side: str) -> tuple[float, float, float]:
    distance_m = float(arm_cfg.base_l_base_r_distance_m)
    if distance_m < 0.0:
        raise ValueError("robot.hardware.arm.base_l_base_r_distance_m must be non-negative")
    half_distance_m = distance_m * 0.5
    if side == "left":
        return (0.0, half_distance_m, 0.0)
    if side == "right":
        return (0.0, -half_distance_m, 0.0)
    raise ValueError("side must be left or right")


def _limit_joint_step(
    candidate_q_deg: tuple[float, float, float, float, float, float, float],
    last_q_deg: tuple[float, float, float, float, float, float, float],
    max_joint_step_deg: float,
    joint_limit_mode: str,
) -> tuple[float, float, float, float, float, float, float]:
    deltas = tuple(candidate_q_deg[index] - last_q_deg[index] for index in range(7))
    max_abs_delta = max(abs(value) for value in deltas)
    if max_abs_delta <= max_joint_step_deg:
        return candidate_q_deg
    if joint_limit_mode == "reject":
        raise RuntimeError("Tianji IK joint step exceeds max_joint_step_deg")
    if joint_limit_mode != "ramp":
        raise ValueError("unsupported Tianji joint_limit_mode")
    return tuple(
        last_q_deg[index] + _clip(deltas[index], -max_joint_step_deg, max_joint_step_deg)
        for index in range(7)
    )


def _mat4x4_array(values: Sequence[Sequence[float]], name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape 4x4")
    if not np.allclose(matrix[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-9):
        raise ValueError(f"{name} last row must be [0, 0, 0, 1]")
    return matrix


def _pose_tuple(
    rotation: np.ndarray,
    xyz: np.ndarray,
) -> tuple[tuple[float, float, float, float], ...]:
    return (
        (float(rotation[0, 0]), float(rotation[0, 1]), float(rotation[0, 2]), float(xyz[0])),
        (float(rotation[1, 0]), float(rotation[1, 1]), float(rotation[1, 2]), float(xyz[1])),
        (float(rotation[2, 0]), float(rotation[2, 1]), float(rotation[2, 2]), float(xyz[2])),
        (0.0, 0.0, 0.0, 1.0),
    )


def _transpose3(matrix: tuple[tuple[float, float, float], ...]) -> tuple[tuple[float, float, float], ...]:
    return (
        (matrix[0][0], matrix[1][0], matrix[2][0]),
        (matrix[0][1], matrix[1][1], matrix[2][1]),
        (matrix[0][2], matrix[1][2], matrix[2][2]),
    )


def _vec6(values: Sequence[float], name: str) -> tuple[float, float, float, float, float, float]:
    vector = tuple(float(value) for value in values)
    if len(vector) != 6:
        raise ValueError(f"{name} must have length 6")
    return (vector[0], vector[1], vector[2], vector[3], vector[4], vector[5])


def _vec3(values: Sequence[float], name: str) -> tuple[float, float, float]:
    vector = tuple(float(value) for value in values)
    if len(vector) != 3:
        raise ValueError(f"{name} must have length 3")
    return (vector[0], vector[1], vector[2])


def _vec9(
    values: Sequence[float],
    name: str,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    vector = tuple(float(value) for value in values)
    if len(vector) != 9:
        raise ValueError(f"{name} must have length 9")
    return (vector[0], vector[1], vector[2], vector[3], vector[4], vector[5], vector[6], vector[7], vector[8])


def _normalized(vector: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError(f"{name} must not be zero")
    return vector / norm


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))
