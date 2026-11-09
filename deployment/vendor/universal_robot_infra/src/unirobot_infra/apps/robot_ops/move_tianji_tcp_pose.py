from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import time

import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation

from unirobot_infra.core.types import RobotAction, RobotState
from unirobot_infra.core.units import deg_to_rad, rad_to_deg
from unirobot_infra.envs.builder import build_base_env
from unirobot_infra.perception.frame_transform import matrix3x3_tuple, transform_vector
from unirobot_infra.robots.arm.tianji_tcp_pose import TianjiIkSolver
from unirobot_infra.teleop.mapping.tianji_kinematics import TianjiKinematics
from unirobot_infra.utils.logging import get_logger, setup_logging
from unirobot_infra.utils.sdk_output_router import start_sdk_output_router_from_config


CONFIG_PATH = str(Path(__file__).resolve().parents[4] / "configs")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="apps/robot_ops/move_tianji_tcp_pose/left")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.logging.level)
    logger = get_logger(__name__)
    with start_sdk_output_router_from_config(cfg.app.sdk_output) as sdk_output_router:
        if sdk_output_router is not None:
            logger.info("SDK 输出重定向目录=%s", sdk_output_router.run_dir)
        side = str(cfg.tcp_pose_move.side)
        logger.info(
            "move Tianji TCP pose side=%s position_frame=%s offset_frame=%s orientation_frame=%s base_xyz_m=%s offset_m=%s target_xyz_m=%s",
            side,
            cfg.tcp_pose_move.position_frame,
            cfg.tcp_pose_move.position_offset_frame,
            cfg.tcp_pose_move.orientation_frame,
            _format_xyz_m(_base_position_base_m(cfg)),
            _format_xyz_m(_position_offset_base_m(cfg, side)),
            _format_xyz_m(_target_position_base_m(cfg, side)),
        )

        env = build_base_env(cfg)
        try:
            _, info = env.reset()
            if bool(cfg.tcp_pose_move.servo.enabled):
                _move_tcp_pose_servo(cfg, env, side, logger)
            else:
                robot_state = _robot_state_by_side(info["robot_states"], side)
                action, target_xyzabc = _build_tcp_pose_action(cfg, robot_state)
                if not bool(cfg.tcp_pose_move.dry_run):
                    env.controller.send_actions((action,), blocking=bool(cfg.tcp_pose_move.blocking))
                logger.info(
                    "tcp pose result dry_run=%s side=%s target_xyzabc_mm_deg=%s target_q_deg=%s",
                    bool(cfg.tcp_pose_move.dry_run),
                    side,
                    _format_xyzabc(target_xyzabc),
                    _format_q_deg(tuple(rad_to_deg(value) for value in action.arm_q_rad)),
                )
        finally:
            env.close()


def _move_tcp_pose_servo(cfg: DictConfig, env, side: str, logger) -> None:
    _validate_servo_config(cfg)
    kinematics, solver = _build_kinematics_and_solver(cfg, side)
    initial_state = _robot_state_by_side(env.controller.read_state(), side)
    initial_pose = _current_tcp_pose_mat_mm(kinematics, initial_state)
    target_pose = _target_pose_mat_mm(cfg, side, initial_pose)
    target_xyzabc = kinematics.mat4x4_to_xyzabc_mm_deg(target_pose)
    period_s = 1.0 / float(cfg.tcp_pose_move.servo.rate_hz)
    deadline_s = time.monotonic() + float(cfg.tcp_pose_move.servo.timeout_s)
    reached_count = 0
    command_count = 0
    last_action = None
    last_servo_xyzabc = target_xyzabc
    while time.monotonic() < deadline_s:
        loop_start = time.monotonic()
        robot_state = _robot_state_by_side(env.controller.read_state(), side)
        current_pose = _current_tcp_pose_mat_mm(kinematics, robot_state)
        if _pose_reached(cfg, current_pose, target_pose):
            reached_count += 1
        else:
            reached_count = 0
            servo_pose = _step_pose_mat_towards(
                current_pose,
                target_pose,
                max_translation_step_m=float(cfg.tcp_pose_move.servo.max_translation_step_m),
                max_orientation_step_deg=float(cfg.tcp_pose_move.servo.max_orientation_step_deg),
            )
            last_action, last_servo_xyzabc = _build_tcp_pose_action_to_mat(
                cfg,
                robot_state,
                kinematics,
                solver,
                servo_pose,
            )
            command_count += 1
            if bool(cfg.tcp_pose_move.dry_run):
                break
            env.controller.send_actions((last_action,), blocking=False)
        if reached_count >= int(cfg.tcp_pose_move.servo.reached_consecutive_frames):
            logger.info(
                "tcp pose servo reached dry_run=%s side=%s commands=%s target_xyzabc_mm_deg=%s",
                bool(cfg.tcp_pose_move.dry_run),
                side,
                command_count,
                _format_xyzabc(target_xyzabc),
            )
            return
        _sleep_remaining(loop_start, period_s)
    if bool(cfg.tcp_pose_move.dry_run) and last_action is not None:
        logger.info(
            "tcp pose servo dry_run side=%s first_step_xyzabc_mm_deg=%s target_q_deg=%s final_target_xyzabc_mm_deg=%s",
            side,
            _format_xyzabc(last_servo_xyzabc),
            _format_q_deg(tuple(rad_to_deg(value) for value in last_action.arm_q_rad)),
            _format_xyzabc(target_xyzabc),
        )
        return
    raise RuntimeError(f"Tianji TCP pose servo not reached within timeout_s={float(cfg.tcp_pose_move.servo.timeout_s)}")


def _build_tcp_pose_action(
    cfg: DictConfig,
    robot_state: RobotState,
) -> tuple[RobotAction, tuple[float, float, float, float, float, float]]:
    if robot_state.arm_joint is None:
        raise RuntimeError("Tianji TCP pose move requires arm_joint state")
    side = robot_state.side
    kinematics, solver = _build_kinematics_and_solver(cfg, side)
    current_pose = _current_tcp_pose_mat_mm(kinematics, robot_state)
    target_pose = _target_pose_mat_mm(cfg, side, current_pose)
    return _build_tcp_pose_action_to_mat(cfg, robot_state, kinematics, solver, target_pose)


def _build_tcp_pose_action_to_mat(
    cfg: DictConfig,
    robot_state: RobotState,
    kinematics: TianjiKinematics,
    solver: TianjiIkSolver,
    target_pose: Sequence[Sequence[float]],
) -> tuple[RobotAction, tuple[float, float, float, float, float, float]]:
    if robot_state.arm_joint is None:
        raise RuntimeError("Tianji TCP pose move requires arm_joint state")
    side = robot_state.side
    reference_q_deg = _rad_tuple_to_deg7(robot_state.arm_joint.q_rad)
    target_xyzabc = kinematics.mat4x4_to_xyzabc_mm_deg(target_pose)
    target_q_deg = solver.solve(
        target_xyzabc_mm_deg=target_xyzabc,
        reference_q_deg=reference_q_deg,
        last_q_deg=reference_q_deg,
    )
    return (
        RobotAction(
            side=side,
            timestamp_ns=robot_state.timestamp_ns,
            arm_q_rad=tuple(deg_to_rad(value) for value in target_q_deg),
        ),
        target_xyzabc,
    )


def _build_kinematics_and_solver(cfg: DictConfig, side: str) -> tuple[TianjiKinematics, TianjiIkSolver]:
    arm_id = str(cfg.robot.hardware.arm.arms[side].arm_id)
    kinematics = TianjiKinematics(cfg.robot.hardware.arm, arm_id=arm_id)
    kinematics.initialize()
    solver = TianjiIkSolver(
        kinematics=kinematics,
        max_joint_step_deg=float(cfg.tcp_pose_move.max_joint_step_deg),
        joint_limit_mode=str(cfg.tcp_pose_move.joint_limit_mode),
    )
    return kinematics, solver


def _current_tcp_pose_mat_mm(
    kinematics: TianjiKinematics,
    robot_state: RobotState,
) -> tuple[tuple[float, float, float, float], ...]:
    if robot_state.arm_joint is None:
        raise RuntimeError("Tianji TCP pose move requires arm_joint state")
    q_deg = _rad_tuple_to_deg7(robot_state.arm_joint.q_rad)
    current_xyzabc = kinematics.fk_xyzabc_mm_deg(q_deg)
    return kinematics.xyzabc_to_mat4x4(current_xyzabc)


def _target_pose_mat_mm(
    cfg: DictConfig,
    side: str,
    current_pose_mat: Sequence[Sequence[float]] | None = None,
) -> tuple[tuple[float, float, float, float], ...]:
    xyz_m = _target_position_base_m(cfg, side)
    rotation = _rotation_base_from_tcp(cfg, side, current_pose_mat)
    return (
        (rotation[0][0], rotation[0][1], rotation[0][2], xyz_m[0] * 1000.0),
        (rotation[1][0], rotation[1][1], rotation[1][2], xyz_m[1] * 1000.0),
        (rotation[2][0], rotation[2][1], rotation[2][2], xyz_m[2] * 1000.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _target_position_base_m(cfg: DictConfig, side: str) -> tuple[float, float, float]:
    if str(cfg.tcp_pose_move.position_frame) != "arm_base":
        raise ValueError("tcp_pose_move.position_frame must be arm_base")
    position = _base_position_base_m(cfg)
    offset = _position_offset_base_m(cfg, side)
    return (position[0] + offset[0], position[1] + offset[1], position[2] + offset[2])


def _base_position_base_m(cfg: DictConfig) -> tuple[float, float, float]:
    return _vector3_tuple(cfg.tcp_pose_move.xyz_m, "tcp_pose_move.xyz_m")


def _position_offset_base_m(cfg: DictConfig, side: str) -> tuple[float, float, float]:
    offset = _vector3_tuple(cfg.tcp_pose_move.position_offset_m, "tcp_pose_move.position_offset_m")
    offset_frame = str(cfg.tcp_pose_move.position_offset_frame)
    if offset_frame == "arm_base":
        return offset
    if offset_frame == "robot_world":
        base_from_world = matrix3x3_tuple(
            cfg.robot.hardware.arm.arms[side].sdk_xyz_from_robot_world,
            f"robot.hardware.arm.arms.{side}.sdk_xyz_from_robot_world",
        )
        return transform_vector(base_from_world, offset)
    raise ValueError("tcp_pose_move.position_offset_frame must be arm_base or robot_world")


def _pose_reached(
    cfg: DictConfig,
    current_pose: Sequence[Sequence[float]],
    target_pose: Sequence[Sequence[float]],
) -> bool:
    position_error_m = _position_error_m(
        _position_from_pose_mat_m(current_pose),
        _position_from_pose_mat_m(target_pose),
    )
    orientation_error_deg = _rotation_error_deg(
        _rotation_from_pose_mat(current_pose),
        _rotation_from_pose_mat(target_pose),
    )
    return (
        position_error_m <= float(cfg.tcp_pose_move.servo.position_tolerance_m)
        and orientation_error_deg <= float(cfg.tcp_pose_move.servo.orientation_tolerance_deg)
    )


def _step_pose_mat_towards(
    current_pose: Sequence[Sequence[float]],
    target_pose: Sequence[Sequence[float]],
    max_translation_step_m: float,
    max_orientation_step_deg: float,
) -> tuple[tuple[float, float, float, float], ...]:
    xyz_m = _step_xyz_towards(
        _position_from_pose_mat_m(current_pose),
        _position_from_pose_mat_m(target_pose),
        max_translation_step_m,
    )
    rotation = _step_rotation_towards(
        _rotation_from_pose_mat(current_pose),
        _rotation_from_pose_mat(target_pose),
        max_orientation_step_deg,
    )
    return (
        (rotation[0][0], rotation[0][1], rotation[0][2], xyz_m[0] * 1000.0),
        (rotation[1][0], rotation[1][1], rotation[1][2], xyz_m[1] * 1000.0),
        (rotation[2][0], rotation[2][1], rotation[2][2], xyz_m[2] * 1000.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _position_from_pose_mat_m(pose_mat: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    rows = tuple(tuple(float(value) for value in row) for row in pose_mat)
    if len(rows) != 4:
        raise ValueError("pose_mat must have 4 rows")
    if any(len(row) != 4 for row in rows):
        raise ValueError("pose_mat must have 4 columns")
    return (rows[0][3] * 0.001, rows[1][3] * 0.001, rows[2][3] * 0.001)


def _position_error_m(
    current_xyz_m: tuple[float, float, float],
    target_xyz_m: tuple[float, float, float],
) -> float:
    return float(
        np.linalg.norm(np.asarray(current_xyz_m, dtype=np.float64) - np.asarray(target_xyz_m, dtype=np.float64))
    )


def _rotation_error_deg(
    current_rotation: tuple[tuple[float, float, float], ...],
    target_rotation: tuple[tuple[float, float, float], ...],
) -> float:
    current = np.asarray(current_rotation, dtype=np.float64)
    target = np.asarray(target_rotation, dtype=np.float64)
    delta = target.T @ current
    cosine = (float(np.trace(delta)) - 1.0) * 0.5
    angle_rad = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return float(np.rad2deg(angle_rad))


def _step_xyz_towards(
    current_xyz_m: tuple[float, float, float],
    target_xyz_m: tuple[float, float, float],
    max_step_m: float,
) -> tuple[float, float, float]:
    current = np.asarray(current_xyz_m, dtype=np.float64)
    target = np.asarray(target_xyz_m, dtype=np.float64)
    delta = target - current
    distance_m = float(np.linalg.norm(delta))
    if distance_m <= max_step_m:
        return target_xyz_m
    stepped = current + delta * (max_step_m / distance_m)
    return (float(stepped[0]), float(stepped[1]), float(stepped[2]))


def _step_rotation_towards(
    current_rotation: tuple[tuple[float, float, float], ...],
    target_rotation: tuple[tuple[float, float, float], ...],
    max_step_deg: float,
) -> tuple[tuple[float, float, float], ...]:
    current = Rotation.from_matrix(np.asarray(current_rotation, dtype=np.float64))
    target = Rotation.from_matrix(np.asarray(target_rotation, dtype=np.float64))
    delta = target * current.inv()
    angle_deg = float(np.rad2deg(delta.magnitude()))
    if angle_deg <= max_step_deg:
        return target_rotation
    stepped = Rotation.from_rotvec(delta.as_rotvec() * (max_step_deg / angle_deg)) * current
    matrix = stepped.as_matrix()
    return (
        (float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2])),
        (float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2])),
        (float(matrix[2, 0]), float(matrix[2, 1]), float(matrix[2, 2])),
    )


def _validate_servo_config(cfg: DictConfig) -> None:
    if float(cfg.tcp_pose_move.servo.rate_hz) <= 0.0:
        raise ValueError("tcp_pose_move.servo.rate_hz must be positive")
    if float(cfg.tcp_pose_move.servo.max_translation_step_m) <= 0.0:
        raise ValueError("tcp_pose_move.servo.max_translation_step_m must be positive")
    if float(cfg.tcp_pose_move.servo.max_orientation_step_deg) <= 0.0:
        raise ValueError("tcp_pose_move.servo.max_orientation_step_deg must be positive")
    if float(cfg.tcp_pose_move.servo.position_tolerance_m) <= 0.0:
        raise ValueError("tcp_pose_move.servo.position_tolerance_m must be positive")
    if float(cfg.tcp_pose_move.servo.orientation_tolerance_deg) <= 0.0:
        raise ValueError("tcp_pose_move.servo.orientation_tolerance_deg must be positive")
    if float(cfg.tcp_pose_move.servo.timeout_s) <= 0.0:
        raise ValueError("tcp_pose_move.servo.timeout_s must be positive")
    if int(cfg.tcp_pose_move.servo.reached_consecutive_frames) <= 0:
        raise ValueError("tcp_pose_move.servo.reached_consecutive_frames must be positive")


def _rotation_base_from_tcp(
    cfg: DictConfig,
    side: str,
    current_pose_mat: Sequence[Sequence[float]] | None = None,
) -> tuple[tuple[float, float, float], ...]:
    orientation_mode = str(cfg.tcp_pose_move.orientation_mode)
    if orientation_mode == "current_tcp":
        if current_pose_mat is None:
            raise ValueError("current_pose_mat is required when orientation_mode=current_tcp")
        return _rotation_from_pose_mat(current_pose_mat)
    if orientation_mode != "axes":
        raise ValueError("tcp_pose_move.orientation_mode must be current_tcp or axes")

    x_axis = _unit_vector3(cfg.tcp_pose_move.local_x_axis, "tcp_pose_move.local_x_axis")
    y_axis = _unit_vector3(cfg.tcp_pose_move.local_y_axis, "tcp_pose_move.local_y_axis")
    if abs(_dot(x_axis, y_axis)) > 1e-6:
        raise ValueError("tcp_pose_move local_x_axis and local_y_axis must be perpendicular")
    z_axis = _cross(x_axis, y_axis)
    if str(cfg.tcp_pose_move.orientation_frame) == "robot_world":
        base_from_world = matrix3x3_tuple(
            cfg.robot.hardware.arm.arms[side].sdk_xyz_from_robot_world,
            f"robot.hardware.arm.arms.{side}.sdk_xyz_from_robot_world",
        )
        x_axis = transform_vector(base_from_world, x_axis)
        y_axis = transform_vector(base_from_world, y_axis)
        z_axis = transform_vector(base_from_world, z_axis)
    elif str(cfg.tcp_pose_move.orientation_frame) != "arm_base":
        raise ValueError("tcp_pose_move.orientation_frame must be arm_base or robot_world")
    return _rotation_from_axes(x_axis, y_axis, z_axis)


def _rotation_from_pose_mat(pose_mat: Sequence[Sequence[float]]) -> tuple[tuple[float, float, float], ...]:
    rows = tuple(tuple(float(value) for value in row) for row in pose_mat)
    if len(rows) != 4:
        raise ValueError("current_pose_mat must have 4 rows")
    if any(len(row) != 4 for row in rows):
        raise ValueError("current_pose_mat must have 4 columns")
    return _rotation_from_axes(
        (rows[0][0], rows[1][0], rows[2][0]),
        (rows[0][1], rows[1][1], rows[2][1]),
        (rows[0][2], rows[1][2], rows[2][2]),
    )


def _rotation_from_axes(
    x_axis: tuple[float, float, float],
    y_axis: tuple[float, float, float],
    z_axis: tuple[float, float, float],
) -> tuple[tuple[float, float, float], ...]:
    rotation = np.asarray(
        [
            [x_axis[0], y_axis[0], z_axis[0]],
            [x_axis[1], y_axis[1], z_axis[1]],
            [x_axis[2], y_axis[2], z_axis[2]],
        ],
        dtype=np.float64,
    )
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("tcp_pose_move orientation axes must form an orthonormal basis")
    if float(np.linalg.det(rotation)) < 0.999999:
        raise ValueError("tcp_pose_move orientation axes must form a right-handed basis")
    return (
        (float(rotation[0, 0]), float(rotation[0, 1]), float(rotation[0, 2])),
        (float(rotation[1, 0]), float(rotation[1, 1]), float(rotation[1, 2])),
        (float(rotation[2, 0]), float(rotation[2, 1]), float(rotation[2, 2])),
    )


def _vector3_tuple(values, name: str) -> tuple[float, float, float]:
    vector = tuple(float(value) for value in values)
    if len(vector) != 3:
        raise ValueError(f"{name} must have length 3")
    return (vector[0], vector[1], vector[2])


def _unit_vector3(values, name: str) -> tuple[float, float, float]:
    vector = np.asarray(_vector3_tuple(values, name), dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"{name} must be non-zero")
    unit = vector / norm
    return (float(unit[0]), float(unit[1]), float(unit[2]))


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return float(left[0] * right[0] + left[1] * right[1] + left[2] * right[2])


def _cross(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    result = np.cross(np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64))
    return (float(result[0]), float(result[1]), float(result[2]))


def _robot_state_by_side(states: Sequence[RobotState], side: str) -> RobotState:
    for state in states:
        if state.side == side:
            return state
    raise RuntimeError(f"robot state missing side={side}")


def _rad_tuple_to_deg7(q_rad: Sequence[float]) -> tuple[float, float, float, float, float, float, float]:
    q = tuple(rad_to_deg(value) for value in q_rad)
    if len(q) != 7:
        raise ValueError("q_rad must have length 7")
    return (q[0], q[1], q[2], q[3], q[4], q[5], q[6])


def _format_xyz_m(xyz_m: tuple[float, float, float]) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in xyz_m) + "]"


def _format_xyzabc(xyzabc: tuple[float, float, float, float, float, float]) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in xyzabc) + "]"


def _format_q_deg(q_deg: tuple[float, ...]) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in q_deg) + "]"


def _sleep_remaining(loop_start_s: float, period_s: float) -> None:
    remaining_s = period_s - (time.monotonic() - loop_start_s)
    if remaining_s > 0.0:
        time.sleep(remaining_s)


if __name__ == "__main__":
    main()
