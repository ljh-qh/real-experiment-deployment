from collections.abc import Sequence

from unirobot_infra.core.types import RobotAction, RobotState, Side
from unirobot_infra.core.units import deg_to_rad, ns_to_s, rad_to_deg
from unirobot_infra.devices.keyboard import KeyboardFrame
from unirobot_infra.robots.arm.tianji_tcp_pose import TianjiIkSolver


TRANSLATION_COMMAND_AXES = {
    "x_pos": (0, 1.0),
    "x_neg": (0, -1.0),
    "y_pos": (1, 1.0),
    "y_neg": (1, -1.0),
    "z_pos": (2, 1.0),
    "z_neg": (2, -1.0),
}

ROTATION_COMMAND_AXES = {
    "roll_pos": (0, 1.0),
    "roll_neg": (0, -1.0),
    "pitch_pos": (1, 1.0),
    "pitch_neg": (1, -1.0),
    "yaw_pos": (2, 1.0),
    "yaw_neg": (2, -1.0),
}


class TianjiCartesianJogMapper:
    def __init__(
        self,
        side: Side,
        solver: TianjiIkSolver,
        robot_state: RobotState,
        sdk_xyz_from_robot_world: Sequence[Sequence[float]],
        translation_speed_mm_s: float,
        rotation_speed_deg_s: float,
        max_dt_s: float,
    ) -> None:
        if robot_state.arm_joint is None:
            raise ValueError("Tianji Cartesian jog mapper requires arm_joint state")
        self.side = side
        self.solver = solver
        self.reference_q_deg = _rad_tuple_to_deg7(robot_state.arm_joint.q_rad)
        self.target_xyzabc_mm_deg = solver.kinematics.fk_xyzabc_mm_deg(self.reference_q_deg)
        self.last_q_deg: tuple[float, float, float, float, float, float, float] | None = self.reference_q_deg
        self.sdk_xyz_from_robot_world = _matrix3x3(
            sdk_xyz_from_robot_world,
            "sdk_xyz_from_robot_world",
        )
        self.translation_speed_mm_s = float(translation_speed_mm_s)
        self.rotation_speed_deg_s = float(rotation_speed_deg_s)
        self.max_dt_s = float(max_dt_s)
        self.last_timestamp_ns: int | None = None

    def map_frame(self, frame: KeyboardFrame) -> tuple[RobotAction, ...] | None:
        dt_s = self._frame_dt_s(frame.timestamp_ns)
        if not frame.active_commands:
            return None

        delta_robot_world_xyz_mm, delta_abc_deg = _delta_xyzabc(
            active_commands=frame.active_commands,
            dt_s=dt_s,
            translation_speed_mm_s=self.translation_speed_mm_s,
            rotation_speed_deg_s=self.rotation_speed_deg_s,
        )
        delta_sdk_xyz_mm = _mat_vec_mul(self.sdk_xyz_from_robot_world, delta_robot_world_xyz_mm)
        self.target_xyzabc_mm_deg = (
            self.target_xyzabc_mm_deg[0] + delta_sdk_xyz_mm[0],
            self.target_xyzabc_mm_deg[1] + delta_sdk_xyz_mm[1],
            self.target_xyzabc_mm_deg[2] + delta_sdk_xyz_mm[2],
            self.target_xyzabc_mm_deg[3] + delta_abc_deg[0],
            self.target_xyzabc_mm_deg[4] + delta_abc_deg[1],
            self.target_xyzabc_mm_deg[5] + delta_abc_deg[2],
        )
        q_deg = self.solver.solve(
            target_xyzabc_mm_deg=self.target_xyzabc_mm_deg,
            reference_q_deg=self.reference_q_deg,
            last_q_deg=self.last_q_deg,
        )
        self.last_q_deg = q_deg
        return (
            RobotAction(
                side=self.side,
                timestamp_ns=frame.timestamp_ns,
                arm_q_rad=tuple(deg_to_rad(value) for value in q_deg),
            ),
        )

    def _frame_dt_s(self, timestamp_ns: int) -> float:
        if self.last_timestamp_ns is None:
            self.last_timestamp_ns = int(timestamp_ns)
            return 0.0
        dt_s = ns_to_s(int(timestamp_ns) - self.last_timestamp_ns)
        self.last_timestamp_ns = int(timestamp_ns)
        return _clip(dt_s, 0.0, self.max_dt_s)


def _delta_xyzabc(
    active_commands: tuple[str, ...],
    dt_s: float,
    translation_speed_mm_s: float,
    rotation_speed_deg_s: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    xyz_direction = [0.0, 0.0, 0.0]
    abc_direction = [0.0, 0.0, 0.0]
    for command in active_commands:
        if command in TRANSLATION_COMMAND_AXES:
            axis, sign = TRANSLATION_COMMAND_AXES[command]
            xyz_direction[axis] += sign
        elif command in ROTATION_COMMAND_AXES:
            axis, sign = ROTATION_COMMAND_AXES[command]
            abc_direction[axis] += sign
        else:
            raise ValueError(f"unsupported keyboard command: {command}")
    return (
        (
            xyz_direction[0] * translation_speed_mm_s * dt_s,
            xyz_direction[1] * translation_speed_mm_s * dt_s,
            xyz_direction[2] * translation_speed_mm_s * dt_s,
        ),
        (
            abc_direction[0] * rotation_speed_deg_s * dt_s,
            abc_direction[1] * rotation_speed_deg_s * dt_s,
            abc_direction[2] * rotation_speed_deg_s * dt_s,
        ),
    )


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _matrix3x3(matrix: Sequence[Sequence[float]], name: str) -> tuple[tuple[float, float, float], ...]:
    rows = tuple(tuple(float(value) for value in row) for row in matrix)
    if len(rows) != 3:
        raise ValueError(f"{name} must have 3 rows")
    if any(len(row) != 3 for row in rows):
        raise ValueError(f"{name} must have 3 columns")
    return rows


def _mat_vec_mul(
    matrix: tuple[tuple[float, float, float], ...],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    )


def _rad_tuple_to_deg7(q_rad: Sequence[float]) -> tuple[float, float, float, float, float, float, float]:
    q = tuple(rad_to_deg(value) for value in q_rad)
    if len(q) != 7:
        raise ValueError("q_rad must have length 7")
    return (q[0], q[1], q[2], q[3], q[4], q[5], q[6])
