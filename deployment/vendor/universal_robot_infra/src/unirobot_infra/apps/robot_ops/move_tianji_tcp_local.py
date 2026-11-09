from pathlib import Path
from collections.abc import Sequence

import hydra
from omegaconf import DictConfig

from unirobot_infra.core.types import RobotAction, RobotState
from unirobot_infra.core.units import deg_to_rad, rad_to_deg
from unirobot_infra.envs.builder import build_base_env
from unirobot_infra.robots.arm.tianji_tcp_pose import TianjiIkSolver
from unirobot_infra.teleop.mapping.tianji_kinematics import TianjiKinematics
from unirobot_infra.utils.logging import get_logger, setup_logging
from unirobot_infra.utils.sdk_output_router import start_sdk_output_router_from_config


CONFIG_PATH = str(Path(__file__).resolve().parents[4] / "configs")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="apps/robot_ops/move_tianji_tcp_local/default")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.logging.level)
    logger = get_logger(__name__)
    with start_sdk_output_router_from_config(cfg.app.sdk_output) as sdk_output_router:
        if sdk_output_router is not None:
            logger.info("SDK 输出重定向目录=%s", sdk_output_router.run_dir)
        side = str(cfg.tcp_local_move.side)
        axis_name, axis_sign = _parse_axis(str(cfg.tcp_local_move.axis))
        distance_mm = axis_sign * float(cfg.tcp_local_move.distance_mm)
        logger.info(
            "move Tianji TCP local side=%s axis=%s distance_mm=%.3f",
            side,
            str(cfg.tcp_local_move.axis),
            float(cfg.tcp_local_move.distance_mm),
        )

        env = build_base_env(cfg)
        try:
            _, info = env.reset()
            robot_state = _robot_state_by_side(info["robot_states"], side)
            action = _build_tcp_local_move_action(cfg, robot_state, axis_name, distance_mm)
            base_action = env.action_layout.hold_action_from_states(tuple(info["robot_states"]))
            send_action = env.action_layout.overlay_actions(base_action, (action,))
            env.step(send_action)
            logger.info(
                "sent side=%s axis=%s target_q_deg=%s",
                side,
                str(cfg.tcp_local_move.axis),
                _format_q_deg(tuple(rad_to_deg(value) for value in action.arm_q_rad)),
            )
        finally:
            env.close()


def _build_tcp_local_move_action(
    cfg: DictConfig,
    robot_state: RobotState,
    axis_name: str,
    distance_mm: float,
) -> RobotAction:
    if robot_state.arm_joint is None:
        raise RuntimeError("Tianji TCP local move requires arm_joint state")
    side = robot_state.side
    arm_id = str(cfg.robot.hardware.arm.arms[side].arm_id)
    q_deg = _rad_tuple_to_deg7(robot_state.arm_joint.q_rad)
    kinematics = TianjiKinematics(cfg.robot.hardware.arm, arm_id=arm_id)
    kinematics.initialize()
    current_xyzabc = kinematics.fk_xyzabc_mm_deg(q_deg)
    current_pose = kinematics.xyzabc_to_mat4x4(current_xyzabc)
    target_pose = _translate_pose_along_local_axis(current_pose, axis_name, distance_mm)
    target_xyzabc = kinematics.mat4x4_to_xyzabc_mm_deg(target_pose)
    solver = TianjiIkSolver(
        kinematics=kinematics,
        max_joint_step_deg=float(cfg.tcp_local_move.max_joint_step_deg),
        joint_limit_mode=str(cfg.tcp_local_move.joint_limit_mode),
    )
    target_q_deg = solver.solve(
        target_xyzabc_mm_deg=target_xyzabc,
        reference_q_deg=q_deg,
        last_q_deg=q_deg,
    )
    return RobotAction(
        side=side,
        timestamp_ns=robot_state.timestamp_ns,
        arm_q_rad=tuple(deg_to_rad(value) for value in target_q_deg),
    )


def _translate_pose_along_local_axis(
    pose_mat: tuple[tuple[float, float, float, float], ...],
    axis_name: str,
    distance_mm: float,
) -> tuple[tuple[float, float, float, float], ...]:
    axis_index = _axis_index(axis_name)
    local_axis_in_base = (
        pose_mat[0][axis_index],
        pose_mat[1][axis_index],
        pose_mat[2][axis_index],
    )
    rows = [list(row) for row in pose_mat]
    rows[0][3] = rows[0][3] + local_axis_in_base[0] * distance_mm
    rows[1][3] = rows[1][3] + local_axis_in_base[1] * distance_mm
    rows[2][3] = rows[2][3] + local_axis_in_base[2] * distance_mm
    return (
        (rows[0][0], rows[0][1], rows[0][2], rows[0][3]),
        (rows[1][0], rows[1][1], rows[1][2], rows[1][3]),
        (rows[2][0], rows[2][1], rows[2][2], rows[2][3]),
        (rows[3][0], rows[3][1], rows[3][2], rows[3][3]),
    )


def _parse_axis(axis: str) -> tuple[str, float]:
    axis_text = axis.strip().lower()
    if axis_text.startswith("+"):
        axis_text = axis_text[1:]
    if axis_text in {"x", "xx"}:
        return "x", 1.0
    if axis_text in {"y", "yy"}:
        return "y", 1.0
    if axis_text in {"z", "zz"}:
        return "z", 1.0
    if axis_text in {"-x", "-xx"}:
        return "x", -1.0
    if axis_text in {"-y", "-yy"}:
        return "y", -1.0
    if axis_text in {"-z", "-zz"}:
        return "z", -1.0
    raise ValueError("tcp_local_move.axis must be x, y, z, -x, -y, or -z")


def _axis_index(axis_name: str) -> int:
    if axis_name == "x":
        return 0
    if axis_name == "y":
        return 1
    if axis_name == "z":
        return 2
    raise ValueError("axis_name must be x, y, or z")


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


def _format_q_deg(q_deg: tuple[float, ...]) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in q_deg) + "]"


if __name__ == "__main__":
    main()
