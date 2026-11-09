from collections.abc import Sequence
from pathlib import Path
import time

import hydra
import yaml
from omegaconf import DictConfig

from unirobot_infra.core.time import now_ns
from unirobot_infra.core.units import deg_to_rad
from unirobot_infra.robots.arm.tianji import TianjiRobot
from unirobot_infra.teleop.mapping.tianji_kinematics import TianjiKinematics
from unirobot_infra.utils.logging import get_logger, setup_logging
from unirobot_infra.utils.sdk_output_router import start_sdk_output_router_from_config


CONFIG_PATH = str(Path(__file__).resolve().parents[4] / "configs")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="apps/robot_ops/capture_tianji_arm_anchor/left")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.logging.level)
    logger = get_logger(__name__)
    with start_sdk_output_router_from_config(cfg.app.sdk_output) as sdk_output_router:
        if sdk_output_router is not None:
            logger.info("SDK 输出重定向目录=%s", sdk_output_router.run_dir)
        run_capture_tianji_arm_anchor(cfg, logger)


def run_capture_tianji_arm_anchor(cfg: DictConfig, logger) -> None:
    side = str(cfg.arm_anchor_capture.side)
    _validate_capture_config(cfg, side)
    robot = TianjiRobot(cfg.robot.hardware.arm)
    output_path = Path(str(cfg.arm_anchor_capture.output_path))
    arm_id = str(cfg.robot.hardware.arm.arms[side].arm_id)
    try:
        robot.connect()
        if bool(cfg.arm_anchor_capture.clear_errors):
            robot.clear_errors((arm_id,))
        time.sleep(float(cfg.arm_anchor_capture.read_settle_s))
        q_deg = robot.read_joint_position_deg(arm_id)
        status = robot.read_arm_status(arm_id)
        anchor = _anchor_record(cfg, side, arm_id, q_deg, status)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(anchor, file, sort_keys=False, allow_unicode=True)
        logger.info(
            "Tianji arm anchor saved side=%s arm=%s label=%s output=%s q_deg=%s",
            side,
            arm_id,
            cfg.arm_anchor_capture.label,
            output_path,
            _format_values(q_deg),
        )
    finally:
        robot.close()


def _validate_capture_config(cfg: DictConfig, side: str) -> None:
    if side not in cfg.robot.sides:
        raise ValueError(f"arm_anchor_capture.side={side} not found in robot.sides")
    if "arm" not in cfg.robot.sides[side].parts:
        raise ValueError(f"arm_anchor_capture.side={side} must have arm part")
    if float(cfg.arm_anchor_capture.read_settle_s) < 0.0:
        raise ValueError("arm_anchor_capture.read_settle_s must be non-negative")


def _anchor_record(cfg: DictConfig, side: str, arm_id: str, q_deg: Sequence[float], status) -> dict:
    q_deg_tuple = _checked_q_deg(q_deg)
    q_rad = tuple(deg_to_rad(value) for value in q_deg_tuple)
    kinematics = TianjiKinematics(cfg.robot.hardware.arm, arm_id=arm_id)
    kinematics.initialize()
    xyzabc = kinematics.fk_xyzabc_mm_deg(q_deg_tuple)
    pose_mat = kinematics.xyzabc_to_mat4x4(xyzabc)
    return {
        "type": "tianji_arm_anchor",
        "label": str(cfg.arm_anchor_capture.label),
        "side": side,
        "arm_id": arm_id,
        "joint_order": [str(joint.name) for joint in cfg.robot.hardware.arm.joints],
        "anchor": {
            "timestamp_ns": now_ns(),
            "source": "current_robot_state",
            "q_deg": [float(value) for value in q_deg_tuple],
            "q_rad": [float(value) for value in q_rad],
            "tcp_xyzabc_mm_deg": [float(value) for value in xyzabc],
            "tcp_xyz_m": [float(xyzabc[0]) * 0.001, float(xyzabc[1]) * 0.001, float(xyzabc[2]) * 0.001],
            "tcp_rotation_base_from_tcp": _rotation_from_pose_mat(pose_mat),
            "status": {
                "cur_state": int(status.cur_state),
                "cmd_state": int(status.cmd_state),
                "err_code": int(status.err_code),
            },
        },
    }


def _checked_q_deg(q_deg: Sequence[float]) -> tuple[float, float, float, float, float, float, float]:
    q = tuple(float(value) for value in q_deg)
    if len(q) != 7:
        raise ValueError("Tianji arm anchor q_deg must have length 7")
    return (q[0], q[1], q[2], q[3], q[4], q[5], q[6])


def _rotation_from_pose_mat(pose_mat: Sequence[Sequence[float]]) -> list[list[float]]:
    rows = tuple(tuple(float(value) for value in row) for row in pose_mat)
    if len(rows) != 4:
        raise ValueError("pose_mat must have 4 rows")
    if any(len(row) != 4 for row in rows):
        raise ValueError("pose_mat must have 4 columns")
    return [
        [rows[0][0], rows[0][1], rows[0][2]],
        [rows[1][0], rows[1][1], rows[1][2]],
        [rows[2][0], rows[2][1], rows[2][2]],
    ]


def _format_values(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(value):.6f}" for value in values) + "]"


if __name__ == "__main__":
    main()
