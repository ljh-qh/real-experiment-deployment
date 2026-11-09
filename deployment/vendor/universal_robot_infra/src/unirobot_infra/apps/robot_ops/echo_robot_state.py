from pathlib import Path
import time

import hydra
from omegaconf import DictConfig

from unirobot_infra.core.units import rad_to_deg
from unirobot_infra.robots.arm.tianji import TianjiRobot
from unirobot_infra.utils.logging import get_logger, setup_logging
from unirobot_infra.utils.sdk_output_router import start_sdk_output_router_from_config


CONFIG_PATH = str(Path(__file__).resolve().parents[4] / "configs")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="apps/robot_ops/echo_robot_state/default")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.logging.level)
    logger = get_logger(__name__)
    with start_sdk_output_router_from_config(cfg.app.sdk_output) as sdk_output_router:
        if sdk_output_router is not None:
            logger.info("SDK 输出重定向目录=%s", sdk_output_router.run_dir)
        robot = TianjiRobot(cfg.robot.hardware.arm)
        logger.info(
            "echo robot state robot_ip=%s clear_errors=%s",
            cfg.robot.hardware.arm.robot_ip,
            bool(cfg.robot_state.clear_errors),
        )
        robot.connect()
        try:
            arm_ids = tuple(_arm_id(cfg, side) for side in cfg.robot_state.sides)
            if bool(cfg.robot_state.clear_errors):
                robot.clear_errors(arm_ids)
            time.sleep(float(cfg.robot_state.read_settle_s))
            for side in cfg.robot_state.sides:
                arm_id = _arm_id(cfg, side)
                state = robot.read_arm_state(arm_id, side)
                q_deg = tuple(rad_to_deg(value) for value in state.joint_pos)
                logger.info(
                    "%s arm=%s q_deg=%s cur_state=%s cmd_state=%s err_code=%s frame_serial=%s",
                    side,
                    arm_id,
                    _format_q(q_deg),
                    state.cur_state,
                    state.cmd_state,
                    state.err_code,
                    state.frame_serial,
                )
        finally:
            robot.close()


def _arm_id(cfg: DictConfig, side: str) -> str:
    return str(cfg.robot.hardware.arm.arms[side].arm_id)


def _format_q(q_deg: tuple[float, ...]) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in q_deg) + "]"


if __name__ == "__main__":
    main()
