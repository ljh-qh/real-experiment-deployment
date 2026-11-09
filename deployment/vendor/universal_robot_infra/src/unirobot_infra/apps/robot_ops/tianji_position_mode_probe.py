from pathlib import Path
import time

import hydra
from omegaconf import DictConfig

from unirobot_infra.core.units import rad_to_deg
from unirobot_infra.robots.arm.tianji import TianjiRobot
from unirobot_infra.utils.logging import get_logger, setup_logging
from unirobot_infra.utils.sdk_output_router import start_sdk_output_router_from_config


CONFIG_PATH = str(Path(__file__).resolve().parents[4] / "configs")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="apps/robot_ops/tianji_position_mode_probe/default")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.logging.level)
    logger = get_logger(__name__)
    with start_sdk_output_router_from_config(cfg.app.sdk_output) as sdk_output_router:
        if sdk_output_router is not None:
            logger.info("SDK 输出重定向目录=%s", sdk_output_router.run_dir)
        robot = TianjiRobot(cfg.robot.hardware.arm)
        logger.info(
            "tianji position mode probe robot_ip=%s clear_errors=%s sides=%s",
            cfg.robot.hardware.arm.robot_ip,
            bool(cfg.probe.clear_errors),
            ",".join(cfg.probe.sides),
        )
        robot.connect()
        try:
            arm_ids = tuple(_arm_id(cfg, side) for side in cfg.probe.sides)
            if bool(cfg.probe.clear_errors):
                robot.clear_errors(arm_ids)
            time.sleep(float(cfg.probe.read_settle_s))
            _log_states(logger, robot, cfg, "before")
            if cfg.probe.sequence == "read_only":
                return
            elif cfg.probe.sequence == "default":
                robot.enter_position_modes(arm_ids)
            elif cfg.probe.sequence == "vel_then_state":
                _enter_position_modes_vel_then_state(robot, cfg, arm_ids)
            elif cfg.probe.sequence == "state_vel":
                _enter_position_modes_state_vel(robot, cfg, arm_ids)
            else:
                raise ValueError(f"unknown Tianji position mode probe sequence: {cfg.probe.sequence}")
            time.sleep(float(cfg.probe.read_settle_s))
            _log_states(logger, robot, cfg, "after")
        finally:
            robot.close()


def _log_states(logger, robot: TianjiRobot, cfg: DictConfig, label: str) -> None:
    for side in cfg.probe.sides:
        arm_id = _arm_id(cfg, side)
        state = robot.read_arm_state(arm_id, side)
        q_deg = tuple(rad_to_deg(value) for value in state.joint_pos)
        q_cmd_deg = tuple(rad_to_deg(value) for value in state.joint_cmd_pos)
        logger.info(
            "%s %s arm=%s q_deg=%s q_cmd_deg=%s c_toq=%s s_toq=%s est_force=%s "
            "cur_state=%s cmd_state=%s err_code=%s frame_serial=%s",
            label,
            side,
            arm_id,
            _format_q(q_deg),
            _format_q(q_cmd_deg),
            _format_q(state.joint_current),
            _format_q(state.joint_torque),
            _format_q(state.est_joint_force),
            state.cur_state,
            state.cmd_state,
            state.err_code,
            state.frame_serial,
        )
        if cfg.probe.log_servo_errors:
            servo_errors = robot._robot.get_servo_error_code(arm_id, lang=str(cfg.probe.servo_error_lang))
            logger.info("%s %s arm=%s servo_errors=%s", label, side, arm_id, servo_errors)


def _arm_id(cfg: DictConfig, side: str) -> str:
    return str(cfg.robot.hardware.arm.arms[side].arm_id)


def _enter_position_modes_vel_then_state(robot: TianjiRobot, cfg: DictConfig, arm_ids: tuple[str, ...]) -> None:
    robot._robot.clear_set()
    for arm_id in arm_ids:
        vel_acc_ok = robot._robot.set_vel_acc(
            arm=arm_id,
            velRatio=int(cfg.robot.hardware.arm.vel_ratio),
            AccRatio=int(cfg.robot.hardware.arm.acc_ratio),
        )
        if not vel_acc_ok:
            raise RuntimeError(f"Tianji set_vel_acc failed for arm {arm_id}")
    send_ok = robot._robot.send_cmd()
    if not send_ok:
        raise RuntimeError("Tianji send_cmd vel_acc failed")
    time.sleep(float(cfg.robot.hardware.arm.position_mode_settle_s))

    robot._robot.clear_set()
    for arm_id in arm_ids:
        state_ok = robot._robot.set_state(arm=arm_id, state=1)
        if not state_ok:
            raise RuntimeError(f"Tianji set_state position failed for arm {arm_id}")
    send_ok = robot._robot.send_cmd()
    if not send_ok:
        raise RuntimeError("Tianji send_cmd position mode failed")
    time.sleep(float(cfg.robot.hardware.arm.position_mode_settle_s))


def _enter_position_modes_state_vel(robot: TianjiRobot, cfg: DictConfig, arm_ids: tuple[str, ...]) -> None:
    robot._robot.clear_set()
    for arm_id in arm_ids:
        state_ok = robot._robot.set_state(arm=arm_id, state=1)
        if not state_ok:
            raise RuntimeError(f"Tianji set_state position failed for arm {arm_id}")
        vel_acc_ok = robot._robot.set_vel_acc(
            arm=arm_id,
            velRatio=int(cfg.probe.state_vel_ratio),
            AccRatio=int(cfg.probe.state_acc_ratio),
        )
        if not vel_acc_ok:
            raise RuntimeError(f"Tianji set_vel_acc failed for arm {arm_id}")
    send_ok = robot._robot.send_cmd()
    if not send_ok:
        raise RuntimeError("Tianji send_cmd position mode failed")
    time.sleep(float(cfg.robot.hardware.arm.position_mode_settle_s))


def _format_q(q_deg: tuple[float, ...]) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in q_deg) + "]"


if __name__ == "__main__":
    main()
