from pathlib import Path
import time

import hydra
from omegaconf import DictConfig

from unirobot_infra.robots.arm.tianji import TianjiRobot, sdk_arm_to_index
from unirobot_infra.utils.logging import get_logger, setup_logging
from unirobot_infra.utils.sdk_output_router import start_sdk_output_router_from_config


CONFIG_PATH = str(Path(__file__).resolve().parents[4] / "configs")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="apps/robot_ops/tianji_sensor_offset/default")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.logging.level)
    logger = get_logger(__name__)
    with start_sdk_output_router_from_config(cfg.app.sdk_output) as sdk_output_router:
        if sdk_output_router is not None:
            logger.info("SDK 输出重定向目录=%s", sdk_output_router.run_dir)
        robot = TianjiRobot(cfg.robot.hardware.arm)
        logger.info(
            "tianji sensor offset robot_ip=%s clear_errors=%s apply=%s sides=%s axes=%s",
            cfg.robot.hardware.arm.robot_ip,
            bool(cfg.sensor_offset.clear_errors),
            bool(cfg.sensor_offset.apply),
            ",".join(cfg.sensor_offset.sides),
            ",".join(str(axis) for axis in cfg.sensor_offset.axes),
        )
        robot.connect()
        try:
            arm_ids = tuple(_arm_id(cfg, side) for side in cfg.sensor_offset.sides)
            if bool(cfg.sensor_offset.clear_errors):
                robot.clear_errors(arm_ids)
            time.sleep(float(cfg.sensor_offset.read_settle_s))
            _calibrate_sensor_offsets(logger, robot, cfg)
        finally:
            robot.close()


def _calibrate_sensor_offsets(logger, robot: TianjiRobot, cfg: DictConfig) -> None:
    axes = tuple(_axis(axis) for axis in cfg.sensor_offset.axes)
    for side in cfg.sensor_offset.sides:
        arm_id = _arm_id(cfg, side)
        state = robot.read_arm_state(arm_id, side)
        if state.cur_state != int(cfg.sensor_offset.required_cur_state):
            raise RuntimeError(
                f"Tianji arm {arm_id} must be reset state before sensor offset calibration: "
                f"cur_state={state.cur_state}"
            )
        arm_index = sdk_arm_to_index(arm_id)
        for axis in axes:
            sensor_k_name = f"R.A{arm_index}.L{axis}.BASIC.SensorK"
            sensor_k_ret, sensor_k = robot._robot.get_param(type="float", paraName=sensor_k_name)
            if int(sensor_k_ret) != 0:
                raise RuntimeError(f"Tianji get_param failed: {sensor_k_name} ret={sensor_k_ret}")
            if float(sensor_k) == 0.0:
                raise RuntimeError(f"Tianji SensorK is zero: {sensor_k_name}")

            sensor_dir_name = f"R.A{arm_index}.L{axis}.BASIC.SensorDir"
            sensor_dir_ret, sensor_dir = robot._robot.get_param(type="int", paraName=sensor_dir_name)
            if int(sensor_dir_ret) != 0:
                raise RuntimeError(f"Tianji get_param failed: {sensor_dir_name} ret={sensor_dir_ret}")

            offset_name = f"R.A{arm_index}.L{axis}.BASIC.SensorOffset"
            offset_ret, current_offset = robot._robot.get_param(type="int", paraName=offset_name)
            if int(offset_ret) != 0:
                raise RuntimeError(f"Tianji get_param failed: {offset_name} ret={offset_ret}")

            target_offset = _target_offset(
                cfg,
                current_offset=float(current_offset),
                sensor_k=float(sensor_k),
                sensor_dir=int(sensor_dir),
                joint_torque=float(state.joint_torque[axis]),
                est_joint_force=float(state.est_joint_force[axis]),
            )
            if cfg.sensor_offset.manual_offset is not None:
                target_offset = float(cfg.sensor_offset.manual_offset)
            logger.info(
                "sensor offset %s arm=%s axis=%d target=%s s_toq=%.6f est_force=%.6f "
                "sensor_k=%.6f sensor_dir=%d gain=%.6f current_offset=%d target_offset=%d raw_target=%.6f",
                side,
                arm_id,
                axis,
                cfg.sensor_offset.target,
                float(state.joint_torque[axis]),
                float(state.est_joint_force[axis]),
                float(sensor_k),
                int(sensor_dir),
                float(cfg.sensor_offset.gain),
                int(current_offset),
                int(target_offset),
                target_offset,
            )
            if cfg.sensor_offset.apply:
                set_ret = robot._robot.set_param(type="int", paraName=offset_name, value=target_offset)
                time.sleep(float(cfg.sensor_offset.write_settle_s))
                readback_ret, readback_offset = robot._robot.get_param(type="int", paraName=offset_name)
                if int(readback_ret) != 0:
                    raise RuntimeError(f"Tianji get_param failed after set: {offset_name} ret={readback_ret}")
                if int(readback_offset) != int(target_offset):
                    raise RuntimeError(
                        f"Tianji set_param readback mismatch: {offset_name} "
                        f"ret={set_ret} expected={int(target_offset)} actual={int(readback_offset)}"
                    )
                logger.info(
                    "sensor offset set %s arm=%s axis=%d ret=%s readback=%d",
                    side,
                    arm_id,
                    axis,
                    set_ret,
                    int(readback_offset),
                )

    if cfg.sensor_offset.apply and cfg.sensor_offset.save:
        save_ret = robot._robot.save_para_file()
        if int(save_ret) != 0:
            raise RuntimeError(f"Tianji save_para_file failed ret={save_ret}")
        logger.info("tianji sensor offsets saved")


def _arm_id(cfg: DictConfig, side: str) -> str:
    return str(cfg.robot.hardware.arm.arms[side].arm_id)


def _axis(axis) -> int:
    value = int(axis)
    if value < 0 or value > 6:
        raise ValueError(f"Tianji sensor offset axis must be in [0, 6], got {value}")
    return value


def _target_offset(
    cfg: DictConfig,
    *,
    current_offset: float,
    sensor_k: float,
    sensor_dir: int,
    joint_torque: float,
    est_joint_force: float,
) -> float:
    if cfg.sensor_offset.target == "sensor_torque_zero":
        correction = joint_torque / sensor_k
    elif cfg.sensor_offset.target == "estimated_force_zero":
        correction = est_joint_force / sensor_k
    elif cfg.sensor_offset.target == "estimated_force_match":
        correction = (joint_torque - est_joint_force) / sensor_k
    else:
        raise ValueError(f"unknown Tianji sensor offset target: {cfg.sensor_offset.target}")
    correction = correction * float(cfg.sensor_offset.gain)
    if sensor_dir == 0:
        return current_offset + correction
    if sensor_dir == 1:
        return current_offset - correction
    raise ValueError(f"Tianji SensorDir must be 0 or 1, got {sensor_dir}")


if __name__ == "__main__":
    main()
