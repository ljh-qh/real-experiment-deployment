from pathlib import Path
from threading import Lock
import time

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

from unirobot_infra.apps.teleop import _teleop_usage_text
from unirobot_infra.calibration.sample_io import (
    build_tianji_calibration_record,
    tianji_tcp_matrix_m,
    write_pose_records,
)
from unirobot_infra.camera import CameraManager
from unirobot_infra.core.types import RobotState
from unirobot_infra.core.units import rad_to_deg
from unirobot_infra.teleop.mapping.tianji_kinematics import TianjiKinematics
from unirobot_infra.utils.logging import get_logger, setup_logging
from unirobot_infra.utils.sdk_output_router import start_sdk_output_router_from_config
from unirobot_infra.wrappers.base import EnvLike
from unirobot_infra.wrappers.builder import build_env
from unirobot_infra.wrappers.image_display import ImageDisplayWrapper


CONFIG_PATH = str(Path(__file__).resolve().parents[4] / "configs")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="apps/calibration/collect")
def main(cfg: DictConfig) -> None:
    console = Console(stderr=True)
    setup_logging(cfg.logging.level, console=console)
    logger = get_logger(__name__)
    with start_sdk_output_router_from_config(cfg.app.sdk_output) as sdk_output_router:
        if sdk_output_router is not None:
            logger.info("SDK 输出重定向目录=%s", sdk_output_router.run_dir)
        collect_calibration_data(cfg, logger, console)


def collect_calibration_data(cfg: DictConfig, logger, console: Console | None = None) -> None:
    side = str(cfg.calibration.side)
    _validate_calibration_side(cfg, side)
    arm_id = str(cfg.robot.hardware.arm.arms[side].arm_id)
    output_path = Path(cfg.calibration.output_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_cfg = _select_camera_config(cfg)
    kinematics = TianjiKinematics(cfg.robot.hardware.arm, arm_id=arm_id)
    kinematics.initialize()
    manager = CameraManager(camera_cfg)
    key_listener = _CalibrationKeyListener()
    latest_camera_frames = _LatestCameraFrames()
    records: list[dict] = []
    frame_index = 0
    base_from_ee = None
    preview_next_read_s = 0.0
    _validate_camera_preview_cfg(cfg)
    logger.info(
        "标定采集启动 side=%s arm=%s cameras=%s output=%s",
        side,
        arm_id,
        ",".join(str(name) for name in camera_cfg),
        output_path,
    )
    env = build_env(cfg, console=console)
    if bool(cfg.app.camera_preview.enabled):
        env = ImageDisplayWrapper(
            env,
            cfg.app.camera_preview,
            frame_source=latest_camera_frames.read,
            camera_names=tuple(str(name) for name in camera_cfg),
        )
        logger.info("相机预览已启动 window=%s", cfg.app.camera_preview.window_name)
    try:
        _, info = env.reset()
        action = env.action_layout.hold_action_from_states(tuple(info["robot_states"]))
        _print_collect_usage(cfg, console)
        while True:
            loop_start = time.monotonic()
            if bool(cfg.app.camera_preview.enabled) and loop_start >= preview_next_read_s:
                latest_camera_frames.update(
                    manager.read(timeout_s=float(cfg.app.camera_preview.read_timeout_s))
                )
                preview_next_read_s = loop_start + float(cfg.app.camera_preview.read_interval_s)
            _, _, _, _, info = env.step(action)
            action = _hold_action_from_sent_actions(env, action, info)
            q_deg = _arm_q_deg_from_info(info, side)
            key = key_listener.pop_key()
            if key == "space":
                base_from_ee = tianji_tcp_matrix_m(kinematics, q_deg)
                images = manager.read(timeout_s=float(cfg.calibration.camera_read_timeout_s))
                latest_camera_frames.update(images)
                image_paths = _write_sample_images(output_dir, len(records), images)
                records.append(
                    build_tianji_calibration_record(
                        side=side,
                        arm_id=arm_id,
                        q_deg=q_deg,
                        base_from_ee=base_from_ee,
                        images=image_paths,
                    )
                )
                logger.info(
                    "已记录标定样本 index=%s tcp_xyz_m=(%.4f, %.4f, %.4f) q_deg=%s images=%s",
                    len(records) - 1,
                    base_from_ee[0, 3],
                    base_from_ee[1, 3],
                    base_from_ee[2, 3],
                    _format_q_deg(q_deg),
                    ",".join(image_paths),
                )
            elif key == "enter":
                break
            frame_index += 1
            _sleep_remaining(loop_start, 1.0 / float(cfg.calibration.poll_hz))
    finally:
        key_listener.close()
        manager.close()
        env.close()
    write_pose_records(output_path, records)
    logger.info("标定记录已保存 count=%s path=%s", len(records), output_path)


def _select_camera_config(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(
        {
            str(camera_name): cfg.camera.cameras[camera_name]
            for camera_name in cfg.calibration.camera_names
        }
    )


def _write_sample_images(output_dir: Path, sample_index: int, images: dict[str, object]) -> dict[str, str]:
    paths = {}
    for camera_name, rgb_image in images.items():
        filename = f"{sample_index:03d}_{camera_name}.png"
        cv2.imwrite(str(output_dir / filename), rgb_image[..., ::-1])
        paths[str(camera_name)] = filename
    return paths


def _validate_camera_preview_cfg(cfg: DictConfig) -> None:
    if not bool(cfg.app.camera_preview.enabled):
        return
    if float(cfg.app.camera_preview.read_timeout_s) <= 0.0:
        raise ValueError("app.camera_preview.read_timeout_s must be positive")
    if float(cfg.app.camera_preview.read_interval_s) <= 0.0:
        raise ValueError("app.camera_preview.read_interval_s must be positive")


def _validate_calibration_side(cfg: DictConfig, side: str) -> None:
    if side not in cfg.robot.sides:
        raise ValueError(f"calibration.side={side} not found in robot.sides")
    if "arm" not in cfg.robot.sides[side].parts:
        raise ValueError(f"calibration.side={side} must have arm part")


def _arm_q_deg_from_info(info: dict, side: str) -> tuple[float, float, float, float, float, float, float]:
    state = _robot_state_by_side(tuple(info["robot_states"]), side)
    if state.arm_joint is None:
        raise RuntimeError(f"RobotState missing arm_joint for side {side}")
    q_deg = tuple(rad_to_deg(value) for value in state.arm_joint.q_rad)
    if len(q_deg) != 7:
        raise ValueError("Tianji arm joint state must have length 7")
    return (q_deg[0], q_deg[1], q_deg[2], q_deg[3], q_deg[4], q_deg[5], q_deg[6])


def _robot_state_by_side(states: tuple[RobotState, ...], side: str) -> RobotState:
    for state in states:
        if state.side == side:
            return state
    raise ValueError(f"robot state missing side {side}")


def _hold_action_from_sent_actions(env: EnvLike, previous_action: tuple[float, ...], info: dict) -> tuple[float, ...]:
    return env.action_layout.overlay_actions(previous_action, tuple(info["sent_actions"]))


def _sleep_remaining(loop_start: float, period_s: float) -> None:
    remaining_s = period_s - (time.monotonic() - loop_start)
    if remaining_s > 0.0:
        time.sleep(remaining_s)


def _print_collect_usage(cfg: DictConfig, console: Console | None) -> None:
    usage = _teleop_usage_text(cfg)
    usage.append("\n采集: SPACE=记录  ENTER=保存退出")
    if console is not None:
        console.print(usage)
        return
    print(usage.plain, flush=True)


def _format_q_deg(q_deg: tuple[float, ...]) -> str:
    return "[" + ", ".join(f"{value:+.1f}" for value in q_deg) + "]"


class _LatestCameraFrames:
    def __init__(self) -> None:
        self._frames: dict[str, np.ndarray] = {}

    def update(self, frames: dict[str, np.ndarray]) -> None:
        self._frames = {name: frame.copy() for name, frame in frames.items()}

    def read(self) -> dict[str, np.ndarray]:
        return {name: frame.copy() for name, frame in self._frames.items()}


class _CalibrationKeyListener:
    def __init__(self) -> None:
        from pynput import keyboard

        self._keyboard = keyboard
        self._keys: list[str] = []
        self._lock = Lock()
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.start()

    def pop_key(self) -> str | None:
        with self._lock:
            if not self._keys:
                return None
            return self._keys.pop(0)

    def close(self) -> None:
        self._listener.stop()

    def _on_press(self, key) -> None:
        with self._lock:
            if key == self._keyboard.Key.space:
                self._keys.append("space")
            elif key == self._keyboard.Key.enter:
                self._keys.append("enter")


if __name__ == "__main__":
    main()
