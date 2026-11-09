from collections import OrderedDict
import multiprocessing as mp
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from unirobot_infra.apps.demo.dual_grasp_cup_bottle import run_dual_grasp_cup_bottle_episode
from unirobot_infra.apps.demo.session import DualGraspDemoSession
from unirobot_infra.camera.manager import CameraManager
from unirobot_infra.camera.shared_zed import SharedZedCameraManager, SharedZedRgbdFrameBus
from unirobot_infra.envs.builder import build_robot_group
from unirobot_infra.recorder import DataRecorder
from unirobot_infra.utils.logging import get_logger, setup_logging
from unirobot_infra.utils.sdk_output_router import start_sdk_output_router_from_config


CONFIG_PATH = str(Path(__file__).resolve().parents[3] / "configs")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="apps/demo_record/dual_grasp_cup_bottle")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.logging.level)
    logger = get_logger(__name__)
    with start_sdk_output_router_from_config(cfg.app.sdk_output) as sdk_output_router:
        if sdk_output_router is not None:
            logger.info("SDK 输出重定向目录=%s", sdk_output_router.run_dir)
        run_demo_record(cfg, logger)


def run_demo_record(cfg: DictConfig, logger) -> None:
    robot_group = build_robot_group(cfg)
    camera_manager, live_target_rgbd_frame_bus = _build_demo_record_camera_manager(cfg, logger)
    recorder = None
    try:
        recorder = DataRecorder(
            cfg.recording,
            robot_group=robot_group,
            cameras=cfg.camera.cameras,
            camera_manager=camera_manager,
        )
        with DualGraspDemoSession.from_config(
            cfg,
            logger,
            robot_group=robot_group,
            live_target_rgbd_frame_bus=live_target_rgbd_frame_bus,
        ) as demo:
            logger.info("demo_record pre-episode initial pose reset start")
            demo.reset.to_initial_state()
            logger.info("demo_record pre-episode initial pose reset done")
            episode_dir = recorder.start_episode()
            logger.info("demo_record episode started: %s", episode_dir)
            try:
                demo.runtimes.start()
                run_dual_grasp_cup_bottle_episode(demo, logger)
            except BaseException:
                if recorder.recording:
                    recorder.discard_episode()
                raise
            else:
                recorder.stop_episode()
                logger.info("demo_record episode save queued: %s", episode_dir)
            logger.info("demo_record post-episode initial pose reset start")
            demo.reset.to_initial_state()
            logger.info("demo_record post-episode initial pose reset done")
    finally:
        if recorder is not None:
            recorder.close()
        elif camera_manager is not None:
            camera_manager.close()


class _CompositeCameraManager:
    def __init__(self, managers) -> None:
        self.managers = tuple(managers)
        self.captures = OrderedDict()
        for manager in self.managers:
            for name in manager.captures:
                if name in self.captures:
                    raise ValueError(f"duplicate camera name in demo_record camera managers: {name}")
                self.captures[name] = manager.captures[name]

    def close(self) -> None:
        for manager in reversed(self.managers):
            manager.close()
        self.captures.clear()


def _build_demo_record_camera_manager(cfg: DictConfig, logger):
    camera_names = tuple(str(name) for name in cfg.recording.camera_names)
    if not camera_names:
        return None, None
    live_camera_name = str(cfg.demo.camera_name)
    if live_camera_name not in camera_names:
        return None, None
    live_camera_cfg = cfg.camera.cameras[live_camera_name]
    if live_camera_cfg.type != "zed_sdk":
        return None, None

    ctx = mp.get_context("spawn")
    rgbd_frame_bus = SharedZedRgbdFrameBus(
        queue=ctx.Queue(maxsize=1),
        enabled_event=ctx.Event(),
    )
    rgbd_frame_bus.queue.cancel_join_thread()
    shared_zed_manager = SharedZedCameraManager(
        camera_name=live_camera_name,
        camera_cfg=live_camera_cfg,
        rgbd_frame_bus=rgbd_frame_bus,
        rgbd_target_hz=float(cfg.demo.perception_rate_hz),
    )
    managers = [shared_zed_manager]
    other_camera_names = tuple(name for name in camera_names if name != live_camera_name)
    if other_camera_names:
        other_camera_cfg = OmegaConf.create({name: cfg.camera.cameras[name] for name in other_camera_names})
        managers.append(CameraManager(other_camera_cfg))
    logger.info(
        "demo_record 使用共享 ZED 相机 name=%s 同时供 recorder RGB 和 live perception RGB-D",
        live_camera_name,
    )
    return _CompositeCameraManager(managers), rgbd_frame_bus


if __name__ == "__main__":
    main()
