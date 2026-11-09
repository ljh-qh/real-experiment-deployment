from pathlib import Path

import hydra
from omegaconf import DictConfig

from unirobot_infra.apps.demo.session import SingleLiveGraspDemoSession, single_release_clearance_tasks
from unirobot_infra.utils.logging import get_logger, setup_logging
from unirobot_infra.utils.sdk_output_router import start_sdk_output_router_from_config


CONFIG_PATH = str(Path(__file__).resolve().parents[4] / "configs")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="apps/demo/left_grasp_cup")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.logging.level)
    logger = get_logger(__name__)
    with start_sdk_output_router_from_config(cfg.app.sdk_output) as sdk_output_router:
        if sdk_output_router is not None:
            logger.info("SDK 输出重定向目录=%s", sdk_output_router.run_dir)
        run_left_grasp_cup(cfg, logger)


def run_left_grasp_cup(cfg: DictConfig, logger) -> None:
    with SingleLiveGraspDemoSession.from_config(cfg, logger) as demo:
        with demo.live_targets() as target_source:
            target_pose = demo.motion.move_to_live_target(target_source)
            demo.grasp.adaptive()
            demo.motion.translate_after_grasp(target_pose)
            demo.grasp.release_to_middle()
            demo.together("post release clearance and release full").run(
                single_release_clearance_tasks(demo)
            )
            demo.runtime.reset_robot()


if __name__ == "__main__":
    main()
