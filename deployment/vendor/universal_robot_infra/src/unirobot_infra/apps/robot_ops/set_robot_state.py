from pathlib import Path

import hydra
from omegaconf import DictConfig

from unirobot_infra.envs.builder import build_base_env
from unirobot_infra.envs.layout import RobotActionLayout
from unirobot_infra.envs.target_action import (
    sent_actions_summary,
    target_summary,
)
from unirobot_infra.utils.logging import get_logger, setup_logging
from unirobot_infra.utils.sdk_output_router import start_sdk_output_router_from_config


CONFIG_PATH = str(Path(__file__).resolve().parents[4] / "configs")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="apps/robot_ops/set_robot_state/default")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.logging.level)
    logger = get_logger(__name__)
    with start_sdk_output_router_from_config(cfg.app.sdk_output) as sdk_output_router:
        if sdk_output_router is not None:
            logger.info("SDK 输出重定向目录=%s", sdk_output_router.run_dir)
        layout = RobotActionLayout.from_robot_config(cfg.robot)
        logger.info(
            "set_robot_state 启动 robot=%s target=%s",
            cfg.robot.name,
            target_summary(cfg.env.reset.targets, layout.segments),
        )

        env = build_base_env(cfg)
        try:
            _, info = env.reset()
            logger.info("已发送 robot state sent=%s", sent_actions_summary(info["sent_actions"]))
        finally:
            env.close()


if __name__ == "__main__":
    main()
