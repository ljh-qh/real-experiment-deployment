from pathlib import Path

import hydra
from omegaconf import DictConfig

from unirobot_infra.apps.demo.session import DualGraspDemoSession, task_cfg
from unirobot_infra.apps.demo.skills import skill_task
from unirobot_infra.utils.logging import get_logger, setup_logging
from unirobot_infra.utils.sdk_output_router import start_sdk_output_router_from_config


CONFIG_PATH = str(Path(__file__).resolve().parents[4] / "configs")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="apps/demo/dual_grasp_cup_bottle")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.logging.level)
    logger = get_logger(__name__)
    with start_sdk_output_router_from_config(cfg.app.sdk_output) as sdk_output_router:
        if sdk_output_router is not None:
            logger.info("SDK 输出重定向目录=%s", sdk_output_router.run_dir)
        run_dual_grasp_cup_bottle(cfg, logger)


def run_dual_grasp_cup_bottle(cfg: DictConfig, logger) -> None:
    with DualGraspDemoSession.from_config(cfg, logger) as demo:
        demo.reset.to_initial_state()
        demo.runtimes.start()
        run_dual_grasp_cup_bottle_episode(demo, logger)
        demo.reset.to_initial_state()


def run_dual_grasp_cup_bottle_episode(demo: DualGraspDemoSession, logger) -> None:
    logger.info("dual grasp live perception start sides=left,right")
    with demo.live_targets(("left", "right")) as target_source:
        target_poses = demo.together("move to live targets").run(
            [
                skill_task("left_arm", lambda: demo.motion.left.move_to_live_target(target_source)),
                skill_task("right_arm", lambda: demo.motion.right.move_to_live_target(target_source)),
            ]
        )

        demo.together("adaptive grasp").run(
            [
                skill_task("left_hand", demo.grasp.left.adaptive),
                skill_task("right_hand", demo.grasp.right.adaptive),
            ]
        )

        demo.together("translate tcp world after grasp").run(
            [
                skill_task(
                    "left_arm",
                    lambda: demo.motion.left.translate_after_grasp(target_poses["left_arm"]),
                ),
                skill_task(
                    "right_arm",
                    lambda: demo.motion.right.translate_after_grasp(target_poses["right_arm"]),
                ),
            ]
        )

        demo.replay.tianji_trajectory()

        demo.together("move to post replay anchors").run(demo.motion.post_replay_anchor_tasks())

        demo.together("release to middle").run(
            [
                skill_task("left_hand", demo.grasp.left.release_to_middle),
                skill_task("right_hand", demo.grasp.right.release_to_middle),
            ]
        )

        demo.together("post release clearance and release full").run(
            [
                skill_task(
                    "left_arm_clearance",
                    lambda: demo.motion.left.post_release_clearance(hold_grasp=False),
                ),
                skill_task(
                    "right_arm_clearance",
                    lambda: demo.motion.right.post_release_clearance(hold_grasp=False),
                ),
                skill_task("left_hand_release_full", demo.grasp.left.release_full),
                skill_task("right_hand_release_full", demo.grasp.right.release_full),
            ]
        )


def _task_cfg(cfg: DictConfig, side: str) -> DictConfig:
    return task_cfg(cfg, side)


if __name__ == "__main__":
    main()
