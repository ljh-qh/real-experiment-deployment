"""Convert unirobot_infra recorded episodes to LeRobot dataset format."""

from __future__ import annotations

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

from unirobot_infra.dataset.convert_config import ConvertConfig
from unirobot_infra.utils.logging import setup_logging


CONFIG_PATH = str(Path(__file__).resolve().parents[4] / "configs")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="apps/postprocess/convert/default")
def main(cfg: DictConfig) -> None:
    console = Console(stderr=True)
    setup_logging(cfg.logging.level, console=console)
    convert_config = ConvertConfig.from_config(OmegaConf.to_container(cfg.convert, resolve=True))

    if bool(cfg.app.update_readme_only):
        if bool(cfg.app.push):
            raise ValueError("app.push is not supported together with app.update_readme_only")
        from unirobot_infra.dataset.dataset_card import update_lerobot_readme

        update_lerobot_readme(
            output_dir=cfg.app.output_dir,
            config=convert_config,
            task_name=cfg.app.task,
            language_instruction=cfg.app.language_instruction,
            config_path=_config_name(),
            raw_data_dir=cfg.app.data_dir,
        )
        return

    from unirobot_infra.dataset.lerobot_converter import convert_to_lerobot

    robot_cfg = None
    if (
        convert_config.state.type in {"tcp_pose_world_with_hand_alpha", "tcp_pose_world_with_hand_joint"}
        or convert_config.action.type in {"delta_ee_world_with_hand_alpha", "delta_ee_world_with_hand_joint"}
    ):
        robot_cfg = cfg.robot
    convert_to_lerobot(
        data_dir=cfg.app.data_dir,
        output_dir=cfg.app.output_dir,
        config=convert_config,
        task_name=cfg.app.task,
        language_instruction=cfg.app.language_instruction,
        push_to_hub=bool(cfg.app.push),
        inspect_timestamps=bool(cfg.app.inspect_timestamps),
        config_path=_config_name(),
        overwrite=bool(cfg.app.overwrite),
        robot_cfg=robot_cfg,
    )


def _config_name() -> str:
    return str(HydraConfig.get().job.config_name)


if __name__ == "__main__":
    main()
