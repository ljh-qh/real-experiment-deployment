from pathlib import Path

import hydra
from omegaconf import DictConfig

from unirobot_infra.calibration.hand_eye import (
    CheckerboardConfig,
    format_transform_report,
    load_calibration_samples,
    save_transform,
    solve_hand_eye,
    transform_output_stem,
)
from unirobot_infra.utils.logging import get_logger, setup_logging


CONFIG_PATH = str(Path(__file__).resolve().parents[4] / "configs")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="apps/calibration/solve")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.logging.level)
    logger = get_logger(__name__)
    result = solve_calibration_from_config(cfg)
    logger.info("\n%s", format_transform_report(result))


def solve_calibration_from_config(cfg: DictConfig):
    checkerboard = CheckerboardConfig(
        cols=int(cfg.calibration.checkerboard.cols),
        rows=int(cfg.calibration.checkerboard.rows),
        square_size_m=float(cfg.calibration.checkerboard.square_size_m),
    )
    intrinsics_path = None if cfg.calibration.intrinsics_path is None else Path(str(cfg.calibration.intrinsics_path))
    samples, _, camera_name = load_calibration_samples(
        data_dir=cfg.calibration.data_dir,
        camera_name=None if cfg.calibration.camera is None else str(cfg.calibration.camera),
        checkerboard=checkerboard,
        intrinsics_path=intrinsics_path,
    )
    result = solve_hand_eye(
        samples=samples,
        mode=str(cfg.calibration.mode),
        method=str(cfg.calibration.method),
    )
    output_stem = Path(cfg.calibration.data_dir) / transform_output_stem(str(cfg.calibration.mode))
    save_transform(output_stem, result.transform)
    get_logger(__name__).info(
        "标定求解完成 camera=%s mode=%s samples=%s output=%s",
        camera_name,
        cfg.calibration.mode,
        result.samples_used,
        output_stem,
    )
    return result


if __name__ == "__main__":
    main()
