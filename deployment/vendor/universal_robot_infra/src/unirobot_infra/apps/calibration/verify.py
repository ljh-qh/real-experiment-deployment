from pathlib import Path

import hydra
from omegaconf import DictConfig
from rich import box
from rich.console import Console
from rich.table import Table

from unirobot_infra.calibration.hand_eye import (
    CheckerboardConfig,
    VerificationStats,
    assess_verification_quality,
    default_transform_path,
    format_verification_report,
    load_calibration_samples,
    load_transform,
    verification_quality_thresholds,
    verify_hand_eye,
)
from unirobot_infra.utils.logging import get_logger, setup_logging


CONFIG_PATH = str(Path(__file__).resolve().parents[4] / "configs")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="apps/calibration/verify")
def main(cfg: DictConfig) -> None:
    console = Console(stderr=True)
    setup_logging(cfg.logging.level, console=console)
    logger = get_logger(__name__)
    stats = verify_calibration_from_config(cfg)
    logger.info("%s", format_verification_report(stats))
    console.print(_verification_quality_table(stats))


def verify_calibration_from_config(cfg: DictConfig):
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
    transform_path = (
        default_transform_path(cfg.calibration.data_dir, str(cfg.calibration.mode))
        if cfg.calibration.transform_path is None
        else Path(str(cfg.calibration.transform_path))
    )
    stats = verify_hand_eye(
        samples=samples,
        transform=load_transform(transform_path),
        mode=str(cfg.calibration.mode),
    )
    get_logger(__name__).info(
        "标定验证完成 camera=%s mode=%s transform=%s",
        camera_name,
        cfg.calibration.mode,
        transform_path,
    )
    return stats


def _verification_quality_table(stats: VerificationStats) -> Table:
    quality = assess_verification_quality(stats)
    table = Table(
        title="误差等级参考",
        box=box.SIMPLE,
        show_edge=True,
        show_lines=False,
        expand=False,
    )
    table.add_column("等级", no_wrap=True)
    table.add_column("平移 mm\n均值/最大", justify="right", no_wrap=True)
    table.add_column("旋转 deg\n均值/最大", justify="right", no_wrap=True)
    for threshold in verification_quality_thresholds():
        table.add_row(
            threshold.label,
            f"<= {threshold.mean_translation_error_mm:g}/{threshold.max_translation_error_mm:g}",
            f"<= {threshold.mean_rotation_error_deg:g}/{threshold.max_rotation_error_deg:g}",
            style="bold green" if quality.level == threshold.level else "",
        )
    table.add_row(
        "需重标",
        "> 10/25",
        "> 1/3",
        style="bold red" if quality.level == "recalibrate" else "",
    )
    return table


if __name__ == "__main__":
    main()
