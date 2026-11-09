"""Inspect timestamp alignment during episode resampling."""

from __future__ import annotations

import argparse
from pathlib import Path

from unirobot_infra.dataset.convert_config import ConvertConfig
from unirobot_infra.dataset.resampling_timeline import write_resampling_timeline_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect timestamp alignment during episode resampling")
    parser.add_argument("--episode-dir", type=Path, required=True, help="Path to episode_XXX directory")
    parser.add_argument("--target-fps", type=int, required=True, help="Target fps used by conversion")
    parser.add_argument("--config", type=str, default=None, help="Optional convert config YAML")
    parser.add_argument(
        "--include-action",
        action="store_true",
        help="Include action timestamps when no config is provided",
    )
    args = parser.parse_args()

    if args.config is None:
        output_path = write_resampling_timeline_html(
            args.episode_dir,
            args.target_fps,
            include_action=args.include_action,
        )
    else:
        config = ConvertConfig.from_yaml(args.config)
        output_path = write_resampling_timeline_html(
            args.episode_dir,
            args.target_fps,
            state_keys=config.state_keys,
            camera_names=config.camera_source_names,
            tactile_images=config.tactile_image_specs,
            tactile_force=config.tactile_force,
            include_action=config.action.source in {"sent_action", "policy_action"},
        )
    print(f"Saved timeline report to {output_path}")


if __name__ == "__main__":
    main()
