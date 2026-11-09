"""Regenerate README.md for an existing LeRobot dataset."""

from __future__ import annotations

import argparse

from unirobot_infra.dataset.convert_config import ConvertConfig
from unirobot_infra.dataset.dataset_card import update_lerobot_readme


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate a local LeRobot dataset card")
    parser.add_argument("--config", type=str, required=True, help="Path to convert config YAML")
    parser.add_argument("--output-dir", type=str, required=True, help="Existing LeRobot dataset root")
    parser.add_argument("--task", type=str, default=None, help="Optional short task name for README.md")
    parser.add_argument(
        "--language-instruction",
        type=str,
        default=None,
        help="Optional language instruction for README.md; defaults to the dataset tasks metadata",
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Optional raw data directory for README.md")
    args = parser.parse_args()

    update_lerobot_readme(
        output_dir=args.output_dir,
        config=ConvertConfig.from_yaml(args.config),
        task_name=args.task,
        language_instruction=args.language_instruction,
        config_path=args.config,
        raw_data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
