"""Push a local LeRobot dataset to HuggingFace Hub."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    from huggingface_hub import HfApi
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    parser = argparse.ArgumentParser(description="Push a local LeRobot dataset to HuggingFace Hub")
    parser.add_argument("--repo-id", type=str, required=True, help="HuggingFace repo ID")
    parser.add_argument("--root", type=str, required=True, help="Local dataset directory")
    parser.add_argument("--private", action="store_true", help="Make the repo private")
    parser.add_argument("--tags", type=str, nargs="*", default=["tianji", "sharpa"], help="Dataset tags")
    args = parser.parse_args()

    root = Path(args.root)
    dataset = LeRobotDataset(repo_id=args.repo_id, root=root)
    print(f"Loaded dataset: {dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames")
    dataset.push_to_hub(tags=args.tags, private=args.private, push_videos=True)

    readme_path = root / "README.md"
    if readme_path.exists():
        HfApi().upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        print(f"Uploaded local README.md from {readme_path}")
    else:
        print("Local README.md not found; keeping LeRobot-generated dataset card")
    print(f"Pushed to: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
