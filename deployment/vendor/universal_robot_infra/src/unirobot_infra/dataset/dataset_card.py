from __future__ import annotations

from pathlib import Path

from unirobot_infra.dataset.convert_config import ConvertConfig


def _format_shape(shape: tuple[int, ...]) -> str:
    return "x".join(str(dim) for dim in shape)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _language_instruction_list(language_instruction: str | None, root: Path) -> list[str]:
    if language_instruction is not None:
        return [language_instruction]
    from lerobot.datasets.io_utils import load_tasks

    tasks = load_tasks(root)
    return [str(name) for name in tasks.index.tolist()]


def _build_dataset_description(
    config: ConvertConfig,
    info: dict,
    language_instructions: list[str],
    *,
    task_name: str | None,
    config_path: str | None,
    raw_data_dir: str | None,
) -> str:
    summary_rows = [
        ["Repo ID", f"`{config.repo_id}`"],
        ["Robot", f"`{info['robot_type']}`"],
        ["Target FPS", f"`{config.target_fps}`"],
        ["Episodes", str(info["total_episodes"])],
        ["Frames", str(info["total_frames"])],
        [
            "Language Instructions",
            "<br>".join(f"`{instruction}`" for instruction in language_instructions),
        ],
    ]
    if task_name is not None:
        summary_rows.insert(1, ["Task Name", f"`{task_name}`"])
    if config_path:
        summary_rows.append(["Convert Config", f"`{config_path}`"])
    if raw_data_dir:
        summary_rows.append(["Raw Data Dir", f"`{raw_data_dir}`"])
    if config.frame_filter.enabled:
        summary_rows.append(["Frame Filter", _frame_filter_description(config)])

    state_shape = tuple(info["features"]["observation.state"]["shape"])
    action_shape = tuple(info["features"]["action"]["shape"])
    layout_rows = [
        [
            "`observation.state`",
            f"`{_format_shape(state_shape)}`",
            "<br>".join(f"`{token}`" for token in config.state_tokens),
        ],
        [
            "`action`",
            f"`{_format_shape(action_shape)}`",
            _action_description(config),
        ],
    ]
    if config.tactile_force is not None:
        tactile_force_shape = tuple(info["features"]["observation.tactile_force"]["shape"])
        layout_rows.append(
            [
                "`observation.tactile_force`",
                f"`{_format_shape(tactile_force_shape)}`",
                _tactile_force_description(config),
            ]
        )

    camera_rows = []
    for camera_name, camera_cfg in config.cameras.items():
        crop = camera_cfg.crop
        crop_str = f"top={crop.top}, bottom={crop.bottom}, left={crop.left}, right={crop.right}"
        resize_str = "keep" if camera_cfg.resize is None else f"{camera_cfg.resize[0]}x{camera_cfg.resize[1]}"
        feature_key = f"observation.images.{camera_name}"
        shape = tuple(info["features"][feature_key]["shape"])
        camera_rows.append(
            [
                f"`{camera_name}`",
                f"`{camera_cfg.source}`",
                f"`{_format_shape(shape)}`",
                f"`{crop_str}`",
                f"`{resize_str}`",
            ]
        )

    tactile_rows = []
    for tactile_name, tactile_cfg in config.tactile_images.items():
        resize_str = "keep" if tactile_cfg.resize is None else f"{tactile_cfg.resize[0]}x{tactile_cfg.resize[1]}"
        source = f"`tactile/{tactile_cfg.side}/{tactile_cfg.finger}/{tactile_cfg.modality}`"
        feature_key = f"observation.images.{tactile_name}"
        storage = _visual_feature_storage(info["features"][feature_key])
        shape = tuple(info["features"][feature_key]["shape"])
        tactile_rows.append(
            [f"`{tactile_name}`", source, f"`{storage}`", f"`{_format_shape(shape)}`", f"`{resize_str}`"]
        )

    resampling_rows = [
        ["`robot_state/<stream>/*`", "nearest timestamp"],
        ["`camera/<source>`", "nearest timestamp"],
    ]
    if config.tactile_images:
        resampling_rows.append(["`tactile/<side>/<finger>`", "nearest timestamp"])
    if config.tactile_force is not None:
        resampling_rows.append([f"`tactile/<side>/<finger>/{config.tactile_force.field}`", "nearest timestamp"])
    if config.action.source in {"sent_action", "policy_action"}:
        resampling_rows.append(["`action/*`", "nearest timestamp"])

    sections = [
        "Converted from raw unirobot_infra recordings into LeRobot dataset format.",
        "### Summary",
        _markdown_table(["Field", "Value"], summary_rows),
        "### Observation And Action Layout",
        _markdown_table(["Feature", "Shape", "Description"], layout_rows),
        "### Cameras",
        _markdown_table(["Camera", "Source", "Output Shape", "Crop", "Resize"], camera_rows),
        "### Tactile Images",
        _markdown_table(["Feature", "Source", "Storage", "Output Shape", "Resize"], tactile_rows),
        "### Tactile Force",
        _markdown_table(["Sensor", "Source", "Output Shape"], _tactile_force_rows(config, info)),
        "### Resampling",
        _markdown_table(["Stream", "Alignment"], resampling_rows),
    ]
    return "\n\n".join(section for section in sections if section)


def _action_description(config: ConvertConfig) -> str:
    if config.action.type == "delta_ee_world_with_hand_alpha":
        return "`delta_ee_world_with_hand_alpha`: " + "<br>".join(
            f"`{side}_delta_ee_world[9]`, `{side}_hand_alpha[1]`"
            for side in config.action.sides
        )
    if config.action.type == "delta_ee_world_with_hand_joint":
        return "`delta_ee_world_with_hand_joint`: " + "<br>".join(
            f"`{side}_delta_ee_world[9]`, `{side}_hand_joint_pos[22]`"
            for side in config.action.sides
        )
    if config.action.source == "next_state":
        return "`next_state` target: " + "<br>".join(f"`{token}[t+1]`" for token in config.action_tokens)
    return f"`action/{config.action.source}` nearest timestamp"


def _frame_filter_description(config: ConvertConfig) -> str:
    frame_filter = config.frame_filter
    return (
        f"`{frame_filter.type}`: "
        f"`max_tcp_translation_m={frame_filter.max_tcp_translation_m}`, "
        f"`max_rotation_6d={frame_filter.max_rotation_6d}`, "
        f"`max_hand_alpha={frame_filter.max_hand_alpha}`, "
        f"`min_run_frames={frame_filter.min_run_frames}`"
    )


def _visual_feature_storage(feature: dict) -> str:
    if feature["dtype"] == "video":
        return "video/mp4"
    if feature["dtype"] == "image":
        return "image/png"
    return str(feature["dtype"])


def _tactile_force_description(config: ConvertConfig) -> str:
    if config.tactile_force is None:
        return ""
    return "<br>".join(
        f"`{sensor.side}.{sensor.finger}.{config.tactile_force.field}`"
        for sensor in config.tactile_force.sensors
    )


def _tactile_force_rows(config: ConvertConfig, info: dict) -> list[list[str]]:
    if config.tactile_force is None:
        return []
    shape = tuple(info["features"]["observation.tactile_force"]["shape"])
    sensor_dim = int(shape[0]) // len(config.tactile_force.sensors)
    rows = []
    for sensor in config.tactile_force.sensors:
        source = f"`tactile/{sensor.side}/{sensor.finger}/{config.tactile_force.field}`"
        rows.append([f"`{sensor.name}`", source, f"`{sensor_dim}`"])
    return rows


def build_lerobot_dataset_card_kwargs(
    root: str | Path,
    config: ConvertConfig,
    *,
    task_name: str | None = None,
    language_instruction: str | None = None,
    config_path: str | None = None,
    raw_data_dir: str | None = None,
) -> dict[str, str]:
    from lerobot.datasets.io_utils import load_info

    root_path = Path(root)
    info = load_info(root_path)
    if task_name is None and "task_name" in info:
        task_name = str(info["task_name"])
    language_instructions = _language_instruction_list(language_instruction, root_path)
    return {
        "dataset_description": _build_dataset_description(
            config,
            info,
            language_instructions,
            task_name=task_name,
            config_path=config_path,
            raw_data_dir=raw_data_dir,
        )
    }


def write_lerobot_dataset_card(
    root: str | Path,
    config: ConvertConfig,
    *,
    task_name: str | None = None,
    language_instruction: str | None = None,
    config_path: str | None = None,
    raw_data_dir: str | None = None,
) -> Path:
    from lerobot.datasets.io_utils import load_info
    from lerobot.datasets.utils import create_lerobot_dataset_card

    root_path = Path(root)
    info = load_info(root_path)
    card_kwargs = build_lerobot_dataset_card_kwargs(
        root_path,
        config,
        task_name=task_name,
        language_instruction=language_instruction,
        config_path=config_path,
        raw_data_dir=raw_data_dir,
    )
    card = create_lerobot_dataset_card(
        tags=["tianji", "sharpa"],
        dataset_info=info,
        repo_id=config.repo_id,
        **card_kwargs,
    )
    readme_path = root_path / "README.md"
    card.save(readme_path)
    return readme_path


def update_lerobot_readme(
    output_dir: str | Path,
    config: ConvertConfig,
    *,
    task_name: str | None = None,
    language_instruction: str | None = None,
    config_path: str | None = None,
    raw_data_dir: str | None = None,
) -> Path:
    output_path = Path(output_dir)
    if not output_path.exists():
        raise FileNotFoundError(f"Dataset output directory does not exist: {output_path}")
    return write_lerobot_dataset_card(
        output_path,
        config,
        task_name=task_name,
        language_instruction=language_instruction,
        config_path=config_path,
        raw_data_dir=raw_data_dir,
    )
