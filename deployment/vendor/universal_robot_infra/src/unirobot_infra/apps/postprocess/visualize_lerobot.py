"""Visualize a LeRobot dataset episode with grouped state/action plots in Rerun."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
_TACTILE_MODALITY_ORDER = ("raw", "deform")
_CAMERA_ORDER_HINTS = ("head", "left", "high", "right")

_PALETTE = [
    [230, 25, 75],
    [60, 180, 75],
    [0, 130, 200],
    [245, 130, 48],
    [145, 30, 180],
    [70, 240, 240],
    [240, 50, 230],
    [210, 245, 60],
    [250, 190, 190],
    [0, 128, 128],
]


@dataclass(frozen=True)
class _TactileName:
    side: str
    finger: str
    modality: str


@dataclass(frozen=True)
class _VisualRows:
    cameras: list[str]
    left_tactile: list[str]
    right_tactile: list[str]


def _to_hwc_uint8_numpy(value) -> np.ndarray:
    import torch

    if isinstance(value, torch.Tensor):
        if value.dtype != torch.float32:
            raise TypeError("image tensor must be torch.float32")
        if value.ndim != 3:
            raise ValueError("image tensor must be CHW")
        channels, height, width = value.shape
        if channels >= height or channels >= width:
            raise ValueError(f"Expected CHW image tensor, got {tuple(value.shape)}")
        return (value * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[0] < array.shape[1] and array.shape[0] < array.shape[2]:
        array = np.transpose(array, (1, 2, 0))
    return np.asarray(array, dtype=np.uint8)


def _to_flat_numpy(value) -> np.ndarray:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().reshape(-1)
    return np.asarray(value).reshape(-1)


def _camera_name(feature_key: str) -> str:
    prefix = "observation.images."
    return feature_key[len(prefix) :] if feature_key.startswith(prefix) else feature_key


def _parse_tactile_name(name: str) -> _TactileName | None:
    parts = name.split("_")
    if len(parts) < 3:
        return None
    if parts[0] == "tactile" and len(parts) >= 4:
        side = parts[1]
        modality = parts[2]
        finger = "_".join(parts[3:])
    else:
        side = parts[0]
        finger = "_".join(parts[1:-1])
        modality = parts[-1]
    if side not in {"left", "right"}:
        return None
    if finger not in _FINGER_ORDER:
        return None
    if modality not in _TACTILE_MODALITY_ORDER:
        return None
    return _TactileName(side=side, finger=finger, modality=modality)


def _group_visual_names(image_names: list[str]) -> _VisualRows:
    tactile_names = [name for name in image_names if _parse_tactile_name(name) is not None]
    tactile_set = set(tactile_names)
    camera_names = [name for name in image_names if name not in tactile_set]
    return _VisualRows(
        cameras=_ordered_camera_names(camera_names),
        left_tactile=_ordered_tactile_names(tactile_names, "left"),
        right_tactile=_ordered_tactile_names(tactile_names, "right"),
    )


def _ordered_camera_names(camera_names: list[str]) -> list[str]:
    ordered = []
    for hint in _CAMERA_ORDER_HINTS:
        for name in camera_names:
            if name not in ordered and hint in name.lower():
                ordered.append(name)
                break
    ordered.extend(name for name in camera_names if name not in ordered)
    return ordered


def _ordered_tactile_names(tactile_names: list[str], side: str) -> list[str]:
    names_by_key = {}
    for name in tactile_names:
        parsed = _parse_tactile_name(name)
        if parsed is not None and parsed.side == side:
            names_by_key[(parsed.modality, parsed.finger)] = name
    ordered = []
    for modality in _TACTILE_MODALITY_ORDER:
        for finger in _FINGER_ORDER:
            key = (modality, finger)
            if key in names_by_key:
                ordered.append(names_by_key[key])
    return ordered


def _image_view_title(name: str) -> str:
    tactile_name = _parse_tactile_name(name)
    if tactile_name is not None:
        finger_idx = _FINGER_ORDER.index(tactile_name.finger) + 1
        return f"{finger_idx} {tactile_name.modality}"
    for hint in _CAMERA_ORDER_HINTS:
        if hint in name.lower():
            return hint
    return name


def _resolve_labels(dataset, feature_key: str, override: list[str] | None) -> list[str]:
    feature = dataset.meta.features[feature_key]
    shape = tuple(feature["shape"])
    if len(shape) != 1:
        raise ValueError(f"{feature_key} must be 1D, got shape {shape}")
    dim = int(shape[0])
    if override is not None:
        if len(override) != dim:
            raise ValueError(f"{feature_key} expects {dim} labels, got {len(override)}")
        return override
    if feature["names"] is not None and len(feature["names"]) == dim:
        return [str(name) for name in feature["names"]]
    stem = feature_key.split(".")[-1]
    return [f"{stem}_{index}" for index in range(dim)]


def _build_blueprint(image_names: list[str]):
    import rerun.blueprint as rrb

    rows = _group_visual_names(image_names)
    views = []
    if rows.cameras:
        views.append(_image_grid(rrb, "Cameras", rows.cameras))
    if rows.left_tactile:
        views.append(_image_grid(rrb, "Left Tactile", rows.left_tactile))
    if rows.right_tactile:
        views.append(_image_grid(rrb, "Right Tactile", rows.right_tactile))
    views.append(
        rrb.Horizontal(
            rrb.TimeSeriesView(origin="/state", name="State", plot_legend=rrb.PlotLegend(visible=True)),
            rrb.TimeSeriesView(origin="/action", name="Action", plot_legend=rrb.PlotLegend(visible=True)),
            column_shares=[0.5, 0.5],
            name="Low Dim",
        )
    )
    return rrb.Blueprint(rrb.Vertical(contents=views), collapse_panels=True)


def _image_grid(rrb, name: str, image_names: list[str]):
    return rrb.Grid(
        contents=[
            rrb.Spatial2DView(origin=f"/images/{image_name}", name=_image_view_title(image_name))
            for image_name in image_names
        ],
        grid_columns=len(image_names),
        name=name,
    )


def _log_series_styles(prefix: str, labels: list[str]) -> None:
    import rerun as rr

    for index, label in enumerate(labels):
        rr.log(
            f"{prefix}/{index}",
            rr.SeriesLines(
                names=label,
                colors=_PALETTE[index % len(_PALETTE)],
                widths=2.0,
            ),
            static=True,
        )


def _log_vector(prefix: str, values) -> None:
    import rerun as rr

    for index, value in enumerate(_to_flat_numpy(values)):
        rr.log(f"{prefix}/{index}", rr.Scalars(float(value)))


def visualize_lerobot_episode(
    dataset,
    episode_index: int,
    *,
    save: bool = False,
    output_dir: Path | None = None,
    display_compressed_images: bool = False,
    state_labels: list[str] | None = None,
    action_labels: list[str] | None = None,
) -> Path | None:
    import rerun as rr
    from lerobot.utils.constants import ACTION, OBS_STATE

    if save and output_dir is None:
        raise ValueError("--output-dir is required when --save is set")

    image_keys = list(dataset.meta.camera_keys)
    image_names = [_camera_name(key) for key in image_keys]
    resolved_state_labels = _resolve_labels(dataset, OBS_STATE, state_labels)
    resolved_action_labels = _resolve_labels(dataset, ACTION, action_labels)

    rr.init(f"{dataset.repo_id}/episode_{episode_index}", spawn=not save)
    rr.send_blueprint(_build_blueprint(image_names), make_default=True, make_active=True)
    _log_series_styles("state", resolved_state_labels)
    _log_series_styles("action", resolved_action_labels)

    for item in dataset:
        rr.set_time("frame_index", sequence=int(item["frame_index"].item()))
        rr.set_time("timestamp", timestamp=float(item["timestamp"].item()))
        for feature_key, image_name in zip(image_keys, image_names, strict=True):
            image = _to_hwc_uint8_numpy(item[feature_key])
            image_entity = rr.Image(image).compress() if display_compressed_images else rr.Image(image)
            rr.log(f"images/{image_name}", image_entity)
        _log_vector("state", item[OBS_STATE])
        _log_vector("action", item[ACTION])

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        repo_id_str = dataset.repo_id.replace("/", "_")
        rrd_path = output_dir / f"{repo_id_str}_episode_{episode_index}.rrd"
        rr.save(rrd_path)
        return rrd_path
    return None


def main() -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    parser = argparse.ArgumentParser(description="Visualize a LeRobot dataset episode with grouped Rerun plots")
    parser.add_argument("--repo-id", type=str, required=True, help="Dataset repo id or local identifier")
    parser.add_argument("--episode-index", type=int, required=True, help="Episode index to visualize")
    parser.add_argument("--root", type=Path, default=None, help="Local dataset root directory")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for .rrd output when saving")
    parser.add_argument("--save", action="store_true", help="Save an .rrd instead of spawning the viewer")
    parser.add_argument(
        "--display-compressed-images",
        action="store_true",
        help="Log compressed images to reduce Rerun memory usage",
    )
    parser.add_argument("--tolerance-s", type=float, default=1e-4, help="LeRobot fps timestamp tolerance")
    parser.add_argument("--state-labels", nargs="*", default=None, help="Legend labels for observation.state")
    parser.add_argument("--action-labels", nargs="*", default=None, help="Legend labels for action")
    args = parser.parse_args()

    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        episodes=[args.episode_index],
        root=args.root,
        tolerance_s=args.tolerance_s,
    )
    rrd_path = visualize_lerobot_episode(
        dataset,
        episode_index=args.episode_index,
        save=args.save,
        output_dir=args.output_dir,
        display_compressed_images=args.display_compressed_images,
        state_labels=args.state_labels,
        action_labels=args.action_labels,
    )
    if rrd_path is not None:
        print(f"Saved Rerun recording to {rrd_path}")


if __name__ == "__main__":
    main()
