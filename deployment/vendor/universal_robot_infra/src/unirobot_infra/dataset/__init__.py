from __future__ import annotations

from importlib import import_module

__all__ = [
    "ConvertConfig",
    "EpisodeReader",
    "ResampledEpisode",
    "convert_to_lerobot",
    "resample_episode",
    "update_lerobot_readme",
]


def __getattr__(name: str):
    if name == "ConvertConfig":
        return getattr(import_module(".convert_config", __name__), name)
    if name == "EpisodeReader":
        return getattr(import_module(".episode_reader", __name__), name)
    if name in {"ResampledEpisode", "resample_episode"}:
        return getattr(import_module(".resampler", __name__), name)
    if name == "convert_to_lerobot":
        return getattr(import_module(".lerobot_converter", __name__), name)
    if name == "update_lerobot_readme":
        return getattr(import_module(".dataset_card", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
