"""Offline OpenPI policy evaluation on a LeRobot episode."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig
from rich.console import Console
from tqdm.auto import tqdm

from unirobot_infra.envs.layout import RobotActionLayout
from unirobot_infra.policy.client import build_policy_client
from unirobot_infra.policy.observation import prepare_policy_image
from unirobot_infra.replay.lerobot_episode import load_lerobot_episode_data
from unirobot_infra.utils.logging import get_logger, setup_logging


CONFIG_PATH = str(Path(__file__).resolve().parents[3] / "configs")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="apps/offline_openpi_policy_eval/joint_absolute")
def main(cfg: DictConfig) -> None:
    console = Console(stderr=True)
    setup_logging(cfg.logging.level, console=console)
    logger = get_logger(__name__)
    _configure_hf_cache(cfg)
    run_offline_openpi_policy_eval(cfg, logger)


def run_offline_openpi_policy_eval(cfg: DictConfig, logger) -> None:
    output_dir = Path(str(cfg.app.eval.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_data = load_lerobot_episode_data(cfg.app.dataset, logger)
    dataset = _load_lerobot_dataset(cfg.app.dataset)
    if len(dataset) != len(episode_data.actions):
        raise RuntimeError(f"LeRobot dataset rows {len(dataset)} does not match loaded actions {len(episode_data.actions)}")

    action_dim = int(episode_data.actions.shape[1])
    _validate_action_dim(cfg, action_dim)
    source_frames = _source_frames(cfg, len(episode_data.actions))
    logger.info(
        "offline OpenPI eval 启动 repo=%s episode=%s frames=%s source_frames=%s action_dim=%s output=%s",
        cfg.app.dataset.repo_id,
        int(cfg.app.dataset.episode_index),
        len(episode_data.actions),
        len(source_frames),
        action_dim,
        output_dir,
    )

    client = build_policy_client(cfg.policy.server)
    try:
        result = _infer_episode(
            cfg=cfg,
            logger=logger,
            client=client,
            dataset=dataset,
            episode_actions=episode_data.actions,
            episode_observations=episode_data.observations,
            source_frames=source_frames,
            action_dim=action_dim,
        )
    finally:
        client.close()

    labels = _action_labels(cfg)
    _write_outputs(
        cfg=cfg,
        output_dir=output_dir,
        result=result,
        gt_actions=episode_data.actions,
        source_frames=source_frames,
        labels=labels,
        logger=logger,
    )


def _configure_hf_cache(cfg: DictConfig) -> None:
    cache_dir = Path(str(cfg.app.cache_dir))
    hf_home = cache_dir / "hf_home"
    datasets_cache = cache_dir / "datasets"
    matplotlib_cache = cache_dir / "matplotlib"
    hf_home.mkdir(parents=True, exist_ok=True)
    datasets_cache.mkdir(parents=True, exist_ok=True)
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)


def _load_lerobot_dataset(dataset_cfg: DictConfig):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(
        repo_id=str(dataset_cfg.repo_id),
        root=Path(str(dataset_cfg.root)),
        episodes=[int(dataset_cfg.episode_index)],
        tolerance_s=float(dataset_cfg.tolerance_s),
        download_videos=False,
    )


def _validate_action_dim(cfg: DictConfig, action_dim: int) -> None:
    expected_dim = int(cfg.app.eval.action_dim)
    if action_dim != expected_dim:
        raise ValueError(f"dataset action dim {action_dim} does not match app.eval.action_dim {expected_dim}")


def _source_frames(cfg: DictConfig, episode_len: int) -> np.ndarray:
    start_frame = int(cfg.app.eval.start_frame)
    end_frame_value = cfg.app.eval.end_frame
    stride = int(cfg.app.eval.stride)
    max_inferences = int(cfg.app.eval.max_inferences)
    if start_frame < 0:
        raise ValueError("app.eval.start_frame must be non-negative")
    if start_frame >= episode_len:
        raise ValueError(f"app.eval.start_frame={start_frame} is outside episode length {episode_len}")
    if stride <= 0:
        raise ValueError("app.eval.stride must be positive")
    if max_inferences < 0:
        raise ValueError("app.eval.max_inferences must be non-negative")
    if end_frame_value is None:
        stop_frame = episode_len
    else:
        end_frame = int(end_frame_value)
        if end_frame < start_frame:
            raise ValueError("app.eval.end_frame must be greater than or equal to app.eval.start_frame")
        if end_frame >= episode_len:
            raise ValueError(f"app.eval.end_frame={end_frame} is outside episode length {episode_len}")
        stop_frame = end_frame + 1
    frames = np.arange(start_frame, stop_frame, stride, dtype=np.int64)
    if max_inferences > 0:
        frames = frames[:max_inferences]
    if len(frames) == 0:
        raise ValueError("offline eval source frame list is empty")
    return frames


def _infer_episode(
    *,
    cfg: DictConfig,
    logger,
    client,
    dataset,
    episode_actions: np.ndarray,
    episode_observations: np.ndarray,
    source_frames: np.ndarray,
    action_dim: int,
) -> dict[str, np.ndarray]:
    max_horizon = int(cfg.app.eval.max_horizon)
    if max_horizon < 0:
        raise ValueError("app.eval.max_horizon must be non-negative")
    overlap_mode = str(cfg.app.eval.overlap_mode)
    if overlap_mode not in {"mean", "first", "last"}:
        raise ValueError("app.eval.overlap_mode must be mean, first, or last")

    pred_sum = np.zeros_like(episode_actions, dtype=np.float64)
    pred_aligned = np.full_like(episode_actions, np.nan, dtype=np.float32)
    pred_count = np.zeros((episode_actions.shape[0],), dtype=np.int64)
    chunk_horizon = 0
    pred_chunks: list[np.ndarray] = []
    chunk_valid_masks: list[np.ndarray] = []

    for source_frame in tqdm(source_frames, desc="Offline OpenPI infer", unit="chunk"):
        item = dataset[int(source_frame)]
        observation = _policy_observation_payload(cfg, item, episode_observations[int(source_frame)])
        response = client.infer(observation)
        chunk = _response_action_chunk(response, str(cfg.policy.action.response_key), action_dim)
        if max_horizon > 0:
            chunk = chunk[:max_horizon]
        chunk_horizon = max(chunk_horizon, int(chunk.shape[0]))
        pred_chunks.append(chunk.astype(np.float32))
        valid_mask = np.zeros((chunk.shape[0],), dtype=bool)
        for horizon_index, prediction in enumerate(chunk):
            target_frame = int(source_frame) + int(horizon_index)
            if target_frame >= len(episode_actions):
                continue
            valid_mask[horizon_index] = True
            if overlap_mode == "mean":
                pred_sum[target_frame] += prediction.astype(np.float64)
                pred_count[target_frame] += 1
            elif overlap_mode == "first":
                if pred_count[target_frame] == 0:
                    pred_aligned[target_frame] = prediction
                    pred_count[target_frame] = 1
            elif overlap_mode == "last":
                pred_aligned[target_frame] = prediction
                pred_count[target_frame] = 1
        chunk_valid_masks.append(valid_mask)
        logger.debug("offline OpenPI source_frame=%s chunk_shape=%s", int(source_frame), chunk.shape)

    if overlap_mode == "mean":
        predicted_frames = pred_count > 0
        pred_aligned[predicted_frames] = (pred_sum[predicted_frames] / pred_count[predicted_frames, None]).astype(np.float32)
    if chunk_horizon == 0:
        raise RuntimeError("OpenPI returned no valid action chunks")
    return {
        "pred_aligned": pred_aligned,
        "pred_count": pred_count,
        "pred_chunks": _pad_chunks(pred_chunks, chunk_horizon, action_dim),
        "chunk_valid_masks": _pad_masks(chunk_valid_masks, chunk_horizon),
    }


def _policy_observation_payload(cfg: DictConfig, item: dict[str, Any], state: np.ndarray) -> dict[str, Any]:
    server_cfg = cfg.policy.observation.server
    if str(server_cfg.format) != "openpi_flat":
        raise ValueError(f"unsupported policy observation format: {server_cfg.format}")
    payload: dict[str, Any] = {
        str(server_cfg.prompt_key): str(cfg.policy.task),
    }
    if bool(server_cfg.state.enabled):
        payload[str(server_cfg.state.key)] = np.asarray(state, dtype=np.float32)
    if bool(server_cfg.images.enabled):
        for mapping in cfg.app.dataset.image_mappings:
            image = _lerobot_image_to_hwc_uint8(item[str(mapping.dataset_key)])
            payload[str(mapping.policy_key)] = prepare_policy_image(image, cfg=server_cfg.images)
    if bool(server_cfg.tactile.enabled):
        raise ValueError("offline OpenPI eval does not support tactile observation yet")
    return payload


def _response_action_chunk(response: dict[str, Any], response_key: str, action_dim: int) -> np.ndarray:
    action_chunk = np.asarray(response[response_key], dtype=np.float32)
    if action_chunk.ndim == 1:
        action_chunk = action_chunk.reshape(1, -1)
    if action_chunk.ndim != 2:
        raise RuntimeError(f"policy actions must be [H,D], got {action_chunk.shape}")
    if action_chunk.shape[1] != action_dim:
        raise RuntimeError(f"policy action dim {action_chunk.shape[1]} does not match dataset action_dim {action_dim}")
    if action_chunk.shape[0] == 0:
        raise RuntimeError("policy returned an empty action horizon")
    if not np.all(np.isfinite(action_chunk)):
        raise RuntimeError("policy actions contain NaN or Inf")
    return action_chunk


def _lerobot_image_to_hwc_uint8(value) -> np.ndarray:
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError("LeRobot image must be a torch.Tensor")
    if value.dtype != torch.float32:
        raise TypeError(f"LeRobot image must be torch.float32, got {value.dtype}")
    if value.ndim != 3 or value.shape[0] != 3:
        raise ValueError(f"LeRobot image must have shape [3,H,W], got {tuple(value.shape)}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("LeRobot image contains NaN or Inf")
    if bool((value < 0).any()) or bool((value > 1).any()):
        raise ValueError("LeRobot image values must be within [0,1]")
    return (value * 255).to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()


def _pad_chunks(chunks: list[np.ndarray], horizon: int, action_dim: int) -> np.ndarray:
    output = np.full((len(chunks), horizon, action_dim), np.nan, dtype=np.float32)
    for index, chunk in enumerate(chunks):
        output[index, : chunk.shape[0], :] = chunk
    return output


def _pad_masks(masks: list[np.ndarray], horizon: int) -> np.ndarray:
    output = np.zeros((len(masks), horizon), dtype=bool)
    for index, mask in enumerate(masks):
        output[index, : mask.shape[0]] = mask
    return output


def _action_labels(cfg: DictConfig) -> list[str]:
    if cfg.app.eval.action_labels is not None:
        labels = [str(label) for label in cfg.app.eval.action_labels]
        expected_dim = int(cfg.app.eval.action_dim)
        if len(labels) != expected_dim:
            raise ValueError(f"app.eval.action_labels length {len(labels)} does not match app.eval.action_dim {expected_dim}")
        return labels
    layout = RobotActionLayout.from_robot_config(cfg.robot)
    labels = []
    for segment in layout.segments:
        for local_index, joint_name in enumerate(segment.joint_names):
            labels.append(f"{segment.side}_{segment.role}_{local_index}:{joint_name}")
    return labels


def _write_outputs(
    *,
    cfg: DictConfig,
    output_dir: Path,
    result: dict[str, np.ndarray],
    gt_actions: np.ndarray,
    source_frames: np.ndarray,
    labels: list[str],
    logger,
) -> None:
    pred_aligned = result["pred_aligned"]
    pred_count = result["pred_count"]
    predicted_frames = pred_count > 0
    if not np.any(predicted_frames):
        raise RuntimeError("offline OpenPI eval produced no aligned predictions")
    errors = pred_aligned[predicted_frames] - gt_actions[predicted_frames]
    frame_metrics = _frame_metrics(predicted_frames, errors)
    summary = _summary_dict(cfg, gt_actions, source_frames, pred_count, errors)
    np.savez_compressed(
        output_dir / "predictions.npz",
        gt_actions=gt_actions,
        pred_aligned=pred_aligned,
        pred_count=pred_count,
        pred_chunks=result["pred_chunks"],
        chunk_valid_masks=result["chunk_valid_masks"],
        source_frames=source_frames,
        labels=np.asarray(labels, dtype=object),
    )
    _write_json(output_dir / "summary.json", summary)
    _write_frame_metrics_csv(output_dir / "frame_metrics.csv", frame_metrics)
    _plot_pred_vs_gt(
        output_dir / "pred_vs_gt.png",
        gt_actions,
        pred_aligned,
        pred_count,
        labels,
        int(cfg.app.eval.plot.max_dims),
    )
    logger.info("offline OpenPI eval 写入 output=%s", output_dir)
    logger.info(
        "offline OpenPI eval summary predicted_frames=%s rmse=%.6f mae=%.6f max_abs=%.6f",
        int(summary["predicted_frames"]),
        float(summary["rmse"]),
        float(summary["mae"]),
        float(summary["max_abs"]),
    )


def _frame_metrics(predicted_frames: np.ndarray, errors: np.ndarray) -> list[dict[str, float | int]]:
    frame_indices = np.flatnonzero(predicted_frames)
    rows = []
    for row_index, frame_index in enumerate(frame_indices):
        err = errors[row_index]
        rows.append(
            {
                "frame": int(frame_index),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err * err))),
                "max_abs": float(np.max(np.abs(err))),
            }
        )
    return rows


def _summary_dict(
    cfg: DictConfig,
    gt_actions: np.ndarray,
    source_frames: np.ndarray,
    pred_count: np.ndarray,
    errors: np.ndarray,
) -> dict[str, Any]:
    return {
        "repo_id": str(cfg.app.dataset.repo_id),
        "root": str(cfg.app.dataset.root),
        "episode_index": int(cfg.app.dataset.episode_index),
        "episode_frames": int(gt_actions.shape[0]),
        "action_dim": int(gt_actions.shape[1]),
        "source_frames": int(len(source_frames)),
        "predicted_frames": int(np.count_nonzero(pred_count)),
        "stride": int(cfg.app.eval.stride),
        "overlap_mode": str(cfg.app.eval.overlap_mode),
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "mae": float(np.mean(np.abs(errors))),
        "max_abs": float(np.max(np.abs(errors))),
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_frame_metrics_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    lines = ["frame,mae,rmse,max_abs"]
    for row in rows:
        lines.append(f"{row['frame']},{row['mae']:.9f},{row['rmse']:.9f},{row['max_abs']:.9f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_pred_vs_gt(
    path: Path,
    gt_actions: np.ndarray,
    pred_aligned: np.ndarray,
    pred_count: np.ndarray,
    labels: list[str],
    max_dims: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    action_dim = gt_actions.shape[1]
    dims = min(action_dim, max_dims)
    cols = 4
    rows = int(np.ceil(dims / cols))
    x = np.arange(gt_actions.shape[0])
    pred_masked = np.ma.masked_invalid(pred_aligned)
    fig, axes = plt.subplots(rows, cols, figsize=(24, 2.8 * rows), constrained_layout=True, squeeze=False)
    for dim in range(rows * cols):
        ax = axes[dim // cols][dim % cols]
        if dim >= dims:
            ax.axis("off")
            continue
        ax.plot(x, gt_actions[:, dim], label="gt", linewidth=1.0)
        ax.plot(x, pred_masked[:, dim], label="pred", linewidth=1.0)
        ax.set_title(labels[dim] if dim < len(labels) else f"dim_{dim}", fontsize=7)
        ax.tick_params(axis="both", labelsize=6)
        if dim == 0:
            ax.legend(fontsize=7)
    fig.suptitle(f"OpenPI offline prediction vs GT action; predicted frames={int(np.count_nonzero(pred_count))}")
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
