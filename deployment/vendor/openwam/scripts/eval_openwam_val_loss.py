#!/usr/bin/env python3
"""Evaluate OpenWAM validation loss on a configured dataset split."""

from __future__ import annotations

import faulthandler
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

faulthandler.enable(file=sys.stderr, all_threads=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train import _build_accelerator, _inject_project_seed, _install_startup_noise_filter  # noqa: E402

logger = logging.getLogger(__name__)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    return int(value)


def _tensor_scalar(value: Any, *, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().float().to(device)
    return torch.tensor(float(value), device=device, dtype=torch.float32)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _evaluate(cfg: DictConfig) -> None:
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from openwam.dataloader.registry import build_dataset
    from openwam.train.openwam_trainer import OpenWAMTrainer
    from openwam.train.utils.seeding import per_step_seed, seed_everything

    _inject_project_seed(cfg)
    project_seed = OmegaConf.select(cfg, "project.seed", default=None)
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    if project_seed is not None:
        seed_everything(int(project_seed), rank=rank)

    split = str(OmegaConf.select(cfg, "evaluation.split", default=None) or OmegaConf.select(cfg, "dataloader.split", default="val"))
    batch_size = int(OmegaConf.select(cfg, "evaluation.batch_size", default=cfg.training.batch_size))
    num_workers = int(OmegaConf.select(cfg, "evaluation.dataset_num_workers", default=cfg.training.dataset_num_workers))
    max_batches = _optional_int(OmegaConf.select(cfg, "evaluation.max_batches", default=None))
    output_path = OmegaConf.select(cfg, "evaluation.output_path", default=None)
    if output_path is None:
        output_path = str(Path(str(OmegaConf.select(cfg, "project.output_dir", default="outputs"))) / "val_loss.json")

    accelerator = _build_accelerator(cfg)
    dataset = build_dataset(cfg.dataloader, split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=list,
        pin_memory=True,
    )

    trainer = OpenWAMTrainer(cfg, accelerator=accelerator, dataset=dataset)
    trainer.architecture.eval()
    trainer.architecture = accelerator.prepare_model(trainer.architecture, evaluation_mode=True)
    dataloader = accelerator.prepare_data_loader(dataloader)
    arch = accelerator.unwrap_model(trainer.architecture)
    arch.set_dtype_device(arch.dtype, accelerator.device)
    arch.move_frozen_to_device(accelerator.device)
    trainer.architecture.eval()

    sums = torch.zeros(4, dtype=torch.float64, device=accelerator.device)
    local_batches = 0
    total_iters = len(dataloader) if max_batches is None else min(len(dataloader), max_batches)
    progress = tqdm(total=total_iters, desc=f"Eval {split}", unit="batch", disable=not accelerator.is_main_process)

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if max_batches is not None and step >= max_batches:
                break
            if project_seed is not None:
                step_seed = per_step_seed(int(project_seed), rank=rank, step=step)
                torch.manual_seed(step_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(step_seed)
            losses = trainer.compute_loss(batch)
            device = losses["total"].device if isinstance(losses["total"], torch.Tensor) else accelerator.device
            n = float(len(batch) if isinstance(batch, list) else 1)
            local = torch.stack(
                [
                    _tensor_scalar(losses["total"], device=device) * n,
                    _tensor_scalar(losses["video"], device=device) * n,
                    _tensor_scalar(losses["action"], device=device) * n,
                    torch.tensor(n, device=device, dtype=torch.float32),
                ]
            ).reshape(1, 4)
            gathered = accelerator.gather(local).sum(dim=0).to(dtype=torch.float64)
            sums += gathered
            local_batches += 1
            progress.update(1)

    progress.close()
    totals = sums
    batches_total = local_batches
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        samples = max(float(totals[3].item()), 1.0)
        result = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(OmegaConf.select(cfg, "training.finetune_ckpt_path", default=None)),
            "dataset_dir": str(OmegaConf.select(cfg, "dataloader.dataset_dir", default=None)),
            "split": split,
            "max_batches": max_batches,
            "num_batches": int(batches_total),
            "num_samples": int(totals[3].item()),
            "loss_total": float(totals[0].item() / samples),
            "loss_video": float(totals[1].item() / samples),
            "loss_action": float(totals[2].item() / samples),
            "world_size": int(accelerator.num_processes),
            "batch_size_per_process": batch_size,
            "metric_note": "Held-out teacher-forced diffusion/flow-matching loss; not rollout success rate.",
        }
        out = Path(str(output_path))
        _write_json(out, result)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="train")
def main(cfg: DictConfig) -> None:
    _install_startup_noise_filter()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print("=" * 60)
        print("OpenWAM Validation Loss")
        print("=" * 60)
        print(OmegaConf.to_yaml(cfg))
        print("=" * 60)
    else:
        import builtins

        builtins.print = lambda *a, **kw: None
    try:
        _evaluate(cfg)
    except BaseException:
        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "?"))
        sys.stderr.write(f"\n===== RANK {rank} EXCEPTION (pre-destroy) =====\n")
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.stdout.flush()
        raise
    finally:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
