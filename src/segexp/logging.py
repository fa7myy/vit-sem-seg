#!/usr/bin/env python3
"""
Run logging, environment capture, and run-directory utilities.
"""

from __future__ import annotations

import csv
import importlib
import json
import os
import platform
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)


def sanitize_name(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "run"


def default_run_name(args: Any) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    mode = "eval" if args.eval_only else "train"
    return sanitize_name(f"{timestamp}_{args.backbone}_{mode}_seed{args.seed}")


def make_run_dir(output_dir: str, run_name: str) -> Path:
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    candidate = base / sanitize_name(run_name)
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate
    suffix = 1
    while True:
        alt = base / f"{sanitize_name(run_name)}_{suffix:02d}"
        if not alt.exists():
            alt.mkdir(parents=True, exist_ok=False)
            return alt
        suffix += 1


def resolve_checkpoint_paths(save: bool, run_dir: Path, run_name: str) -> Tuple[str, str, str]:
    if not save:
        return "", "", ""
    checkpoints_dir = run_dir / "checkpoints"
    final_ckpt_path = checkpoints_dir / f"{run_name}_final.pth"
    best_ckpt_path = checkpoints_dir / f"{run_name}_best.pth"
    interrupted_ckpt_path = checkpoints_dir / f"{run_name}_interrupted.pth"
    return str(final_ckpt_path), str(best_ckpt_path), str(interrupted_ckpt_path)


def collect_env_info(device: torch.device) -> Dict[str, Any]:
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    for package in ("torchvision", "timm", "mmcv", "mmseg", "mmdet"):
        try:
            module = importlib.import_module(package)
            versions[package] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[package] = "not-installed"
    gpu_names: List[str] = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            gpu_names.append(torch.cuda.get_device_name(idx))
    return {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_names": gpu_names,
        "versions": versions,
    }


class RunLogger:
    train_fields = ("epoch", "avg_loss", "steps", "epoch_time_sec")
    eval_fields = (
        "epoch",
        "pixel_acc",
        "mIoU",
        "mean_class_acc",
        "eval_time_sec",
        "model_forward_time_sec",
        "mean_inference_time_ms",
        "throughput_img_s",
        "num_eval_images",
    )

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.events_path = self.run_dir / "events.log"
        self.train_metrics_path = self.run_dir / "train_metrics.csv"
        self.eval_metrics_path = self.run_dir / "eval_metrics.csv"

    def write_json(self, name: str, payload: Dict[str, Any]) -> None:
        path = self.run_dir / name
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def append_event(self, line: str) -> None:
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _append_csv(self, path: Path, fieldnames: Tuple[str, ...], row: Dict[str, Any]) -> None:
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def log_train_epoch(self, row: Dict[str, Any]) -> None:
        self._append_csv(self.train_metrics_path, self.train_fields, row)

    def log_eval_epoch(self, row: Dict[str, Any]) -> None:
        self._append_csv(self.eval_metrics_path, self.eval_fields, row)


def log(msg: str, run_logger: RunLogger | None = None) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    if run_logger is not None:
        run_logger.append_event(line)


def write_run_config(run_logger: RunLogger | None, run_info: Dict[str, Any]) -> None:
    if run_logger is not None:
        run_logger.write_json("run_config.json", run_info)


def init_device(deterministic: bool) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = not deterministic
    return device


def init_run_context(
    args: Any,
    device: torch.device,
    pretrain_size: int,
    resolved_timm_model: str,
) -> Tuple[Path, str, RunLogger | None, Dict[str, Any], float, str, str, str]:
    run_name = args.run_name or default_run_name(args)
    run_dir = make_run_dir(args.output_dir, run_name)
    run_name = run_dir.name
    run_logger: RunLogger | None = RunLogger(run_dir) if args.save_logs else None
    run_start_ts = time.time()
    run_info: Dict[str, Any] = {
        "run_name": run_name,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "cwd": str(Path.cwd()),
        "args": vars(args),
        "backbone_source": {
            "backbone": args.backbone,
            "checkpoint_path": args.ckpt if args.ckpt else None,
            "resolved_timm_model": resolved_timm_model if resolved_timm_model else None,
            "pretrain_size": pretrain_size,
        },
        "environment": collect_env_info(device),
    }
    final_ckpt_path, best_ckpt_path, interrupted_ckpt_path = resolve_checkpoint_paths(args.save, run_dir, run_name)
    run_info["checkpointing"] = {
        "save_checkpoints": bool(args.save),
        "final_checkpoint_path": final_ckpt_path if final_ckpt_path else None,
        "best_checkpoint_path": best_ckpt_path if best_ckpt_path else None,
        "interrupted_checkpoint_path": interrupted_ckpt_path if interrupted_ckpt_path else None,
    }
    run_info["run_dir"] = str(run_dir.resolve())
    write_run_config(run_logger, run_info)
    log(f"[run] run directory: {run_dir}", run_logger)
    return (
        run_dir,
        run_name,
        run_logger,
        run_info,
        run_start_ts,
        final_ckpt_path,
        best_ckpt_path,
        interrupted_ckpt_path,
    )
