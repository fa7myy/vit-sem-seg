#!/usr/bin/env python3
"""
Standalone linear probe with swappable ViT-B backbones (DINOv2/CLIP/MAE) + ViT-Adapter + multi-scale head.
Run from /mnt/c/Projects/thesis without modifying the ViT-Adapter repo.

Example:
  python /mnt/c/Projects/thesis/run_experiment.py \
    --data-root /path/to/VOCdevkit \
    --backbone dinov2 \
    --timm-model vit_base_patch14_dinov2.lvd142m \
    --img-size 512 \
    --batch-size 2 \
    --epochs 10
"""

from __future__ import annotations

import argparse
import csv
import json
import importlib
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from eval_utils import (
    estimate_flops,
    evaluate,
    save_class_metrics_csv,
    save_confusion_matrix_csv,
)

try:
    from torchvision.datasets import VOCSegmentation
    from torchvision.transforms import InterpolationMode
    import torchvision.transforms.functional as TF
except Exception as exc:
    raise RuntimeError("torchvision is required for VOCSegmentation") from exc


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# OpenAI CLIP normalization (RGB) for inputs scaled to [0, 1].
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

DEFAULT_TIMM_MODELS = {
    "dinov2": "vit_base_patch14_dinov2.lvd142m",
    "clip": "clip_vit_base_patch16_224.openai",
    "mae": "mae_vit_base_patch16",
}

DEFAULT_PRETRAIN_SIZE = {
    "dinov2": 592,  # 518@p14 -> 592@p16
    "clip": 224,
    "mae": 224,
}


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str):
    """
    Adds paired --name / --no-name flags that set a boolean dest.
    Compatible with Python versions < 3.9 that lack BooleanOptionalAction.
    """
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text + " (default: {})".format(default))
    group.add_argument(f"--no-{name}", dest=dest, action="store_false", help="Disable " + help_text.lower())
    parser.set_defaults(**{dest: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ViT-Adapter linear probe (VOC2012) with DINOv2/CLIP/MAE backbones.")
    parser.add_argument("--data-root", type=str, default="",
                        help="Path containing VOCdevkit (torchvision VOCSegmentation root).")
    add_bool_arg(parser, "download", True, "Download VOC2012 via torchvision if not present.")
    parser.add_argument("--backbone", type=str, choices=["dinov2", "clip", "mae"], default="dinov2",
                        help="Pretrained ViT-B source.")
    parser.add_argument("--ckpt", type=str, default="",
                        help="Path to a pretrained checkpoint (optional; overrides --timm-model).")
    parser.add_argument("--timm-model", type=str, default="",
                        help="timm model name to load (defaults per backbone).")
    parser.add_argument("--img-size", type=int, default=512,
                        help="Input image size (square, must be divisible by 32). 384 is a good speed/acc tradeoff.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=20,
                        help="Print training loss every N iterations.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=5e-6,
                        help="Smaller LR for pretrained transformer blocks.")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="Deprecated (use --weight-decay-head).")
    parser.add_argument("--weight-decay-head", type=float, default=0.0,
                        help="Weight decay for head + adapter params.")
    parser.add_argument("--weight-decay-backbone", type=float, default=0.05,
                        help="Weight decay for transformer backbone params.")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Global grad-norm clip (0 disables).")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save final and best checkpoints under <run_dir>/checkpoints/ using fixed names.",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run a single forward pass on random input and exit.")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision.")
    parser.add_argument("--torch-compile", action="store_true",
                        help="Use torch.compile when available (CUDA + torch>=2.0).")
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze ViT-Adapter backbone (linear probe). Default is backbone trainable.",
    )
    parser.add_argument("--unfreeze-at-epoch", type=int, default=-1,
                        help="Epoch (1-indexed) to unfreeze last transformer blocks; -1 disables.")
    parser.add_argument("--unfreeze-last-n-blocks", type=int, default=4,
                        help="Number of final transformer blocks to unfreeze.")
    add_bool_arg(parser, "syncbn", False,
                 "Use SyncBatchNorm (requires torch.distributed). By default converts SyncBN to BN for single GPU.")
    add_bool_arg(parser, "with-cp", False,
                 "Enable gradient checkpointing (with_cp) in ViT-Adapter to save memory.")
    add_bool_arg(parser, "miou-ignore-empty", True,
                 "Compute mIoU over classes with non-empty union.")
    add_bool_arg(parser, "measure-inference-time", True,
                 "Measure per-image inference time during evaluation (adds sync overhead on CUDA).")
    parser.add_argument(
        "--input-norm",
        type=str,
        choices=["imagenet", "clip"],
        default="imagenet",
        help="Image normalization applied after to_tensor(). Keep constant for fair encoder comparisons.",
    )
    add_bool_arg(
        parser,
        "clip-zero-missing-patch-embed-bias",
        False,
        "Zero patch_embed.proj.bias when missing in CLIP checkpoints (helps emulate biasless CLIP patch embedding).",
    )
    parser.add_argument("--pretrain-size", type=int, default=0,
                        help="Backbone pretrain resolution (0 selects a sensible default for the chosen backbone).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for Python/NumPy/PyTorch.")
    add_bool_arg(parser, "deterministic", False,
                 "Enable deterministic mode (slower, more reproducible).")
    add_bool_arg(parser, "save-logs", True,
                 "Persist structured run artifacts (JSON/CSV).")
    add_bool_arg(parser, "profile-flops", False,
                 "Estimate FLOPs per image using fvcore (if available).")
    parser.add_argument("--output-dir", type=str, default="runs",
                        help="Base directory for run artifacts.")
    parser.add_argument("--run-name", type=str, default="",
                        help="Optional run name (defaults to timestamp_backbone_mode_seed).")
    parser.add_argument("--target-miou", type=float, default=0.0,
                        help="Optional convergence threshold. If >0, logs first epoch reaching this mIoU.")
    return parser.parse_args()


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


def seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def sanitize_name(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "run"


def default_run_name(args: argparse.Namespace) -> str:
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


def first_epoch_reaching(history: List[Dict[str, Any]], threshold: float) -> int:
    if threshold <= 0:
        return -1
    for row in history:
        if row["mIoU"] >= threshold:
            return int(row["epoch"])
    return -1


class Vocab:
    num_classes: int = 21
    ignore_index: int = 255


class VocTransform:
    def __init__(self, size: int, mean: Tuple[float, float, float], std: Tuple[float, float, float]):
        self.size = (size, size)
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = TF.resize(image, self.size, interpolation=InterpolationMode.BICUBIC)
        target = TF.resize(target, self.size, interpolation=InterpolationMode.NEAREST)
        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.std)
        target = torch.from_numpy(np.array(target, dtype="uint8")).long()
        return image, target


class MultiScaleFPNHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, fpn_dim: int = 256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, fpn_dim, kernel_size=1) for _ in range(4)
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1) for _ in range(4)
        ])
        self.classifier = nn.Conv2d(fpn_dim, num_classes, kernel_size=1)

    def forward(self, feats: Tuple[torch.Tensor, ...], out_size: Tuple[int, int]) -> torch.Tensor:
        if len(feats) != 4:
            raise ValueError("Expected 4 feature maps from ViT-Adapter.")
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, feats)]
        # Top-down fusion from stride 32 -> 16 -> 8 -> 4.
        for i in range(3, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:],
                               mode="bilinear", align_corners=False)
            laterals[i - 1] = laterals[i - 1] + up
        outs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        logits = self.classifier(outs[0])
        if logits.shape[-2:] != out_size:
            logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
        return logits


class ViTAdapterLinearProbe(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = MultiScaleFPNHead(in_channels=backbone.embed_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(tuple(feats), out_size=x.shape[-2:])


def _clean_state_dict(raw: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for k, v in raw.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if "mask_token" in k:
            continue
        if k.startswith("head.") or k.startswith("fc_norm"):
            continue
        k = k.replace("ls1.gamma", "gamma1").replace("ls2.gamma", "gamma2")
        if k == "patch_embed.proj.weight" and v.ndim == 4 and v.shape[-1] != 16:
            v = F.interpolate(v, size=(16, 16), mode="bilinear", align_corners=False)
        cleaned[k] = v
    return cleaned


def load_state_dict_from_ckpt(path: str) -> Dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    elif isinstance(raw, dict) and "model" in raw:
        raw = raw["model"]
    return _clean_state_dict(raw)


def load_state_dict_from_timm(model_name: str) -> Dict[str, torch.Tensor]:
    try:
        import timm
    except Exception as exc:
        raise RuntimeError("timm is required when using --timm-model") from exc
    model = timm.create_model(model_name, pretrained=True)
    state_dict = model.state_dict()
    return _clean_state_dict(state_dict)


def resolve_timm_model(backbone: str, timm_model: str) -> str:
    if timm_model:
        return timm_model
    return DEFAULT_TIMM_MODELS[backbone]


def resolve_pretrain_size(backbone: str, requested: int) -> int:
    if requested and requested > 0:
        return requested
    return DEFAULT_PRETRAIN_SIZE[backbone]


def load_pretrained_state_dict(backbone: str, timm_model: str, ckpt_path: str) -> Dict[str, torch.Tensor]:
    if ckpt_path:
        return load_state_dict_from_ckpt(ckpt_path)
    model_name = resolve_timm_model(backbone, timm_model)
    return load_state_dict_from_timm(model_name)


def convert_syncbn_to_bn(module: nn.Module) -> nn.Module:
    """Recursively replace SyncBatchNorm with BatchNorm2d for single-GPU usage."""
    module_output = module
    if isinstance(module, nn.SyncBatchNorm):
        module_output = nn.BatchNorm2d(
            module.num_features,
            eps=module.eps,
            momentum=module.momentum,
            affine=module.affine,
            track_running_stats=module.track_running_stats,
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    else:
        for name, child in module.named_children():
            new_child = convert_syncbn_to_bn(child)
            if new_child is not child:
                module_output.add_module(name, new_child)
    return module_output


def build_backbone(ViTAdapter, pretrain_size: int, with_cp: bool):
    backbone = ViTAdapter(
        pretrain_size=pretrain_size,
        img_size=pretrain_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        drop_path_rate=0.3,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=12,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        window_attn=[True, True, False, True, True, False,
                     True, True, False, True, True, False],
        window_size=[14, 14, None, 14, 14, None,
                     14, 14, None, 14, 14, None],
        pretrained=None,
        with_cp=with_cp,
    )
    return backbone


def _is_transformer_param(name: str) -> bool:
    return (
        name == "pos_embed"
        or name == "cls_token"
        or name.startswith("patch_embed.")
        or name.startswith("blocks.")
        or name.startswith("norm.")
    )


def _freeze_transformer(backbone: nn.Module) -> None:
    for name, p in backbone.named_parameters():
        if _is_transformer_param(name):
            p.requires_grad = False


def _unfreeze_last_blocks(backbone: nn.Module, last_n: int) -> None:
    if last_n <= 0:
        return
    num_blocks = len(backbone.blocks)
    start = max(0, num_blocks - last_n)
    for idx, block in enumerate(backbone.blocks):
        requires = idx >= start
        for p in block.parameters():
            p.requires_grad = requires
    if hasattr(backbone, "norm"):
        for p in backbone.norm.parameters():
            p.requires_grad = True


def set_trainable(model: nn.Module, freeze_backbone: bool) -> None:
    if freeze_backbone:
        _freeze_transformer(model.backbone)
        for name, p in model.backbone.named_parameters():
            if not _is_transformer_param(name):
                p.requires_grad = True
    else:
        for p in model.backbone.parameters():
            p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True


def split_param_groups(model: nn.Module) -> Tuple[list, list]:
    backbone_params = []
    head_adapter_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone."):
            subname = name[len("backbone."):]
            if _is_transformer_param(subname):
                backbone_params.append(p)
            else:
                head_adapter_params.append(p)
        else:
            head_adapter_params.append(p)
    return backbone_params, head_adapter_params


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> Tuple[torch.optim.Optimizer, list]:
    backbone_params, head_params = split_param_groups(model)
    param_groups = []
    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": args.lr,
            "weight_decay": args.weight_decay_head,
        })
    if backbone_params:
        # Small backbone LR to avoid overwriting pretrained transformer features.
        param_groups.append({
            "params": backbone_params,
            "lr": args.backbone_lr,
            "weight_decay": args.weight_decay_backbone,
        })
    optimizer = torch.optim.AdamW(param_groups)
    trainable_params = head_params + backbone_params
    return optimizer, trainable_params


def format_param_count(count: int) -> str:
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M"
    if count >= 1_000:
        return f"{count / 1_000:.2f}K"
    return str(count)


def print_trainable_summary(model: nn.Module, prefix: str) -> None:
    backbone_params, head_params = split_param_groups(model)
    backbone_count = sum(p.numel() for p in backbone_params)
    head_count = sum(p.numel() for p in head_params)
    print(f"{prefix} trainable params: backbone={format_param_count(backbone_count)} "
          f"head+adapter={format_param_count(head_count)}")


def log(msg: str, run_logger: RunLogger | None = None) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    if run_logger is not None:
        run_logger.append_event(line)


def main() -> None:
    args = parse_args()
    if args.img_size % 32 != 0:
        raise ValueError("--img-size must be divisible by 32.")
    if args.ckpt and args.timm_model:
        raise ValueError("Provide only one of --ckpt or --timm-model (or neither to use defaults).")
    if "--weight-decay-head" not in sys.argv and args.weight_decay != 0.0:
        args.weight_decay_head = args.weight_decay

    seed_everything(args.seed, args.deterministic)

    run_name = args.run_name or default_run_name(args)
    run_dir = make_run_dir(args.output_dir, run_name)
    run_name = run_dir.name
    run_logger: RunLogger | None = None
    if args.save_logs:
        run_logger = RunLogger(run_dir)

    repo_root = (Path(__file__).resolve().parent / "ViT-Adapter").resolve()
    sys.path.insert(0, str(repo_root / "segmentation"))
    sys.path.insert(0, str(repo_root / "detection"))
    sys.path.insert(0, str(repo_root))

    try:
        from segmentation.mmseg_custom.models.backbones.vit_adapter import ViTAdapter
    except Exception as exc:
        raise RuntimeError(
            "Failed to import ViTAdapter. Install mmcv/mmseg and ensure "
            "ops are built via detection/ops/make.sh."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = not args.deterministic

    pretrain_size = resolve_pretrain_size(args.backbone, args.pretrain_size)
    resolved_timm_model = resolve_timm_model(args.backbone, args.timm_model) if not args.ckpt else ""
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
    if run_logger is not None:
        run_logger.write_json("run_config.json", run_info)
    log(f"[run] run directory: {run_dir}", run_logger)

    backbone = build_backbone(ViTAdapter, pretrain_size, with_cp=args.with_cp)
    model = ViTAdapterLinearProbe(backbone=backbone, num_classes=Vocab.num_classes)
    if args.syncbn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    else:
        model = convert_syncbn_to_bn(model)

    state_dict = load_pretrained_state_dict(args.backbone, args.timm_model, args.ckpt)
    missing, unexpected = model.backbone.load_state_dict(state_dict, strict=False)
    matched = len(model.backbone.state_dict()) - len(missing)
    post_load_actions: List[str] = []
    if (
        args.backbone == "clip"
        and args.clip_zero_missing_patch_embed_bias
        and "patch_embed.proj.bias" in missing
        and hasattr(model.backbone, "patch_embed")
        and hasattr(model.backbone.patch_embed, "proj")
        and getattr(model.backbone.patch_embed.proj, "bias", None) is not None
    ):
        with torch.no_grad():
            model.backbone.patch_embed.proj.bias.zero_()
        post_load_actions.append("zeroed patch_embed.proj.bias (missing in CLIP checkpoint)")
        log("[load] post-load: zeroed patch_embed.proj.bias (missing in CLIP checkpoint)", run_logger)
    load_report = {
        "num_loaded_keys": len(state_dict),
        "num_backbone_keys": len(model.backbone.state_dict()),
        "matched_keys": matched,
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "missing_key_names": sorted(list(missing)),
        "unexpected_key_names": sorted(list(unexpected)),
        "post_load_actions": post_load_actions,
    }
    log(
        f"[load] matched={load_report['matched_keys']} missing={load_report['missing_keys']} "
        f"unexpected={load_report['unexpected_keys']}",
        run_logger,
    )
    if run_logger is not None:
        run_logger.write_json("load_report.json", load_report)

    set_trainable(model, args.freeze_backbone)
    print_trainable_summary(model, "[params]")
    total_param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run_info["model"] = {
        "num_classes": Vocab.num_classes,
        "total_params": total_param_count,
        "trainable_params": trainable_param_count,
    }
    run_info["load_report"] = {
        "matched_keys": load_report["matched_keys"],
        "missing_keys": load_report["missing_keys"],
        "unexpected_keys": load_report["unexpected_keys"],
    }
    if run_logger is not None:
        run_logger.write_json("run_config.json", run_info)

    model.to(device)
    model_forward = model
    if args.torch_compile and device.type == "cuda" and hasattr(torch, "compile"):
        model_forward = torch.compile(model)
    if args.profile_flops:
        flops_info = estimate_flops(model, args.img_size, device)
        run_info["model"]["flops_profile"] = flops_info
        if flops_info.get("available"):
            log(f"[profile] flops_per_image={flops_info['gflops_per_image']:.3f} GFLOPs", run_logger)
        else:
            log(f"[profile] skipped FLOPs profiling: {flops_info.get('error', 'unknown error')}", run_logger)
        if run_logger is not None:
            run_logger.write_json("run_config.json", run_info)

    if args.dry_run:
        model_forward.eval()
        x = torch.randn(1, 3, args.img_size, args.img_size, device=device)
        with torch.no_grad():
            y = model_forward(x)
        log(f"[dry-run] output shape: {tuple(y.shape)}", run_logger)
        if run_logger is not None:
            run_logger.write_json(
                "summary.json",
                {
                    "mode": "dry-run",
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "total_time_sec": time.time() - run_start_ts,
                    "output_shape": tuple(y.shape),
                },
            )
        return

    if not args.data_root:
        raise ValueError("--data-root is required unless --dry-run is set.")

    if args.input_norm == "clip":
        image_mean, image_std = CLIP_MEAN, CLIP_STD
    else:
        image_mean, image_std = IMAGENET_MEAN, IMAGENET_STD
    log(f"[data] input_norm={args.input_norm}", run_logger)
    transform = VocTransform(args.img_size, mean=image_mean, std=image_std)
    try:
        train_set = VOCSegmentation(
            root=args.data_root,
            year="2012",
            image_set="train",
            download=bool(args.download),
            transforms=transform,
        )
        val_set = VOCSegmentation(
            root=args.data_root,
            year="2012",
            image_set="val",
            download=bool(args.download),
            transforms=transform,
        )
    except RuntimeError as e:
        raise RuntimeError(
            f"VOC load failed at data_root='{args.data_root}'. "
            "Expected layout: <data_root>/VOC2012 with subfolders JPEGImages, SegmentationClass, "
            "ImageSets/Segmentation, etc. If VOC2012 lives elsewhere, point --data-root to its parent. "
            "Original error: " + str(e)
        ) from e

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    val_generator = torch.Generator()
    val_generator.manual_seed(args.seed + 1)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=val_generator,
    )

    run_info["dataset"] = {
        "name": "VOC2012",
        "train_size": len(train_set),
        "val_size": len(val_set),
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }
    if run_logger is not None:
        run_logger.write_json("run_config.json", run_info)

    criterion = nn.CrossEntropyLoss(ignore_index=Vocab.ignore_index)
    optimizer, trainable_params = build_optimizer(model, args)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if args.eval_only:
        eval_start = time.time()
        metrics = evaluate(
            model_forward,
            val_loader,
            device,
            Vocab.num_classes,
            Vocab.ignore_index,
            args.miou_ignore_empty,
            args.measure_inference_time,
        )
        eval_time = time.time() - eval_start
        log(
            f"[eval] pixel_acc={metrics['pixel_acc']:.4f} mIoU={metrics['mIoU']:.4f} "
            f"mean_class_acc={metrics['mean_class_acc']:.4f} time={eval_time:.2f}s "
            f"infer={metrics['mean_inference_time_ms']:.2f}ms/img",
            run_logger,
        )
        if run_logger is not None:
            eval_row = {
                "epoch": 0,
                "pixel_acc": metrics["pixel_acc"],
                "mIoU": metrics["mIoU"],
                "mean_class_acc": metrics["mean_class_acc"],
                "eval_time_sec": eval_time,
                "model_forward_time_sec": metrics["model_forward_time_sec"],
                "mean_inference_time_ms": metrics["mean_inference_time_ms"],
                "throughput_img_s": metrics["throughput_img_s"],
                "num_eval_images": metrics["num_eval_images"],
            }
            run_logger.log_eval_epoch(eval_row)
            save_confusion_matrix_csv(run_logger.run_dir, 0, metrics["confusion_matrix"])
            save_class_metrics_csv(
                run_logger.run_dir,
                0,
                metrics["per_class_iou"],
                metrics["per_class_acc"],
                metrics["gt_count"],
                metrics["union"],
            )
            run_logger.write_json(
                "summary.json",
                {
                    "mode": "eval-only",
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "total_time_sec": time.time() - run_start_ts,
                    "eval": eval_row,
                    "target_miou": args.target_miou,
                    "epochs_to_target_miou": 0 if args.target_miou > 0 and metrics["mIoU"] >= args.target_miou else -1,
                },
            )
        return

    did_unfreeze = False
    log_interval = max(1, args.log_interval)
    train_history: List[Dict[str, Any]] = []
    eval_history: List[Dict[str, Any]] = []
    best_miou = float("-inf")
    best_epoch = -1
    interrupted = False
    interrupted_epoch = 0
    interrupted_iter = 0
    try:
        for epoch in range(1, args.epochs + 1):
            interrupted_epoch = epoch
            interrupted_iter = 0
            # Optional staged unfreezing of the last N transformer blocks.
            if (args.freeze_backbone and not did_unfreeze and args.unfreeze_at_epoch >= 0
                    and epoch == args.unfreeze_at_epoch):
                _unfreeze_last_blocks(model.backbone, args.unfreeze_last_n_blocks)
                optimizer, trainable_params = build_optimizer(model, args)
                print_trainable_summary(model, f"[params] after unfreeze@{epoch}")
                did_unfreeze = True

            epoch_start = time.time()
            model_forward.train()
            running_loss = 0.0
            running_steps = 0
            epoch_loss = 0.0
            epoch_steps = 0
            iter_start = time.time()
            for i, (images, targets) in enumerate(train_loader, 1):
                interrupted_iter = i
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model_forward(images)
                    loss = criterion(logits, targets)
                scaler.scale(loss).backward()
                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                scaler.step(optimizer)
                scaler.update()

                loss_val = loss.item()
                running_loss += loss_val
                running_steps += 1
                epoch_loss += loss_val
                epoch_steps += 1

                if i % log_interval == 0:
                    avg = running_loss / max(1, running_steps)
                    elapsed = time.time() - iter_start
                    log(
                        f"[train] epoch={epoch} iter={i}/{len(train_loader)} "
                        f"loss={avg:.4f} batch={images.size(0)} time={elapsed:.2f}s",
                        run_logger,
                    )
                    running_loss = 0.0
                    running_steps = 0
                    iter_start = time.time()

            avg_loss = epoch_loss / max(1, epoch_steps)
            epoch_time = time.time() - epoch_start
            train_row = {
                "epoch": epoch,
                "avg_loss": avg_loss,
                "steps": epoch_steps,
                "epoch_time_sec": epoch_time,
            }
            train_history.append(train_row)
            if run_logger is not None:
                run_logger.log_train_epoch(train_row)
            log(f"[train] epoch={epoch} avg_loss={avg_loss:.4f} time={epoch_time:.2f}s", run_logger)
            interrupted_iter = 0

            if args.eval_every > 0 and epoch % args.eval_every == 0:
                eval_start = time.time()
                metrics = evaluate(
                    model_forward,
                    val_loader,
                    device,
                    Vocab.num_classes,
                    Vocab.ignore_index,
                    args.miou_ignore_empty,
                    args.measure_inference_time,
                )
                eval_time = time.time() - eval_start
                eval_row = {
                    "epoch": epoch,
                    "pixel_acc": metrics["pixel_acc"],
                    "mIoU": metrics["mIoU"],
                    "mean_class_acc": metrics["mean_class_acc"],
                    "eval_time_sec": eval_time,
                    "model_forward_time_sec": metrics["model_forward_time_sec"],
                    "mean_inference_time_ms": metrics["mean_inference_time_ms"],
                    "throughput_img_s": metrics["throughput_img_s"],
                    "num_eval_images": metrics["num_eval_images"],
                }
                eval_history.append(eval_row)
                if best_ckpt_path and eval_row["mIoU"] > best_miou:
                    best_miou = eval_row["mIoU"]
                    best_epoch = epoch
                    os.makedirs(Path(best_ckpt_path).parent, exist_ok=True)
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "args": vars(args),
                            "best": {"epoch": best_epoch, "mIoU": best_miou},
                        },
                        best_ckpt_path,
                    )
                    log(
                        f"[save] best path={best_ckpt_path} epoch={best_epoch} mIoU={best_miou:.4f}",
                        run_logger,
                    )
                if run_logger is not None:
                    run_logger.log_eval_epoch(eval_row)
                    save_confusion_matrix_csv(run_logger.run_dir, epoch, metrics["confusion_matrix"])
                    save_class_metrics_csv(
                        run_logger.run_dir,
                        epoch,
                        metrics["per_class_iou"],
                        metrics["per_class_acc"],
                        metrics["gt_count"],
                        metrics["union"],
                    )
                log(
                    f"[eval] epoch={epoch} pixel_acc={metrics['pixel_acc']:.4f} "
                    f"mIoU={metrics['mIoU']:.4f} mean_class_acc={metrics['mean_class_acc']:.4f} "
                    f"time={eval_time:.2f}s infer={metrics['mean_inference_time_ms']:.2f}ms/img",
                    run_logger,
                )
    except KeyboardInterrupt:
        interrupted = True
        log(
            f"[interrupt] Ctrl+C received at epoch={interrupted_epoch} iter={interrupted_iter}.",
            run_logger,
        )
        if interrupted_ckpt_path:
            os.makedirs(Path(interrupted_ckpt_path).parent, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "interrupt": {
                        "epoch": interrupted_epoch,
                        "iter": interrupted_iter,
                        "time": datetime.now().isoformat(timespec="seconds"),
                    },
                    "best": {"epoch": best_epoch, "mIoU": best_miou if best_epoch >= 0 else None},
                    "train_history": train_history,
                    "eval_history": eval_history,
                },
                interrupted_ckpt_path,
            )
            log(f"[save] interrupted checkpoint path={interrupted_ckpt_path}", run_logger)

    summary: Dict[str, Any] = {
        "mode": "train",
        "status": "interrupted" if interrupted else "completed",
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "total_time_sec": time.time() - run_start_ts,
        "epochs": args.epochs,
        "num_train_points": len(train_history),
        "num_eval_points": len(eval_history),
        "target_miou": args.target_miou,
        "epochs_to_target_miou": first_epoch_reaching(eval_history, args.target_miou),
    }
    if train_history:
        summary["final_train_loss"] = train_history[-1]["avg_loss"]
    if eval_history:
        best = max(eval_history, key=lambda row: row["mIoU"])
        summary["best_mIoU"] = best["mIoU"]
        summary["best_epoch"] = best["epoch"]
        summary["final_eval"] = eval_history[-1]
    if interrupted:
        summary["interrupted_epoch"] = interrupted_epoch
        summary["interrupted_iter"] = interrupted_iter
    if best_epoch >= 0 and best_ckpt_path:
        summary["best_checkpoint_path"] = best_ckpt_path
    if interrupted and interrupted_ckpt_path:
        summary["interrupted_checkpoint_path"] = interrupted_ckpt_path

    if final_ckpt_path and not interrupted:
        os.makedirs(Path(final_ckpt_path).parent, exist_ok=True)
        torch.save({"model": model.state_dict(), "args": vars(args), "summary": summary}, final_ckpt_path)
        summary["checkpoint_path"] = final_ckpt_path
        log(f"[save] {final_ckpt_path} total_time={summary['total_time_sec']:.2f}s", run_logger)

    if run_logger is not None:
        run_logger.write_json("summary.json", summary)


if __name__ == "__main__":
    main()
