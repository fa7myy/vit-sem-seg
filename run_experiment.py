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
import copy
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

from eval_utils import estimate_flops
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from segexp.backbone import (
    build_probe_model,
    load_backbone_weights,
    print_trainable_summary,
    resolve_pretrain_size,
    resolve_timm_model,
    set_trainable,
    update_model_run_info,
)
from segexp.data import build_voc_loaders
from segexp.logging import (
    RunLogger,
    init_device,
    init_run_context,
    log,
    seed_everything,
    write_run_config,
)
from segexp.train import (
    build_summary,
    build_training_components,
    maybe_run_eval_only,
    maybe_save_final_checkpoint,
    run_training,
)


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
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help=(
            "Train-data percent sweep. Runs one full training per split (same hyperparams). "
            "Examples: --splits 10 25 50 100 OR --splits [10,25,50,100]."
        ),
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=-1,
        help="Seed for deterministic subset selection; -1 uses --seed.",
    )
    parser.add_argument("--output-dir", type=str, default="runs",
                        help="Base directory for run artifacts.")
    parser.add_argument("--run-name", type=str, default="",
                        help="Optional run name (defaults to timestamp_backbone_mode_seed).")
    parser.add_argument("--target-miou", type=float, default=0.0,
                        help="Optional convergence threshold. If >0, logs first epoch reaching this mIoU.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.img_size % 32 != 0:
        raise ValueError("--img-size must be divisible by 32.")
    if args.ckpt and args.timm_model:
        raise ValueError("Provide only one of --ckpt or --timm-model (or neither to use defaults).")


def parse_splits(raw: List[str] | None) -> List[int]:
    if not raw:
        return []
    text = " ".join(raw).strip()
    if not text:
        return []
    for ch in "[]()":
        text = text.replace(ch, " ")
    text = text.replace(",", " ")
    parts = [p for p in text.split() if p]
    splits: List[int] = []
    for p in parts:
        splits.append(int(p))
    for s in splits:
        if s <= 0 or s > 100:
            raise ValueError(f"--splits values must be in 1..100, got {s}")
    # Preserve order but drop duplicates.
    seen = set()
    unique: List[int] = []
    for s in splits:
        if s in seen:
            continue
        seen.add(s)
        unique.append(s)
    return unique


def make_sweep_run_name(args: argparse.Namespace, splits: List[int]) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    split_tag = "-".join(str(s) for s in splits)
    base = args.run_name or f"{timestamp}_{args.backbone}_train_seed{args.seed}_splits{split_tag}"
    return base


def setup_vit_adapter_paths(repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root / "segmentation"))
    sys.path.insert(0, str(repo_root / "detection"))
    sys.path.insert(0, str(repo_root))


def import_vit_adapter() -> Any:
    try:
        from segmentation.mmseg_custom.models.backbones.vit_adapter import ViTAdapter
    except Exception as exc:
        raise RuntimeError(
            "Failed to import ViTAdapter. Install mmcv/mmseg and ensure "
            "ops are built via detection/ops/make.sh."
        ) from exc
    return ViTAdapter


def configure_model_runtime(
    args: argparse.Namespace,
    model: nn.Module,
    device: torch.device,
    run_info: Dict[str, Any],
    run_logger: RunLogger | None,
) -> nn.Module:
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
        write_run_config(run_logger, run_info)
    return model_forward


def maybe_run_dry_run(
    args: argparse.Namespace,
    model_forward: nn.Module,
    device: torch.device,
    run_logger: RunLogger | None,
    run_start_ts: float,
) -> bool:
    if not args.dry_run:
        return False
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
    return True


def run_single(
    args: argparse.Namespace,
    ViTAdapter: Any,
    device: torch.device,
    pretrain_size: int,
    split_percent: int | None,
    split_seed: int | None,
) -> None:
    seed_everything(args.seed, args.deterministic)

    resolved_timm_model = resolve_timm_model(args.backbone, args.timm_model) if not args.ckpt else ""
    (
        _run_dir,
        _run_name,
        run_logger,
        run_info,
        run_start_ts,
        final_ckpt_path,
        best_ckpt_path,
        interrupted_ckpt_path,
    ) = init_run_context(args, device, pretrain_size, resolved_timm_model)

    model = build_probe_model(ViTAdapter, args, pretrain_size)
    load_report = load_backbone_weights(args, model, run_logger)
    set_trainable(model, args.freeze_backbone)
    print_trainable_summary(model, "[params]")
    update_model_run_info(model, run_info, load_report, run_logger)

    model_forward = configure_model_runtime(args, model, device, run_info, run_logger)
    if maybe_run_dry_run(args, model_forward, device, run_logger, run_start_ts):
        return

    train_loader, val_loader = build_voc_loaders(
        args,
        run_logger,
        run_info,
        split_percent=split_percent,
        split_seed=split_seed,
    )
    criterion, optimizer, trainable_params, use_amp, scaler = build_training_components(args, model, device)

    if maybe_run_eval_only(args, model_forward, val_loader, device, run_logger, run_start_ts):
        return

    (
        train_history,
        eval_history,
        _best_miou,
        best_epoch,
        interrupted,
        interrupted_epoch,
        interrupted_iter,
    ) = run_training(
        args=args,
        model=model,
        model_forward=model_forward,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        trainable_params=trainable_params,
        use_amp=use_amp,
        scaler=scaler,
        best_ckpt_path=best_ckpt_path,
        interrupted_ckpt_path=interrupted_ckpt_path,
        run_logger=run_logger,
    )

    summary = build_summary(
        args=args,
        run_start_ts=run_start_ts,
        train_history=train_history,
        eval_history=eval_history,
        interrupted=interrupted,
        interrupted_epoch=interrupted_epoch,
        interrupted_iter=interrupted_iter,
        best_epoch=best_epoch,
        best_ckpt_path=best_ckpt_path,
        interrupted_ckpt_path=interrupted_ckpt_path,
    )
    maybe_save_final_checkpoint(
        final_ckpt_path=final_ckpt_path,
        model=model,
        args=args,
        summary=summary,
        interrupted=interrupted,
        run_logger=run_logger,
    )
    if run_logger is not None:
        run_logger.write_json("summary.json", summary)


def main() -> None:
    args = parse_args()
    validate_args(args)

    splits = parse_splits(args.splits)
    if splits and (args.eval_only or args.dry_run):
        raise ValueError("--splits is only supported for training runs (not with --eval-only or --dry-run).")

    repo_root = (PROJECT_ROOT / "ViT-Adapter").resolve()
    setup_vit_adapter_paths(repo_root)
    ViTAdapter = import_vit_adapter()

    device = init_device(args.deterministic)
    pretrain_size = resolve_pretrain_size(args.backbone, args.pretrain_size)
    split_seed = args.seed if args.split_seed < 0 else int(args.split_seed)

    if not splits:
        run_single(args, ViTAdapter, device, pretrain_size, split_percent=None, split_seed=None)
        return

    base_run_name = make_sweep_run_name(args, splits)
    for split_percent in splits:
        split_args = copy.deepcopy(args)
        split_args.run_name = f"{base_run_name}_p{split_percent}"
        run_single(
            split_args,
            ViTAdapter,
            device,
            pretrain_size,
            split_percent=int(split_percent),
            split_seed=split_seed,
        )


if __name__ == "__main__":
    main()
