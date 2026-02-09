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
import os
import time
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from torchvision.datasets import VOCSegmentation
    from torchvision.transforms import InterpolationMode
    import torchvision.transforms.functional as TF
except Exception as exc:
    raise RuntimeError("torchvision is required for VOCSegmentation") from exc


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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
    parser.add_argument("--save", type=str, default="",
                        help="Optional checkpoint path to save after training.")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run a single forward pass on random input and exit.")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision.")
    parser.add_argument("--torch-compile", action="store_true",
                        help="Use torch.compile when available (CUDA + torch>=2.0).")
    add_bool_arg(parser, "freeze-backbone", True,
                 "Freeze ViT-Adapter backbone (linear probe). Use --no-freeze-backbone to train all.")
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
    parser.add_argument("--pretrain-size", type=int, default=0,
                        help="Backbone pretrain resolution (0 selects a sensible default for the chosen backbone).")
    return parser.parse_args()


class Vocab:
    num_classes: int = 21
    ignore_index: int = 255


class VocTransform:
    def __init__(self, size: int):
        self.size = (size, size)

    def __call__(self, image, target):
        image = TF.resize(image, self.size, interpolation=InterpolationMode.BICUBIC)
        target = TF.resize(target, self.size, interpolation=InterpolationMode.NEAREST)
        image = TF.to_tensor(image)
        image = TF.normalize(image, IMAGENET_MEAN, IMAGENET_STD)
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


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor,
                     num_classes: int, ignore_index: int) -> torch.Tensor:
    mask = target != ignore_index
    if mask.sum() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.int64)
    pred = pred[mask]
    target = target[mask]
    hist = torch.bincount(
        num_classes * target + pred,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    return hist.cpu()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             miou_ignore_empty: bool) -> Dict[str, float]:
    model.eval()
    hist = torch.zeros((Vocab.num_classes, Vocab.num_classes), dtype=torch.int64)
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1)
            hist += confusion_matrix(preds, targets, Vocab.num_classes, Vocab.ignore_index)
    diag = torch.diag(hist).float()
    acc = diag.sum() / hist.sum().float().clamp(min=1)
    union = (hist.sum(1) + hist.sum(0) - torch.diag(hist)).float()
    if miou_ignore_empty:
        mask = union > 0
        iu = torch.zeros_like(diag)
        iu[mask] = diag[mask] / union[mask]
        miou = iu[mask].mean().item() if mask.any() else 0.0
    else:
        iu = diag / union.clamp(min=1)
        miou = iu.mean().item()
    return {"pixel_acc": acc.item(), "mIoU": miou}


def main() -> None:
    args = parse_args()
    if args.img_size % 32 != 0:
        raise ValueError("--img-size must be divisible by 32.")
    if args.ckpt and args.timm_model:
        raise ValueError("Provide only one of --ckpt or --timm-model (or neither to use defaults).")
    if "--weight-decay-head" not in sys.argv and args.weight_decay != 0.0:
        args.weight_decay_head = args.weight_decay

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
        torch.backends.cudnn.benchmark = True

    pretrain_size = resolve_pretrain_size(args.backbone, args.pretrain_size)
    backbone = build_backbone(ViTAdapter, pretrain_size, with_cp=args.with_cp)
    model = ViTAdapterLinearProbe(backbone=backbone, num_classes=Vocab.num_classes)
    if args.syncbn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    else:
        model = convert_syncbn_to_bn(model)

    state_dict = load_pretrained_state_dict(args.backbone, args.timm_model, args.ckpt)
    missing, unexpected = model.backbone.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[load] Unexpected keys: {len(unexpected)}")

    set_trainable(model, args.freeze_backbone)
    print_trainable_summary(model, "[params]")
    model.to(device)
    model_forward = model
    if args.torch_compile and device.type == "cuda" and hasattr(torch, "compile"):
        model_forward = torch.compile(model)

    if args.dry_run:
        model_forward.eval()
        x = torch.randn(1, 3, args.img_size, args.img_size, device=device)
        with torch.no_grad():
            y = model_forward(x)
        print(f"[dry-run] output shape: {tuple(y.shape)}")
        return

    if not args.data_root:
        raise ValueError("--data-root is required unless --dry-run is set.")

    transform = VocTransform(args.img_size)
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
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=Vocab.ignore_index)
    optimizer, trainable_params = build_optimizer(model, args)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if args.eval_only:
        metrics = evaluate(model_forward, val_loader, device, args.miou_ignore_empty)
        print(f"[eval] pixel_acc={metrics['pixel_acc']:.4f} mIoU={metrics['mIoU']:.4f}")
        return

    did_unfreeze = False
    log_interval = max(1, args.log_interval)
    for epoch in range(1, args.epochs + 1):
        # Optional staged unfreezing of the last N transformer blocks.
        if (args.freeze_backbone and not did_unfreeze and args.unfreeze_at_epoch >= 0
                and epoch == args.unfreeze_at_epoch):
            _unfreeze_last_blocks(model.backbone, args.unfreeze_last_n_blocks)
            optimizer, trainable_params = build_optimizer(model, args)
            print_trainable_summary(model, f"[params] after unfreeze@{epoch}")
            did_unfreeze = True

        model_forward.train()
        running_loss = 0.0
        running_steps = 0
        epoch_loss = 0.0
        epoch_steps = 0
        iter_start = time.time()
        for i, (images, targets) in enumerate(train_loader, 1):
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
                print(f"[train] epoch={epoch} iter={i}/{len(train_loader)} "
                      f"loss={avg:.4f} batch={images.size(0)} time={elapsed:.2f}s")
                running_loss = 0.0
                running_steps = 0
                iter_start = time.time()

        avg_loss = epoch_loss / max(1, epoch_steps)
        print(f"[train] epoch={epoch} avg_loss={avg_loss:.4f}")

        if args.eval_every > 0 and epoch % args.eval_every == 0:
            metrics = evaluate(model_forward, val_loader, device, args.miou_ignore_empty)
            print(f"[eval] epoch={epoch} pixel_acc={metrics['pixel_acc']:.4f} mIoU={metrics['mIoU']:.4f}")

    if args.save:
        os.makedirs(Path(args.save).parent, exist_ok=True)
        torch.save({"model": model.state_dict(), "args": vars(args)}, args.save)
        print(f"[save] {args.save}")


if __name__ == "__main__":
    main()
