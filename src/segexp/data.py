#!/usr/bin/env python3
"""
VOC transforms and dataloader construction.
"""

from __future__ import annotations

import random
import hashlib
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from .logging import RunLogger, log, write_run_config

try:
    from torchvision.datasets import VOCSegmentation
    from torchvision.transforms import InterpolationMode
    import torchvision.transforms.functional as TF
except Exception as exc:
    raise RuntimeError("torchvision is required for VOCSegmentation") from exc


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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


def resolve_input_norm(args: Any) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    if args.input_norm == "clip":
        return CLIP_MEAN, CLIP_STD
    return IMAGENET_MEAN, IMAGENET_STD


def build_voc_loaders(
    args: Any,
    run_logger: RunLogger | None,
    run_info: Dict[str, Any],
    split_percent: int | None = None,
    split_seed: int | None = None,
) -> Tuple[DataLoader, DataLoader]:
    if not args.data_root:
        raise ValueError("--data-root is required unless --dry-run is set.")

    image_mean, image_std = resolve_input_norm(args)
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
    except RuntimeError as error:
        raise RuntimeError(
            f"VOC load failed at data_root='{args.data_root}'. "
            "Expected layout: <data_root>/VOC2012 with subfolders JPEGImages, SegmentationClass, "
            "ImageSets/Segmentation, etc. If VOC2012 lives elsewhere, point --data-root to its parent. "
            "Original error: " + str(error)
        ) from error

    full_train_size = len(train_set)
    subset_indices = None
    subset_info = None
    if split_percent is not None:
        if split_percent <= 0 or split_percent > 100:
            raise ValueError(f"split_percent must be in 1..100, got {split_percent}")
        effective_seed = int(args.seed) if split_seed is None else int(split_seed)
        if split_percent == 100:
            subset_size = full_train_size
        else:
            # Deterministic rounding; log realized percent explicitly.
            subset_size = max(1, int(round(full_train_size * (split_percent / 100.0))))
        gen = torch.Generator()
        gen.manual_seed(effective_seed)
        perm = torch.randperm(full_train_size, generator=gen).tolist()
        subset_indices = perm[:subset_size]
        sha1 = hashlib.sha1((",".join(str(i) for i in subset_indices)).encode("utf-8")).hexdigest()
        realized_percent = 100.0 * (len(subset_indices) / float(full_train_size)) if full_train_size > 0 else 0.0
        subset_info = {
            "requested_percent": int(split_percent),
            "realized_percent": realized_percent,
            "subset_size": int(len(subset_indices)),
            "full_train_size": int(full_train_size),
            "split_seed": int(effective_seed),
            "selection": "torch.randperm_prefix",
            "indices_sha1": sha1,
        }
        train_set = Subset(train_set, subset_indices)
        if run_logger is not None:
            run_logger.write_json(
                "train_subset_indices.json",
                {
                    "split": subset_info,
                    "indices": subset_indices,
                },
            )

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
        "full_train_size": int(full_train_size),
        "val_size": len(val_set),
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }
    if subset_info is not None:
        run_info["dataset"]["train_subset"] = subset_info
    write_run_config(run_logger, run_info)
    return train_loader, val_loader
