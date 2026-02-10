#!/usr/bin/env python3
"""
Evaluation utilities for semantic segmentation experiments.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


VOC_CLASS_NAMES = (
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


def confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> torch.Tensor:
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


def metrics_from_hist(hist: torch.Tensor, miou_ignore_empty: bool) -> Dict[str, Any]:
    diag = torch.diag(hist).float()
    gt_count = hist.sum(1).float()
    pred_count = hist.sum(0).float()
    total = hist.sum().float().clamp(min=1)
    pixel_acc = (diag.sum() / total).item()

    union = gt_count + pred_count - diag
    per_class_iou = diag / union.clamp(min=1)
    if miou_ignore_empty:
        iou_mask = union > 0
    else:
        iou_mask = torch.ones_like(union, dtype=torch.bool)
    miou = per_class_iou[iou_mask].mean().item() if iou_mask.any() else 0.0

    per_class_acc = diag / gt_count.clamp(min=1)
    class_acc_mask = gt_count > 0
    mean_class_acc = per_class_acc[class_acc_mask].mean().item() if class_acc_mask.any() else 0.0

    return {
        "pixel_acc": pixel_acc,
        "mIoU": miou,
        "mean_class_acc": mean_class_acc,
        "per_class_iou": per_class_iou.tolist(),
        "per_class_acc": per_class_acc.tolist(),
        "gt_count": gt_count.long().tolist(),
        "union": union.long().tolist(),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    miou_ignore_empty: bool,
    measure_inference_time: bool,
) -> Dict[str, Any]:
    model.eval()
    hist = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    total_images = 0
    model_forward_time_sec = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if measure_inference_time and device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_start = time.time()
            logits = model(images)
            if measure_inference_time and device.type == "cuda":
                torch.cuda.synchronize(device)
            if measure_inference_time:
                model_forward_time_sec += time.time() - forward_start
            preds = logits.argmax(dim=1)
            hist += confusion_matrix(preds, targets, num_classes, ignore_index)
            total_images += images.size(0)
    metrics = metrics_from_hist(hist, miou_ignore_empty)
    metrics["confusion_matrix"] = hist
    metrics["num_eval_images"] = total_images
    metrics["model_forward_time_sec"] = model_forward_time_sec
    if total_images > 0 and model_forward_time_sec > 0:
        metrics["mean_inference_time_ms"] = 1000.0 * model_forward_time_sec / total_images
        metrics["throughput_img_s"] = total_images / model_forward_time_sec
    else:
        metrics["mean_inference_time_ms"] = 0.0
        metrics["throughput_img_s"] = 0.0
    return metrics


def save_confusion_matrix_csv(
    output_dir: Path,
    epoch: int,
    hist: torch.Tensor,
    class_names: Tuple[str, ...] = VOC_CLASS_NAMES,
) -> Path:
    path = output_dir / f"confusion_matrix_epoch_{epoch:03d}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gt/pred", *class_names])
        for row_idx in range(hist.shape[0]):
            writer.writerow([class_names[row_idx], *hist[row_idx].tolist()])
    return path


def save_class_metrics_csv(
    output_dir: Path,
    epoch: int,
    class_iou: List[float],
    class_acc: List[float],
    gt_count: List[int],
    union: List[int],
    class_names: Tuple[str, ...] = VOC_CLASS_NAMES,
) -> Path:
    path = output_dir / f"class_metrics_epoch_{epoch:03d}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=("class_id", "class_name", "iou", "class_acc", "gt_pixels", "union_pixels"),
        )
        writer.writeheader()
        for class_id, class_name in enumerate(class_names):
            writer.writerow(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "iou": class_iou[class_id],
                    "class_acc": class_acc[class_id],
                    "gt_pixels": gt_count[class_id],
                    "union_pixels": union[class_id],
                }
            )
    return path


def estimate_flops(model: nn.Module, img_size: int, device: torch.device) -> Dict[str, Any]:
    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception as exc:
        return {"available": False, "error": f"fvcore import failed: {exc}"}

    was_training = model.training
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    try:
        with torch.no_grad():
            flops = float(FlopCountAnalysis(model, x).total())
    except Exception as exc:
        if was_training:
            model.train()
        return {"available": False, "error": f"FLOPs analysis failed: {exc}"}
    if was_training:
        model.train()
    return {
        "available": True,
        "flops_per_image": flops,
        "gflops_per_image": flops / 1e9,
        "input_shape": [1, 3, img_size, img_size],
    }
