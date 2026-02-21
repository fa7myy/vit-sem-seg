#!/usr/bin/env python3
"""
Training/evaluation loops and checkpoint utilities.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from eval_utils import evaluate, save_class_metrics_csv, save_confusion_matrix_csv
from .backbone import _unfreeze_last_blocks, build_optimizer, print_trainable_summary
from .logging import RunLogger, log
from .model import Vocab


def build_training_components(
    args: Any,
    model: nn.Module,
    device: torch.device,
) -> Tuple[nn.Module, torch.optim.Optimizer, list, bool, torch.cuda.amp.GradScaler]:
    criterion = nn.CrossEntropyLoss(ignore_index=Vocab.ignore_index)
    optimizer, trainable_params = build_optimizer(model, args)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    return criterion, optimizer, trainable_params, use_amp, scaler


def evaluate_once(
    model_forward: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    args: Any,
    epoch: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
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
    return eval_row, metrics, eval_time


def save_eval_artifacts(
    run_logger: RunLogger | None,
    epoch: int,
    eval_row: Dict[str, Any],
    metrics: Dict[str, Any],
) -> None:
    if run_logger is None:
        return
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


def maybe_run_eval_only(
    args: Any,
    model_forward: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    run_logger: RunLogger | None,
    run_start_ts: float,
) -> bool:
    if not args.eval_only:
        return False
    eval_row, metrics, eval_time = evaluate_once(model_forward, val_loader, device, args, epoch=0)
    log(
        f"[eval] pixel_acc={metrics['pixel_acc']:.4f} mIoU={metrics['mIoU']:.4f} "
        f"mean_class_acc={metrics['mean_class_acc']:.4f} time={eval_time:.2f}s "
        f"infer={metrics['mean_inference_time_ms']:.2f}ms/img",
        run_logger,
    )
    if run_logger is not None:
        save_eval_artifacts(run_logger, 0, eval_row, metrics)
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
    return True


def maybe_unfreeze_backbone(
    args: Any,
    model: nn.Module,
    did_unfreeze: bool,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    trainable_params: list,
) -> Tuple[bool, torch.optim.Optimizer, list]:
    if (args.freeze_backbone and not did_unfreeze and args.unfreeze_at_epoch >= 0
            and epoch == args.unfreeze_at_epoch):
        _unfreeze_last_blocks(model.backbone, args.unfreeze_last_n_blocks)
        optimizer, trainable_params = build_optimizer(model, args)
        print_trainable_summary(model, f"[params] after unfreeze@{epoch}")
        did_unfreeze = True
    return did_unfreeze, optimizer, trainable_params


def train_one_epoch(
    epoch: int,
    model_forward: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    trainable_params: list,
    grad_clip: float,
    log_interval: int,
    run_logger: RunLogger | None,
    interrupt_state: Dict[str, int],
) -> Dict[str, Any]:
    epoch_start = time.time()
    model_forward.train()
    running_loss = 0.0
    running_steps = 0
    epoch_loss = 0.0
    epoch_steps = 0
    iter_start = time.time()
    for batch_index, (images, targets) in enumerate(train_loader, 1):
        interrupt_state["iter"] = batch_index
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model_forward(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.item()
        running_loss += loss_val
        running_steps += 1
        epoch_loss += loss_val
        epoch_steps += 1

        if batch_index % log_interval == 0:
            avg = running_loss / max(1, running_steps)
            elapsed = time.time() - iter_start
            log(
                f"[train] epoch={epoch} iter={batch_index}/{len(train_loader)} "
                f"loss={avg:.4f} batch={images.size(0)} time={elapsed:.2f}s",
                run_logger,
            )
            running_loss = 0.0
            running_steps = 0
            iter_start = time.time()

    avg_loss = epoch_loss / max(1, epoch_steps)
    epoch_time = time.time() - epoch_start
    return {
        "epoch": epoch,
        "avg_loss": avg_loss,
        "steps": epoch_steps,
        "epoch_time_sec": epoch_time,
    }


def maybe_save_best_checkpoint(
    best_ckpt_path: str,
    eval_row: Dict[str, Any],
    epoch: int,
    model: nn.Module,
    args: Any,
    best_miou: float,
    best_epoch: int,
    run_logger: RunLogger | None,
) -> Tuple[float, int]:
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
        log(f"[save] best path={best_ckpt_path} epoch={best_epoch} mIoU={best_miou:.4f}", run_logger)
    return best_miou, best_epoch


def save_interrupted_checkpoint(
    interrupted_ckpt_path: str,
    model: nn.Module,
    args: Any,
    interrupted_epoch: int,
    interrupted_iter: int,
    best_epoch: int,
    best_miou: float,
    train_history: List[Dict[str, Any]],
    eval_history: List[Dict[str, Any]],
    run_logger: RunLogger | None,
) -> None:
    if not interrupted_ckpt_path:
        return
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


def run_training(
    args: Any,
    model: nn.Module,
    model_forward: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    trainable_params: list,
    use_amp: bool,
    scaler: torch.cuda.amp.GradScaler,
    best_ckpt_path: str,
    interrupted_ckpt_path: str,
    run_logger: RunLogger | None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float, int, bool, int, int]:
    did_unfreeze = False
    log_interval = max(1, args.log_interval)
    train_history: List[Dict[str, Any]] = []
    eval_history: List[Dict[str, Any]] = []
    best_miou = float("-inf")
    best_epoch = -1
    interrupted = False
    interrupted_epoch = 0
    interrupt_state = {"iter": 0}

    try:
        for epoch in range(1, args.epochs + 1):
            interrupted_epoch = epoch
            interrupt_state["iter"] = 0
            did_unfreeze, optimizer, trainable_params = maybe_unfreeze_backbone(
                args, model, did_unfreeze, epoch, optimizer, trainable_params
            )

            train_row = train_one_epoch(
                epoch=epoch,
                model_forward=model_forward,
                train_loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                criterion=criterion,
                device=device,
                use_amp=use_amp,
                trainable_params=trainable_params,
                grad_clip=args.grad_clip,
                log_interval=log_interval,
                run_logger=run_logger,
                interrupt_state=interrupt_state,
            )
            train_history.append(train_row)
            if run_logger is not None:
                run_logger.log_train_epoch(train_row)
            log(
                f"[train] epoch={epoch} avg_loss={train_row['avg_loss']:.4f} "
                f"time={train_row['epoch_time_sec']:.2f}s",
                run_logger,
            )
            interrupt_state["iter"] = 0

            if args.eval_every > 0 and epoch % args.eval_every == 0:
                eval_row, metrics, eval_time = evaluate_once(model_forward, val_loader, device, args, epoch=epoch)
                eval_history.append(eval_row)
                best_miou, best_epoch = maybe_save_best_checkpoint(
                    best_ckpt_path=best_ckpt_path,
                    eval_row=eval_row,
                    epoch=epoch,
                    model=model,
                    args=args,
                    best_miou=best_miou,
                    best_epoch=best_epoch,
                    run_logger=run_logger,
                )
                save_eval_artifacts(run_logger, epoch, eval_row, metrics)
                log(
                    f"[eval] epoch={epoch} pixel_acc={metrics['pixel_acc']:.4f} "
                    f"mIoU={metrics['mIoU']:.4f} mean_class_acc={metrics['mean_class_acc']:.4f} "
                    f"time={eval_time:.2f}s infer={metrics['mean_inference_time_ms']:.2f}ms/img",
                    run_logger,
                )
    except KeyboardInterrupt:
        interrupted = True
        log(f"[interrupt] Ctrl+C received at epoch={interrupted_epoch} iter={interrupt_state['iter']}.", run_logger)
        save_interrupted_checkpoint(
            interrupted_ckpt_path=interrupted_ckpt_path,
            model=model,
            args=args,
            interrupted_epoch=interrupted_epoch,
            interrupted_iter=interrupt_state["iter"],
            best_epoch=best_epoch,
            best_miou=best_miou,
            train_history=train_history,
            eval_history=eval_history,
            run_logger=run_logger,
        )

    return (
        train_history,
        eval_history,
        best_miou,
        best_epoch,
        interrupted,
        interrupted_epoch,
        interrupt_state["iter"],
    )


def first_epoch_reaching(history: List[Dict[str, Any]], threshold: float) -> int:
    if threshold <= 0:
        return -1
    for row in history:
        if row["mIoU"] >= threshold:
            return int(row["epoch"])
    return -1


def build_summary(
    args: Any,
    run_start_ts: float,
    train_history: List[Dict[str, Any]],
    eval_history: List[Dict[str, Any]],
    interrupted: bool,
    interrupted_epoch: int,
    interrupted_iter: int,
    best_epoch: int,
    best_ckpt_path: str,
    interrupted_ckpt_path: str,
) -> Dict[str, Any]:
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
    return summary


def maybe_save_final_checkpoint(
    final_ckpt_path: str,
    model: nn.Module,
    args: Any,
    summary: Dict[str, Any],
    interrupted: bool,
    run_logger: RunLogger | None,
) -> None:
    if not final_ckpt_path or interrupted:
        return
    os.makedirs(Path(final_ckpt_path).parent, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args), "summary": summary}, final_ckpt_path)
    summary["checkpoint_path"] = final_ckpt_path
    log(f"[save] {final_ckpt_path} total_time={summary['total_time_sec']:.2f}s", run_logger)
