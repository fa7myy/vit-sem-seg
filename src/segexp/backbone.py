#!/usr/bin/env python3
"""
Backbone construction, pretrained loading, and trainable parameter policies.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .logging import RunLogger, log, write_run_config
from .model import ViTAdapterLinearProbe, Vocab


DEFAULT_TIMM_MODELS = {
    "dinov2": "vit_base_patch14_dinov2.lvd142m",
    "clip": "clip_vit_base_patch16_224.openai",
    "mae": "mae_vit_base_patch16",
}

DEFAULT_PRETRAIN_SIZE = {
    "dinov2": 592,
    "clip": 224,
    "mae": 224,
}


def _clean_state_dict(raw: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in raw.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if "mask_token" in key:
            continue
        if key.startswith("head.") or key.startswith("fc_norm"):
            continue
        key = key.replace("ls1.gamma", "gamma1").replace("ls2.gamma", "gamma2")
        if key == "patch_embed.proj.weight" and value.ndim == 4 and value.shape[-1] != 16:
            value = F.interpolate(value, size=(16, 16), mode="bilinear", align_corners=False)
        cleaned[key] = value
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
    return _clean_state_dict(model.state_dict())


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
    return load_state_dict_from_timm(resolve_timm_model(backbone, timm_model))


def convert_syncbn_to_bn(module: nn.Module) -> nn.Module:
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


def build_backbone(ViTAdapter: Any, pretrain_size: int, with_cp: bool) -> nn.Module:
    return ViTAdapter(
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
        window_attn=[True, True, False, True, True, False, True, True, False, True, True, False],
        window_size=[14, 14, None, 14, 14, None, 14, 14, None, 14, 14, None],
        pretrained=None,
        with_cp=with_cp,
    )


def build_probe_model(ViTAdapter: Any, args: Any, pretrain_size: int) -> nn.Module:
    backbone = build_backbone(ViTAdapter, pretrain_size, with_cp=args.with_cp)
    model = ViTAdapterLinearProbe(backbone=backbone, num_classes=Vocab.num_classes)
    if args.syncbn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    else:
        model = convert_syncbn_to_bn(model)
    return model


def load_backbone_weights(args: Any, model: nn.Module, run_logger: RunLogger | None) -> Dict[str, Any]:
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
    return load_report


def update_model_run_info(
    model: nn.Module,
    run_info: Dict[str, Any],
    load_report: Dict[str, Any],
    run_logger: RunLogger | None,
) -> None:
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
    write_run_config(run_logger, run_info)


def _is_transformer_param(name: str) -> bool:
    return (
        name == "pos_embed"
        or name == "cls_token"
        or name.startswith("patch_embed.")
        or name.startswith("blocks.")
        or name.startswith("norm.")
    )


def _freeze_transformer(backbone: nn.Module) -> None:
    for name, parameter in backbone.named_parameters():
        if _is_transformer_param(name):
            parameter.requires_grad = False


def _unfreeze_last_blocks(backbone: nn.Module, last_n: int) -> None:
    if last_n <= 0:
        return
    num_blocks = len(backbone.blocks)
    start = max(0, num_blocks - last_n)
    for block_index, block in enumerate(backbone.blocks):
        requires = block_index >= start
        for parameter in block.parameters():
            parameter.requires_grad = requires
    if hasattr(backbone, "norm"):
        for parameter in backbone.norm.parameters():
            parameter.requires_grad = True


def set_trainable(model: nn.Module, freeze_backbone: bool) -> None:
    if freeze_backbone:
        _freeze_transformer(model.backbone)
        for name, parameter in model.backbone.named_parameters():
            if not _is_transformer_param(name):
                parameter.requires_grad = True
    else:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = True
    for parameter in model.head.parameters():
        parameter.requires_grad = True


def split_param_groups(model: nn.Module) -> Tuple[list, list]:
    backbone_params = []
    head_adapter_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone."):
            subname = name[len("backbone."):]
            if _is_transformer_param(subname):
                backbone_params.append(parameter)
            else:
                head_adapter_params.append(parameter)
        else:
            head_adapter_params.append(parameter)
    return backbone_params, head_adapter_params


def build_optimizer(model: nn.Module, args: Any) -> Tuple[torch.optim.Optimizer, list]:
    backbone_params, head_params = split_param_groups(model)
    param_groups = []
    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": args.lr,
            "weight_decay": args.weight_decay_head,
        })
    if backbone_params:
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
    backbone_count = sum(parameter.numel() for parameter in backbone_params)
    head_count = sum(parameter.numel() for parameter in head_params)
    print(
        f"{prefix} trainable params: backbone={format_param_count(backbone_count)} "
        f"head+adapter={format_param_count(head_count)}"
    )
