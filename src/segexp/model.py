#!/usr/bin/env python3
"""
Segmentation model components: vocabulary, head, and probe wrapper.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Vocab:
    num_classes: int = 21
    ignore_index: int = 255


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
        for i in range(3, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode="bilinear", align_corners=False)
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
