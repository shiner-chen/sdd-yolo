"""
SDD-YOLO Dual Attention Module  (RKNN NPU-friendly, auto channel detection)
============================================================================
CBAM-style Channel + Spatial Attention — no Softmax, no graph split on RK3588.

Operators used: AdaptiveAvgPool2d, Conv2d(1×1 + 7×7), SiLU, Sigmoid, mean, max
All natively supported by RKNN 2.x NPU.

Formula (paper Section 4.5):
  A = σ(W_c · GAP(F)) ⊗ σ(Conv_7×7([AvgPool(F); MaxPool(F)]))
"""

import torch
import torch.nn as nn


class DualAttention(nn.Module):
    """Dual Attention: Channel Attention ⊗ Spatial Attention (no Softmax).

    Channel count is auto-detected on the first forward pass so this module
    is compatible with Ultralytics' parse_model channel-scaling without needing
    explicit channel args in the YAML.

    Args:
        reduction (int): Channel squeeze ratio. Default 16.

    YAML usage:
        - [19, 1, DualAttention, []]        # empty args — channels auto-detected
    """

    def __init__(self, reduction: int = 16):
        super().__init__()
        self.reduction = reduction
        # Layers built lazily on first forward (channel count unknown at init)
        self.ca: nn.Module | None = None
        self.sa: nn.Module | None = None

    # ------------------------------------------------------------------
    def _build(self, c1: int) -> None:
        """Build channel & spatial attention layers for `c1` input channels."""
        c_mid = max(c1 // self.reduction, 4)

        # Channel Attention: σ(W_c · GAP(F))
        # 1×1 Conv instead of Linear — equally expressive, more NPU-friendly
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                       # GAP [B,C,H,W]→[B,C,1,1]
            nn.Conv2d(c1, c_mid, 1, bias=False),           # squeeze
            nn.SiLU(),
            nn.Conv2d(c_mid, c1, 1, bias=False),           # excite  (W_c)
            nn.Sigmoid(),
        )

        # Spatial Attention: σ(Conv_7×7(proj(F)))
        # ReduceMean/ReduceMax(axis=1) are NOT supported by RKNN 2.x NPU;
        # replace with a 1×1 Conv projection (c1→1) which is fully NPU-native.
        # A learned projection is more expressive than a fixed mean/max anyway.
        self.sa_proj = nn.Conv2d(c1, 1, 1, bias=False)     # channel collapse
        self.sa = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Lazy build on first call
        if self.ca is None:
            self._build(x.shape[1])
            device, dtype = x.device, x.dtype
            self.ca      = self.ca.to(device=device, dtype=dtype)
            self.sa_proj = self.sa_proj.to(device=device, dtype=dtype)
            self.sa      = self.sa.to(device=device, dtype=dtype)

        # 1. Channel attention
        x = x * self.ca(x)

        # 2. Spatial attention — 1×1 Conv replaces ReduceMean/ReduceMax
        #    all ops are Conv2d + Sigmoid: fully RKNN NPU-native
        sa = self.sa(self.sa_proj(x))   # [B,C,H,W]→[B,1,H,W]→[B,1,H,W]
        return x * sa
