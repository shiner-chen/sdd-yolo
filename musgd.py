"""
MuSGD Optimizer for SDD-YOLO
==============================
Vision-centric adaptation of the Muon optimizer (Moonshot AI, 2025),
as described in SDD-YOLO Section 4.6.

Core idea — for every 2-D weight matrix W with gradient G:
  1. Accumulate Nesterov momentum buffer B
  2. Orthogonalize B via Newton-Schulz iteration:  B̂ = NS(B)
  3. Update:  W ← W − η · B̂  (scaled to preserve gradient norm)

This keeps the update direction orthonormal, preventing rank collapse of
weight matrices under sparse small-target supervision.

1-D parameters (bias, BatchNorm γ/β) bypass Newton-Schulz and use plain
SGD with momentum, exactly as in the paper.

Reference implementation:
  Kosson et al., "Muon: An optimizer for hidden layers in neural networks"
  https://github.com/KellerJordan/Muon
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.optim import Optimizer


# ─────────────────────────────────────────────────────────────────────────────
def newton_schulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Approximate orthogonal polar factor of G via Newton-Schulz iteration.

    Converges in 5 steps for well-conditioned matrices.  The coefficients
    (a, b, c) are chosen so the quintic polynomial x*(a + b*x²+ c*x⁴)
    converges to the sign function on [-1, 1].

    Args:
        G: gradient tensor, must have ndim >= 2.
        steps: number of Newton-Schulz iterations (5 is standard).
        eps: small constant to prevent division by zero.

    Returns:
        Orthogonalised tensor with same dtype and shape as G.
    """
    assert G.ndim >= 2, f"NS requires ndim>=2, got shape {G.shape}"
    a, b, c = (3.4445, -4.7750, 2.0315)

    # Work in bfloat16 for numerical stability (same trick as Muon paper)
    X = G.reshape(G.shape[0], -1).bfloat16()
    X = X / (X.norm() + eps)

    # NS requires more columns than rows; transpose if necessary
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        X = a * X + b * (A @ X) + c * (A @ (A @ X))

    if transposed:
        X = X.T

    return X.reshape(G.shape).to(G.dtype)


# ─────────────────────────────────────────────────────────────────────────────
class MuSGD(Optimizer):
    """MuSGD: Nesterov SGD with Newton-Schulz gradient orthogonalization.

    Parameter groups are split automatically by the helper
    ``build_musgd_param_groups()``:
      • **matrix** (ndim ≥ 2) — Newton-Schulz orthogonalized Nesterov SGD
      • **vector** (ndim < 2)  — plain Nesterov SGD (bias / BN params)

    Args:
        params: iterable of parameters or param-group dicts.
        lr (float): learning rate. Paper uses 0.01 for backbone matrices.
        momentum (float): Nesterov momentum coefficient. Default 0.95.
        nesterov (bool): use Nesterov momentum (True follows paper).
        ns_steps (int): Newton-Schulz iterations. 5 is standard.
        weight_decay (float): L2 regularisation (applied before momentum).

    Example::

        param_groups = build_musgd_param_groups(model, backbone_lr=0.01)
        optimizer = MuSGD(param_groups, lr=0.01, momentum=0.95)
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr         = group["lr"]
            momentum   = group["momentum"]
            nesterov   = group["nesterov"]
            ns_steps   = group["ns_steps"]
            decay      = group["weight_decay"]
            use_ns     = group.get("use_ns", True)   # False for vector params

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad.data

                # Weight decay (L2 regularisation)
                if decay != 0.0:
                    g = g.add(p.data, alpha=decay)

                # Momentum buffer
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(g)
                buf = state["buf"]

                # Standard Nesterov momentum accumulation
                buf.mul_(momentum).add_(g)          # buf = m*buf + g
                if nesterov:
                    g = g.add(buf, alpha=momentum)  # g  = g  + m*buf  (Nesterov)
                else:
                    g = buf

                # ── Newton-Schulz orthogonalization (matrix params only) ──
                if use_ns and g.ndim >= 2:
                    g_ns = newton_schulz5(g, steps=ns_steps)
                    # Scale to preserve gradient norm (prevent magnitude collapse)
                    scale = g.norm() / (g_ns.norm() + 1e-7)
                    g = g_ns * scale

                p.data.add_(g, alpha=-lr)

        return loss


# ─────────────────────────────────────────────────────────────────────────────
def build_musgd_param_groups(
    model: torch.nn.Module,
    backbone_layers: int = 10,
    matrix_lr: float = 0.01,
    vector_lr: float = 0.001,
    neck_head_lr: float = 0.001,
    weight_decay: float = 0.0005,
    momentum: float = 0.95,
    ns_steps: int = 5,
) -> list[dict]:
    """Build three parameter groups for MuSGD training.

    Groups:
      0 — backbone 2-D weight matrices → MuSGD (NS orthogonalization ON)
      1 — backbone 1-D params (bias, BN) → Nesterov SGD (NS OFF)
      2 — neck / head all params → Nesterov SGD (NS OFF, higher LR for new layers)

    Args:
        model: the SDD-YOLO DetectionModel.
        backbone_layers: last backbone layer index (inclusive). Default 10.
        matrix_lr: learning rate for backbone weight matrices.
        vector_lr: learning rate for backbone biases / BN params.
        neck_head_lr: learning rate for neck + detection head layers.
        weight_decay: L2 regularisation coefficient.
        momentum: Nesterov momentum.
        ns_steps: Newton-Schulz iterations.

    Returns:
        List of param-group dicts ready for ``MuSGD(param_groups, ...)``.
    """
    backbone_matrices = []
    backbone_vectors  = []
    neck_head_params  = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # model.model.N.xxx — extract layer index N
        parts = name.split(".")
        try:
            layer_idx = int(parts[1])           # 'model.N.weight'
        except (IndexError, ValueError):
            layer_idx = 999                     # treat as non-backbone

        if layer_idx <= backbone_layers:
            if param.ndim >= 2:
                backbone_matrices.append(param)  # Conv weight, C3k2 weight …
            else:
                backbone_vectors.append(param)   # bias, BN γ/β
        else:
            neck_head_params.append(param)       # neck + P2 + DualAttention + Detect

    return [
        # Group 0: backbone 2-D matrices — MuSGD (NS ON)
        dict(
            params       = backbone_matrices,
            lr           = matrix_lr,
            weight_decay = weight_decay,
            momentum     = momentum,
            ns_steps     = ns_steps,
            use_ns       = True,
            nesterov     = True,
            label        = "backbone_matrix",
        ),
        # Group 1: backbone 1-D params — Nesterov SGD (NS OFF)
        dict(
            params       = backbone_vectors,
            lr           = vector_lr,
            weight_decay = 0.0,                  # no WD on bias/BN
            momentum     = momentum,
            ns_steps     = ns_steps,
            use_ns       = False,
            nesterov     = True,
            label        = "backbone_vector",
        ),
        # Group 2: neck + head — Nesterov SGD (NS OFF, higher base LR)
        dict(
            params       = neck_head_params,
            lr           = neck_head_lr,
            weight_decay = weight_decay,
            momentum     = momentum,
            ns_steps     = ns_steps,
            use_ns       = False,
            nesterov     = True,
            label        = "neck_head",
        ),
    ]
