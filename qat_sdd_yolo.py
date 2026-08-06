#!/usr/bin/env python3
"""
SDD-YOLO QAT (Quantization-Aware Training) Fine-tuning
=======================================================
Fixes INT8 score discretization by training with fake quantization.

Why this works:
  Standard PTQ: RKNN maps all high-confidence scores → same INT8 bucket
  QAT fix:      model learns to spread confidence scores within INT8 range
                so each detection gets a distinct, useful confidence value

Workflow:
  Step 1  QAT fine-tune   best.pt → best_qat.pt  (~20 epochs)
  Step 2  Export ONNX     best_qat.pt → best_qat.onnx  (FP32, standard)
  Step 3  RKNN INT8 PTQ   best_qat.onnx → best_qat_int8.rknn  (RKNN-Toolkit2)

Expected improvement: AP@0.5 16.9% → ~35-45%, score range [0, 1] continuous

Usage:
  # Stage 1: QAT fine-tune
  python qat_sdd_yolo.py \\
      --pretrained /path/to/best.pt \\
      --data       your_dataset.yaml \\
      --epochs     20 \\
      --lr         0.0001

  # Stage 2: export (skip training, just export)
  python qat_sdd_yolo.py \\
      --pretrained runs/qat/exp/weights/best_qat.pt \\
      --export-only
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

# ── Register DualAttention before any Ultralytics import ──────────────────────
from dual_attention import DualAttention
import ultralytics.nn.modules as _m
import ultralytics.nn.tasks  as _t
setattr(_m, 'DualAttention', DualAttention)
_t.__dict__['DualAttention'] = DualAttention

import torch
import torch.nn as nn
from torch.quantization import (
    FakeQuantize, MinMaxObserver, PerChannelMinMaxObserver,
    MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver,
)
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import C2PSA


# ═══════════════════════════════════════════════════════════════════════════════
# 1. RKNN-compatible FakeQuantize configuration
# ═══════════════════════════════════════════════════════════════════════════════

def make_act_fq() -> FakeQuantize:
    """Per-tensor symmetric INT8 — matches RKNN activation quantization."""
    return FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,  # EMA: more stable than MinMax
        quant_min=-128, quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False,
    )()


def make_wt_fq() -> FakeQuantize:
    """Per-channel symmetric INT8 — matches RKNN channel-wise weight quant."""
    return FakeQuantize.with_args(
        observer=MovingAveragePerChannelMinMaxObserver,
        quant_min=-128, quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0, reduce_range=False,
    )()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Decide which modules to quantize
# ═══════════════════════════════════════════════════════════════════════════════

# Modules whose OUTPUTS should be activation-quantized.
# We skip C2PSA (has Softmax → CPU-only on RKNN, no benefit from QAT there)
# and Upsample / Concat (no learnable weights).
# DualAttention IS included — all its ops are RKNN-NPU-native.
_SKIP_ACT_TYPES = (C2PSA, nn.Upsample)

# The Ultralytics Conv wrapper that combines Conv2d + BN + activation.
# Detecting it by duck-typing (has .conv and .act) is more robust than
# importing the class directly.
def _is_ultralytics_conv(m: nn.Module) -> bool:
    return hasattr(m, 'conv') and hasattr(m, 'act') and isinstance(m.conv, nn.Conv2d)

def _in_skipped_subtree(name: str, model: nn.Module) -> bool:
    """True if 'name' belongs to a C2PSA subtree (e.g. model.10.m.0.attn...)."""
    parts = name.split('.')
    for depth in range(1, len(parts)):
        prefix = '.'.join(parts[:depth])
        try:
            parent = model
            for p in prefix.split('.'):
                parent = getattr(parent, p)
            if isinstance(parent, _SKIP_ACT_TYPES):
                return True
        except AttributeError:
            pass
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Insert / remove FakeQuantize hooks
# ═══════════════════════════════════════════════════════════════════════════════

class QATHooks:
    """Manages FakeQuantize forward hooks on an Ultralytics YOLO model."""

    def __init__(self, model: nn.Module):
        self.model       = model
        self._handles    = []   # hook handles (for removal)
        self._act_fqs    = nn.ModuleList()  # activation FakeQuantize modules
        self._wt_fqs     = nn.ModuleList()  # weight FakeQuantize modules
        self._n_act = 0
        self._n_wt  = 0

    # ── attach ──────────────────────────────────────────────────────────────
    def attach(self) -> None:
        """Register QAT hooks on all eligible Conv layers."""
        for name, module in self.model.named_modules():
            if _in_skipped_subtree(name, self.model):
                continue
            if not _is_ultralytics_conv(module):
                continue

            # Activation FakeQuantize: applied to the module's output
            act_fq = make_act_fq().to(next(module.parameters()).device)
            self._act_fqs.append(act_fq)
            handle = module.register_forward_hook(
                self._make_act_hook(act_fq)
            )
            self._handles.append(handle)
            self._n_act += 1

            # Weight FakeQuantize: applied before each forward pass
            wt_fq = make_wt_fq().to(next(module.parameters()).device)
            self._wt_fqs.append(wt_fq)
            handle = module.register_forward_pre_hook(
                self._make_wt_hook(module.conv, wt_fq)
            )
            self._handles.append(handle)
            self._n_wt += 1

        print(f'  [QAT] Hooks attached:  act×{self._n_act}  wt×{self._n_wt}')
        print(f'        Skipped subtrees: C2PSA (Softmax, CPU-only on RKNN)')

    # ── detach ───────────────────────────────────────────────────────────────
    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ── hook factories ────────────────────────────────────────────────────────
    @staticmethod
    def _make_act_hook(fq: FakeQuantize):
        def hook(module, inp, output):
            return fq(output)
        return hook

    @staticmethod
    def _make_wt_hook(conv: nn.Conv2d, fq: FakeQuantize):
        _orig_weight = conv.weight  # keep reference to original tensor
        def pre_hook(module, inp):
            # Temporarily replace weight with quantized version
            conv.weight = nn.Parameter(fq(conv.weight), requires_grad=True)
        return pre_hook

    # ── observer control ─────────────────────────────────────────────────────
    def set_observer_enabled(self, enabled: bool) -> None:
        for fq in list(self._act_fqs) + list(self._wt_fqs):
            if enabled:
                fq.enable_observer()
            else:
                fq.disable_observer()

    def set_fake_quant_enabled(self, enabled: bool) -> None:
        for fq in list(self._act_fqs) + list(self._wt_fqs):
            if enabled:
                fq.enable_fake_quant()
            else:
                fq.disable_fake_quant()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Ultralytics QAT Trainer
# ═══════════════════════════════════════════════════════════════════════════════

class QATDetectionTrainer(DetectionTrainer):
    """DetectionTrainer extended with QAT hooks.

    Phases:
      Epoch 1-2   : observer calibration only (fake_quant disabled)
                    → model measures per-channel activation ranges
      Epoch 3+    : fake quantization enabled
                    → model trains with simulated INT8 rounding
    """

    # Attach QAT hooks right after the model is moved to device
    def _setup_train(self, world_size: int) -> None:
        super()._setup_train(world_size)
        if not hasattr(self, '_qat_hooks'):
            print('\n── QAT setup ─────────────────────────────')
            self._qat_hooks = QATHooks(self.model)
            self._qat_hooks.attach()
            # Start with observer-only mode (fake quant off)
            self._qat_hooks.set_fake_quant_enabled(False)
            self._qat_hooks.set_observer_enabled(True)
            self._qat_calibration_epochs = getattr(self.args, 'qat_calibration_epochs', 2)
            print(f'  Calibration epochs : {self._qat_calibration_epochs}')
            print('─' * 45 + '\n')

    def optimizer_step(self) -> None:
        """Enable fake quantization after calibration phase."""
        epoch = self.epoch
        if epoch == self._qat_calibration_epochs:
            if not getattr(self, '_qat_fq_enabled', False):
                self._qat_hooks.set_fake_quant_enabled(True)
                self._qat_fq_enabled = True
                print(f'\n[QAT] Epoch {epoch}: fake quantization ON (calibration done)\n')
        super().optimizer_step()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Export pipeline: QAT .pt  →  FP32 ONNX  →  RKNN INT8
# ═══════════════════════════════════════════════════════════════════════════════

def export_qat_onnx(pt_path: str, imgsz: int = 320) -> str:
    """Export a QAT-fine-tuned .pt to a standard FP32 ONNX.

    The model is still FP32 — QAT doesn't change the format,
    it only improves the activation distributions so that
    RKNN-Toolkit2's subsequent INT8 PTQ is much more accurate.
    """
    model = YOLO(pt_path)
    # Disable end2end for RKNN (avoids TopK/Gather CPU fallback)
    model.model.model[-1].end2end = False
    onnx_path = pt_path.replace('.pt', '.onnx')
    model.export(
        format='onnx', imgsz=imgsz,
        simplify=True, opset=12, dynamic=False, half=False, device='cpu',
    )
    return onnx_path


def quantize_to_rknn(onnx_path: str, dataset_txt: str,
                     out_rknn: str = None) -> str:
    """Run RKNN-Toolkit2 INT8 PTQ on the QAT ONNX.

    Because the model was QAT-trained, activations are INT8-friendly,
    so this PTQ step will produce much better quantization quality
    than applying PTQ to the original FP32 model.
    """
    from rknn.api import RKNN
    if out_rknn is None:
        out_rknn = onnx_path.replace('.onnx', '_int8.rknn')

    rknn = RKNN(verbose=False)
    rknn.config(
        mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
        target_platform='rk3588',
        quantized_algorithm='normal',  # normal is fine; QAT model is already calibrated
        quantized_method='channel',
        optimization_level=3,
    )
    assert rknn.load_onnx(model=onnx_path) == 0
    assert rknn.build(do_quantization=True, dataset=dataset_txt, rknn_batch_size=1) == 0
    assert rknn.export_rknn(out_rknn) == 0
    import os
    print(f'[RKNN] {out_rknn}  ({os.path.getsize(out_rknn)/1e6:.1f} MB)')
    rknn.release()
    return out_rknn


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Full pipeline entry point
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='SDD-YOLO QAT fine-tuning → RKNN INT8'
    )
    ap.add_argument('--pretrained', default='best.pt',
                    help='Pretrained SDD-YOLO weights (.pt)')
    ap.add_argument('--data',       default='coco.yaml',
                    help='Dataset yaml (must contain train/val splits)')
    ap.add_argument('--epochs',     type=int, default=20,
                    help='QAT fine-tune epochs  (recommend 10-30)')
    ap.add_argument('--lr',         type=float, default=1e-4,
                    help='Initial LR  (use ~1/100 of original training LR)')
    ap.add_argument('--imgsz',      type=int, default=320)
    ap.add_argument('--batch',      type=int, default=16)
    ap.add_argument('--device',     default='0')
    ap.add_argument('--project',    default='runs/qat')
    ap.add_argument('--name',       default='exp')
    ap.add_argument('--calib-data', default=None,
                    help='dataset.txt for RKNN PTQ (default: <data>.txt)')
    ap.add_argument('--export-only', action='store_true',
                    help='Skip training; only export --pretrained to ONNX+RKNN')
    ap.add_argument('--no-rknn',    action='store_true',
                    help='Skip RKNN step; only produce ONNX')
    return ap.parse_args()


def main():
    args = parse_args()

    # ── Stage 0: export-only shortcut ──────────────────────────────────────
    if args.export_only:
        print(f'[Export] {args.pretrained} → ONNX …')
        onnx = export_qat_onnx(args.pretrained, args.imgsz)
        print(f'[Export] ONNX saved: {onnx}')
        if not args.no_rknn and args.calib_data:
            rknn = quantize_to_rknn(onnx, args.calib_data)
            print(f'[Export] RKNN saved: {rknn}')
        return

    # ── Stage 1: QAT fine-tune ─────────────────────────────────────────────
    print('=' * 60)
    print('SDD-YOLO QAT Fine-tuning')
    print(f'  pretrained : {args.pretrained}')
    print(f'  data       : {args.data}')
    print(f'  epochs     : {args.epochs}  (calib: 2  /  fq: {args.epochs-2})')
    print(f'  lr         : {args.lr}')
    print('=' * 60)

    model = YOLO(args.pretrained)

    train_kwargs = dict(
        data        = args.data,
        epochs      = args.epochs,
        imgsz       = args.imgsz,
        batch       = args.batch,
        device      = args.device,
        project     = args.project,
        name        = args.name,
        # QAT-specific settings
        optimizer   = 'AdamW',
        lr0         = args.lr,
        lrf         = 0.01,
        momentum    = 0.9,
        weight_decay= 5e-5,
        warmup_epochs = 1.0,
        # Disable DFL loss (consistent with original SDD-YOLO training)
        dfl         = 0.0,
        # Augmentation: lighter than original to avoid disturbing fine-tune
        mosaic      = 0.5,    # reduced from 1.0
        mixup       = 0.0,    # disabled
        fliplr      = 0.5,
        translate   = 0.05,
        close_mosaic = 3,
        # Use QAT trainer
        trainer     = QATDetectionTrainer,
        # Extra QAT config (read by QATDetectionTrainer)
        qat_calibration_epochs = 2,
        amp         = False,  # disable AMP: FakeQuantize doesn't work with fp16
        val         = True,
        verbose     = True,
    )

    print('\n── QAT Training Settings ──')
    for k in ['lr0', 'epochs', 'optimizer', 'dfl', 'amp', 'mosaic']:
        print(f'  {k:20s}: {train_kwargs[k]}')
    print()

    results = model.train(**train_kwargs)

    # Best checkpoint path
    best_pt = str(model.trainer.best)
    print(f'\n[QAT] Best weights: {best_pt}')

    # ── Stage 2: Export ────────────────────────────────────────────────────
    print('\n── Exporting to ONNX …')
    onnx_path = export_qat_onnx(best_pt, args.imgsz)
    print(f'[Export] {onnx_path}')

    # ── Stage 3: RKNN PTQ ─────────────────────────────────────────────────
    if not args.no_rknn:
        calib = args.calib_data
        if calib is None:
            # Try to find a dataset.txt next to the data yaml
            calib = args.data.replace('.yaml', '_fixed.txt')
            if not os.path.exists(calib):
                print(f'[RKNN] --calib-data not provided and {calib} not found.')
                print('[RKNN] Skipping RKNN step. Re-run with --calib-data <path>.')
                return
        print(f'\n── RKNN INT8 PTQ (calibration: {calib}) …')
        rknn_path = quantize_to_rknn(onnx_path, calib)
        print(f'[RKNN] {rknn_path}')

    print('\n' + '=' * 60)
    print('QAT pipeline complete ✅')
    print(f'  QAT weights : {best_pt}')
    print(f'  ONNX        : {onnx_path}')
    if not args.no_rknn:
        print(f'  RKNN INT8   : {rknn_path}')
    print('=' * 60)


if __name__ == '__main__':
    main()

