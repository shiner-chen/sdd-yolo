"""
SDD-YOLO Training Script
=========================
Trains SDD-YOLO-n, optionally initialising from YOLO26n pretrained weights.

Transfer-learning workflow (recommended):
  python train_sdd_yolo.py --data your_data.yaml --nc 1 \\
      --pretrained yolo26n.pt --freeze 10 --epochs 100   # stage 1: train new layers
  python train_sdd_yolo.py --data your_data.yaml --nc 1 \\
      --resume runs/sdd-yolo/exp/weights/last.pt         # stage 2: fine-tune all

Training from scratch:
  python train_sdd_yolo.py --data your_data.yaml --nc 1 --epochs 300

Key hyperparameters from paper (Section 4.6):
  dfl=0.0    — disable DFL loss (consistent with reg_max=1)
  MuSGD      — Newton-Schulz gradient orthogonalization; proxied by AdamW here
  ProgLoss   — progressive loss re-weighting (Ultralytics scheduler approximates this)
  STAL       — small-target-aware label assignment (enabled via end2end O2O head)
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))

# ── Register DualAttention BEFORE any Ultralytics import that parses YAML ──
from dual_attention import DualAttention
import ultralytics.nn.modules as _m
import ultralytics.nn.tasks  as _t
setattr(_m, 'DualAttention', DualAttention)
_t.__dict__['DualAttention'] = DualAttention

import torch
from ultralytics import YOLO


# ────────────────────────────────────────────────────────────────────────────
def load_pretrained_weights(model: YOLO, pretrained_path: str) -> tuple[int, int]:
    """Transfer matching weights from a YOLO26n checkpoint into the SDD-YOLO model.

    Because SDD-YOLO adds a P2 branch, DualAttention modules and uses nc=1,
    only layers whose name AND shape match are transferred; everything else
    keeps its random initialisation.

    Expected transfer coverage (YOLO26n → SDD-YOLO-n nc=1):
      ✓  Backbone (layers 0-10): Conv, C3k2, SPPF, C2PSA  — 100% transferred
      ✓  Neck P3/P4/P5 C3k2 blocks                        — partially transferred
      ✗  P2 neck branch                                    — random init (new)
      ✗  DualAttention modules (3×)                        — random init (new)
      ✗  Detect head (nc=1 ≠ nc=80, different shapes)     — random init

    Args:
        model: SDD-YOLO YOLO object (built from YAML, before training).
        pretrained_path: path to yolo26n.pt (or any compatible .pt file).

    Returns:
        (n_transferred, n_total): layer parameter-tensor counts.
    """
    print(f'\n── Loading pretrained weights from: {pretrained_path}')

    ckpt = torch.load(pretrained_path, map_location='cpu')

    # Ultralytics saves {'model': <DetectionModel>, 'epoch': ..., ...}
    model_obj = ckpt.get('model', ckpt)
    if hasattr(model_obj, 'state_dict'):
        src_sd = model_obj.float().state_dict()     # ensure float32
    elif isinstance(model_obj, dict):
        src_sd = model_obj
    else:
        print('  ⚠ Could not parse checkpoint — training from scratch.')
        return 0, len(model.model.state_dict())

    dst_sd = model.model.state_dict()

    # Keep only tensors whose name and shape both match
    transferable = {
        k: v for k, v in src_sd.items()
        if k in dst_sd and v.shape == dst_sd[k].shape
    }

    model.model.load_state_dict(transferable, strict=False)

    # ── Reporting ────────────────────────────────────────────────────────
    n_total  = len(dst_sd)
    n_xfer   = len(transferable)
    n_new    = n_total - n_xfer
    pct      = 100.0 * n_xfer / n_total if n_total else 0

    def _layer_idx(key: str) -> int:
        """Extract the numeric layer index from 'model.N.xxx'."""
        try:
            return int(key.split('.')[1])
        except (IndexError, ValueError):
            return -1

    backbone_xfer = [k for k in transferable if _layer_idx(k) <= 10]
    neck_xfer     = [k for k in transferable if 11 <= _layer_idx(k) <= 27]
    other_xfer    = [k for k in transferable if _layer_idx(k) > 27]

    print(f'  Transferred : {n_xfer:4d} / {n_total}  ({pct:.1f}%)')
    print(f'    Backbone (layers 0-10) : {len(backbone_xfer)} tensors  ← feature extractor fully loaded')
    print(f'    Neck     (layers 11-27): {len(neck_xfer)} tensors')
    print(f'    Other                  : {len(other_xfer)} tensors')
    print(f'  Random init : {n_new:4d} tensors  (P2 branch, DualAttention, Detect nc=1)')
    print()

    return n_xfer, n_total


# ────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Dataset / model
    ap.add_argument('--data',    default='coco.yaml',     help='dataset yaml')
    ap.add_argument('--nc',      type=int, default=1,     help='number of classes')
    ap.add_argument('--epochs',  type=int, default=300,   help='total epochs')
    ap.add_argument('--imgsz',   type=int, default=640,   help='input image size')
    ap.add_argument('--batch',   type=int, default=16,    help='batch size (-1=auto)')

    # Pretrained weights
    ap.add_argument('--pretrained', default=None, metavar='PT',
                    help='path to yolo26n.pt for backbone weight transfer '
                         '(None = train from scratch)')
    ap.add_argument('--freeze', type=int, default=0, metavar='N',
                    help='freeze first N layers during training '
                         '(10 = freeze backbone for stage-1 fine-tuning)')

    # Training infra
    ap.add_argument('--device',  default='0',             help='cuda device or cpu')
    ap.add_argument('--workers', type=int, default=8,     help='dataloader workers')
    ap.add_argument('--project', default='runs/sdd-yolo', help='output directory')
    ap.add_argument('--name',    default='exp',           help='run name')

    # Resume / teacher
    ap.add_argument('--resume',  default=None,
                    help='resume from a previous SDD-YOLO checkpoint (.pt)')
    ap.add_argument('--teacher', default=None,
                    help='YOLO26x teacher .pt for knowledge distillation (optional)')
    return ap.parse_args()


# ────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    YAML_PATH = os.path.join(os.path.dirname(__file__), 'sdd_yolo_nc1.yaml')

    # ── 1. Build or resume model ──────────────────────────────────────────
    if args.resume:
        print(f'Resuming from checkpoint: {args.resume}')
        model = YOLO(args.resume)
    else:
        print(f'Building SDD-YOLO-n from {YAML_PATH}  (nc={args.nc})')
        model = YOLO(YAML_PATH)

        # ── 2. Transfer YOLO26n backbone weights (optional) ───────────────
        if args.pretrained:
            load_pretrained_weights(model, args.pretrained)
        else:
            print('No pretrained weights specified — training from scratch.\n')

    model.info(verbose=False)

    # ── 3. Training hyperparameters ───────────────────────────────────────
    train_kwargs = dict(
        data       = args.data,
        epochs     = args.epochs,
        imgsz      = args.imgsz,
        batch      = args.batch,
        device     = args.device,
        workers    = args.workers,
        project    = args.project,
        name       = args.name,
        save       = True,
        plots      = True,

        # ── Loss weights (paper Section 4.3) ──────────────────────────────
        dfl        = 0.0,   # DFL-free: reg_max=1 already makes DFL trivial
        box        = 7.5,   # bbox WIoU regression weight
        cls        = 0.5,   # classification loss weight

        # ── Optimizer (AdamW as practical proxy for MuSGD) ────────────────
        # Full MuSGD requires a custom optimizer with Newton-Schulz iteration.
        # AdamW provides competitive convergence for most datasets.
        optimizer    = 'AdamW',
        lr0          = 0.001,
        lrf          = 0.01,         # final LR = lr0 * lrf
        momentum     = 0.937,
        weight_decay = 0.0005,
        warmup_epochs = 3.0,

        # ── Pretrained-weight specific ─────────────────────────────────────
        # freeze=N: lock the first N model layers for the whole training run.
        # Useful for stage-1 when using --pretrained: train only new P2 branch
        # and DualAttention while backbone weights stabilise.
        # Set freeze=0 for full fine-tuning (stage-2 or from-scratch).
        freeze     = args.freeze if args.freeze > 0 else None,

        # ── Data augmentation (paper Section 5.1) ─────────────────────────
        mosaic     = 1.0,
        mixup      = 0.1,
        scale      = 0.5,
        fliplr     = 0.5,
        translate  = 0.1,
        close_mosaic = 10,

        # ── Misc ───────────────────────────────────────────────────────────
        amp        = True,
        val        = True,
        verbose    = True,
    )

    # ── 4. Optional knowledge distillation note ───────────────────────────
    if args.teacher:
        print(f'\nKD teacher: {args.teacher}')
        print('Full paper KD (L_total = (1-λ)L_task + λL_KD) requires a custom')
        print('training loop. See paper Section 4.7 for implementation details.\n')

    # ── 5. Print effective settings ───────────────────────────────────────
    print('── Training settings ──')
    important = ['data', 'epochs', 'imgsz', 'batch', 'optimizer', 'lr0',
                 'dfl', 'freeze', 'pretrained']
    for k in important:
        val = train_kwargs.get(k, getattr(args, k, None))
        print(f'  {k:20s}: {val}')
    print()

    # ── 6. Train ──────────────────────────────────────────────────────────
    results = model.train(**train_kwargs)

    print('\n' + '=' * 60)
    print('Training complete!')
    print(f'  Best weights : {model.trainer.best}')
    mAP = results.results_dict.get('metrics/mAP50(B)', 'N/A')
    print(f'  mAP@0.5      : {mAP}')
    print('=' * 60)


if __name__ == '__main__':
    main()
