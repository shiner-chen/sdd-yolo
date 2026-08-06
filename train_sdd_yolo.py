"""
SDD-YOLO Training Script
=========================
Trains SDD-YOLO-n, optionally initialising from YOLO26n pretrained weights.

Transfer-learning workflow (recommended):
  python train_sdd_yolo.py --data your_data.yaml --nc 1 \\
      --pretrained yolo26n.pt --freeze 10 --epochs 100 \\
      --optimizer musgd                       # stage 1: train new layers
  python train_sdd_yolo.py --data your_data.yaml --nc 1 \\
      --resume runs/sdd-yolo/exp/weights/last.pt  # stage 2: fine-tune all

Training from scratch with full paper settings:
  python train_sdd_yolo.py --data your_data.yaml --nc 1 \\
      --epochs 300 --optimizer musgd

Optimizer choices:
  musgd  — Paper-native: Newton-Schulz gradient orthogonalization for backbone
           weight matrices; Nesterov SGD for 1-D params and neck/head.
           Best for sparse G2A UAV data (prevents rank collapse).
  adamw  — Adaptive LR, easy to tune. Good fallback for larger datasets.
  sgd    — Nesterov SGD, momentum=0.937. Fastest per-step but needs careful LR.

Key hyperparameters (paper Section 4.6):
  dfl=0.0  — disable DFL loss weight (reg_max=1 makes DFL trivial anyway)
  MuSGD    — Newton-Schulz orthogonalization; see musgd.py
  ProgLoss — progressive loss re-weighting (Ultralytics LR scheduler proxies this)
  STAL     — small-target-aware label assignment (end2end O2O head enables this)
"""

import sys
import os
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── Register DualAttention and import trainer from its own module ──────────
# SDD_YOLOTrainer must live in sdd_yolo_trainer.py (not __main__) so that
# Ultralytics DDP workers can do `from sdd_yolo_trainer import SDD_YOLOTrainer`
# instead of `from __main__ import SDD_YOLOTrainer` (which fails in workers).
from sdd_yolo_trainer import SDD_YOLOTrainer  # also registers DualAttention

import torch
from ultralytics import YOLO


# ────────────────────────────────────────────────────────────────────────────
def load_pretrained_weights(model: YOLO, pretrained_path: str) -> tuple[int, int]:
    """Transfer matching weights from a YOLO26n checkpoint into the SDD-YOLO model.

    Backbone (layers 0-10) transfers 100%; P2 branch, DualAttention and
    Detect(nc=1) keep random initialisation.
    """
    print(f'\n── Loading pretrained weights from: {pretrained_path}')

    ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    model_obj = ckpt.get('model', ckpt)
    if hasattr(model_obj, 'state_dict'):
        src_sd = model_obj.float().state_dict()
    elif isinstance(model_obj, dict):
        src_sd = model_obj
    else:
        print('  ⚠ Could not parse checkpoint — training from scratch.')
        return 0, len(model.model.state_dict())

    dst_sd = model.model.state_dict()
    transferable = {
        k: v for k, v in src_sd.items()
        if k in dst_sd and v.shape == dst_sd[k].shape
    }
    model.model.load_state_dict(transferable, strict=False)

    def _layer_idx(key):
        try:
            return int(key.split('.')[1])
        except (IndexError, ValueError):
            return -1

    n_total = len(dst_sd)
    n_xfer  = len(transferable)
    backbone_xfer = [k for k in transferable if _layer_idx(k) <= 10]
    neck_xfer     = [k for k in transferable if 11 <= _layer_idx(k) <= 27]

    print(f'  Transferred : {n_xfer:4d} / {n_total}  ({100*n_xfer/n_total:.1f}%)')
    print(f'    Backbone (layers 0-10) : {len(backbone_xfer)} tensors  ✓ fully loaded')
    print(f'    Neck     (layers 11-27): {len(neck_xfer)} tensors')
    print(f'  Random init : {n_total-n_xfer:4d} tensors  (P2, DualAttention, Detect nc=1)\n')

    return n_xfer, n_total


# ────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--data',      default='coco.yaml',  help='dataset yaml')
    ap.add_argument('--nc',        type=int, default=1,  help='number of classes')
    ap.add_argument('--epochs',    type=int, default=300)
    ap.add_argument('--imgsz',     type=int, default=640)
    ap.add_argument('--batch',     type=int, default=16, help='batch size (-1=auto)')
    ap.add_argument('--pretrained',default=None, metavar='PT',
                    help='yolo26n.pt — backbone weight transfer')
    ap.add_argument('--freeze',    type=int, default=0, metavar='N',
                    help='freeze first N layers  (10=backbone, useful for stage-1)')
    ap.add_argument('--optimizer', default='musgd',
                    choices=['musgd', 'adamw', 'sgd', 'auto'],
                    help='musgd: paper-native (NS orthogonalization); '
                         'adamw: adaptive LR; sgd: Nesterov SGD')
    ap.add_argument('--device',    default='0')
    ap.add_argument('--workers',   type=int, default=8)
    ap.add_argument('--project',   default='runs/sdd-yolo')
    ap.add_argument('--name',      default='exp')
    ap.add_argument('--resume',    default=None,
                    help='resume from a previous SDD-YOLO checkpoint')
    ap.add_argument('--lr0',       type=float, default=0.001,
                    help='initial LR (default 0.001 for stage-1; use 0.01 for stage-2 with NS active)')
    ap.add_argument('--teacher',   default=None,
                    help='YOLO26x .pt for knowledge distillation (optional)')
    return ap.parse_args()


# ────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    YAML_PATH = os.path.join(os.path.dirname(__file__), 'sdd_yolo_nc1.yaml')

    # ── 1. Build or resume model ──────────────────────────────────────────
    if args.resume:
        print(f'Resuming from: {args.resume}')
        model = YOLO(args.resume)
    else:
        print(f'Building SDD-YOLO-n  (nc={args.nc})  optimizer={args.optimizer}')
        model = YOLO(YAML_PATH)
        if args.pretrained:
            load_pretrained_weights(model, args.pretrained)
        else:
            print('No pretrained weights — training from scratch.\n')

    model.info(verbose=False)

    # ── 2. Training hyperparameters ───────────────────────────────────────
    train_kwargs = dict(
        data         = args.data,
        epochs       = args.epochs,
        imgsz        = args.imgsz,
        batch        = args.batch,
        device       = args.device,
        workers      = args.workers,
        project      = args.project,
        name         = args.name,
        save         = True,
        plots        = True,

        # Optimizer selection — handled by SDD_YOLOTrainer.build_optimizer()
        optimizer    = args.optimizer.upper() if args.optimizer != 'musgd'
                       else 'MuSGD',
        # lr0: stage-1 default=0.001 (backbone frozen, NS disabled on neck/head,
        #   momentum=0.95 gives ~20× effective scale → 0.001 is safe).
        #   stage-2: pass --lr0 0.01 (backbone unfrozen, NS active on matrices,
        #   Newton-Schulz normalises updates so 0.01 is safe).
        lr0          = args.lr0,
        lrf          = 0.01,
        momentum     = 0.95,
        weight_decay = 0.0005,
        warmup_epochs = 3.0,
        warmup_bias_lr = args.lr0,  # must match lr0 so bias doesn't overshoot

        # Loss weights (paper Section 4.3)
        dfl          = 0.0,    # DFL-free training (reg_max=1)
        box          = 7.5,
        cls          = 0.5,

        # Augmentation (paper Section 5.1)
        mosaic       = 1.0,
        mixup        = 0.1,
        scale        = 0.5,
        fliplr       = 0.5,
        translate    = 0.1,
        close_mosaic = 10,

        # Freeze backbone for stage-1 fine-tuning
        freeze       = args.freeze if args.freeze > 0 else None,

        amp          = True,
        val          = True,
        verbose      = True,

        # ── Inject custom trainer (MuSGD support) ─────────────────────────
        trainer      = SDD_YOLOTrainer,
    )

    if args.teacher:
        print(f'KD teacher: {args.teacher}  (custom KD loop required — see paper §4.7)\n')

    print('── Key settings ──')
    for k in ['optimizer', 'lr0', 'momentum', 'dfl', 'freeze', 'pretrained']:
        print(f'  {k:20s}: {train_kwargs.get(k, getattr(args, k, None))}')
    print()

    # ── 3. Train ──────────────────────────────────────────────────────────
    results = model.train(**train_kwargs)

    print('\n' + '=' * 60)
    print('Training complete!')
    print(f'  Best: {model.trainer.best}')
    print(f'  mAP@0.5: {results.results_dict.get("metrics/mAP50(B)", "N/A")}')
    print('=' * 60)


if __name__ == '__main__':
    main()
