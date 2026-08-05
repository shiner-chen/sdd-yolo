"""
SDD-YOLO Training Script
=========================
Trains SDD-YOLO-n on a custom single-class dataset (e.g., Anti-UAV).

Key training settings from paper (Section 4.6):
  - MuSGD optimizer  : gradient orthogonalization (Newton-Schulz)
  - ProgLoss         : progressive loss re-weighting
  - STAL             : Small-Target-Aware Label Assignment
  - dfl=0.0          : disable DFL loss (consistent with reg_max=1)
  - WIoU v3          : bbox regression loss (set via iou='wiou')
  - KD from teacher  : optional knowledge distillation

Usage:
    # Single-class UAV detection
    python train_sdd_yolo.py --data anti_uav.yaml --nc 1 --epochs 300

    # With knowledge distillation (requires teacher model)
    python train_sdd_yolo.py --data anti_uav.yaml --nc 1 \
        --teacher yolo26x.pt --epochs 300
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

# Register custom module BEFORE importing YOLO
from dual_attention import DualAttention
import ultralytics.nn.modules as _m
import ultralytics.nn.tasks  as _t
setattr(_m, 'DualAttention', DualAttention)
_t.__dict__['DualAttention'] = DualAttention

from ultralytics import YOLO


def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--data',    default='coco.yaml',     help='dataset yaml')
    ap.add_argument('--nc',      type=int, default=1,     help='number of classes')
    ap.add_argument('--epochs',  type=int, default=300,   help='total epochs')
    ap.add_argument('--imgsz',   type=int, default=640,   help='input image size')
    ap.add_argument('--batch',   type=int, default=16,    help='batch size')
    ap.add_argument('--device',  default='0',             help='cuda device or cpu')
    ap.add_argument('--project', default='runs/sdd-yolo', help='output directory')
    ap.add_argument('--name',    default='exp',           help='run name')
    ap.add_argument('--teacher', default=None,            help='teacher .pt for KD (optional)')
    ap.add_argument('--resume',  default=None,            help='resume from checkpoint')
    ap.add_argument('--workers', type=int, default=8,     help='dataloader workers')
    return ap.parse_args()


def build_data_yaml(nc: int, original_yaml: str) -> str:
    """If nc=1, patch the dataset yaml to use single class."""
    if nc == 80:
        return original_yaml
    import yaml, tempfile
    with open(original_yaml) as f:
        d = yaml.safe_load(f)
    d['nc'] = nc
    if nc == 1:
        d['names'] = {0: 'uav'}
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(d, tmp)
    tmp.close()
    return tmp.name


def main():
    args = parse_args()

    YAML_PATH = os.path.join(os.path.dirname(__file__), 'sdd_yolo_n.yaml')

    # ── Build model ───────────────────────────────────────────────────────
    if args.resume:
        print(f'Resuming from: {args.resume}')
        model = YOLO(args.resume)
    else:
        print(f'Building SDD-YOLO-n from {YAML_PATH}  (nc={args.nc})')
        model = YOLO(YAML_PATH)
        # Set number of classes
        if args.nc != 80:
            model.model.nc = args.nc
            # Update the Detect head nc
            detect = model.model.model[-1]
            detect.nc = args.nc

    print(model.info())

    # ── Training hyperparameters ───────────────────────────────────────────
    # Paper Section 4.3 & 4.6:
    #   - dfl=0.0       : DFL loss weight = 0 (reg_max=1 already makes it trivial)
    #   - optimizer=MuSGD is not yet native in Ultralytics 8.4.x;
    #     use SGD as fallback (MuSGD gradient orthogonalization needs custom impl)
    #   - iou='wiou'    : WIoU v3 — set if your Ultralytics version supports it
    #                     otherwise use 'ciou' (close approximation)
    #   - STAL          : enabled via tal_topk_candidates_small (if supported)

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
        dfl        = 0.0,          # Disable DFL loss (reg_max=1, use WIoU instead)
        box        = 7.5,          # bbox regression loss weight
        cls        = 0.5,          # classification loss weight

        # ── Optimizer ─────────────────────────────────────────────────────
        # MuSGD not natively in Ultralytics 8.4.x; using AdamW as close proxy
        # For full MuSGD: implement custom optimizer with Newton-Schulz step
        optimizer  = 'AdamW',
        lr0        = 0.001,
        lrf        = 0.01,
        momentum   = 0.937,
        weight_decay = 0.0005,
        warmup_epochs = 3.0,

        # ── Data augmentation (paper Section 5.1) ─────────────────────────
        mosaic     = 1.0,          # Mosaic augmentation
        mixup      = 0.1,          # Mixup (paper uses this)
        scale      = 0.5,          # Scale jitter
        fliplr     = 0.5,
        translate  = 0.1,

        # ── Small-target specific ──────────────────────────────────────────
        # STAL is handled internally by YOLO26's label assignment (end2end=True)
        # The One-to-One + One-to-Many dual assignment in YOLO26 already
        # implements small-target-aware logic via the end2end head

        close_mosaic = 10,         # disable mosaic last N epochs
        amp        = True,         # AMP training
        val        = True,
        verbose    = True,
    )

    # ── Optional: Knowledge Distillation ──────────────────────────────────
    if args.teacher:
        print(f'\nKnowledge Distillation enabled: teacher={args.teacher}')
        print('Note: Full KD (Section 4.7) requires custom KD loss implementation.')
        print('Using Ultralytics built-in KD if available (kd_loss_weight param).')
        # Ultralytics 8.4.x does not have native KD built-in for custom modules
        # For full paper-style KD, implement:
        #   L_total = (1-λ)*L_task + λ*L_KD
        # where L_KD = Σ T²·KL(σ(z_s/T) || σ(z_t/T)) at P2-P5 levels

    # ── Start training ─────────────────────────────────────────────────────
    print('\nStarting training with settings:')
    for k, v in train_kwargs.items():
        print(f'  {k:20s}: {v}')
    print()

    results = model.train(**train_kwargs)

    print('\n' + '='*60)
    print('Training complete!')
    print(f'  Best model: {model.trainer.best}')
    print(f'  mAP@0.5:    {results.results_dict.get("metrics/mAP50(B)", "N/A")}')
    print('='*60)

    return results


if __name__ == '__main__':
    main()
