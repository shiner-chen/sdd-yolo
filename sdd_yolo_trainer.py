"""
SDD-YOLO custom trainer module.

Lives in its own file so that Ultralytics DDP can import it by module name
in worker processes.  When the trainer class is defined in __main__ (the
launch script), Ultralytics generates:
    from __main__ import SDD_YOLOTrainer
in the DDP temp script, which fails because __main__ in the worker context
is the DDP temp script itself, not train_sdd_yolo.py.

By defining SDD_YOLOTrainer here, SDD_YOLOTrainer.__module__ == 'sdd_yolo_trainer',
and Ultralytics instead generates:
    from sdd_yolo_trainer import SDD_YOLOTrainer
which succeeds as long as this directory is on PYTHONPATH.
"""

import sys
import os

# Ensure the sdd-yolo directory is on sys.path so dual_attention / musgd are
# importable even when this module is loaded by a DDP worker subprocess.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── Register DualAttention BEFORE any Ultralytics import that parses YAML ──
from dual_attention import DualAttention
import ultralytics.nn.modules as _m
import ultralytics.nn.tasks as _t
setattr(_m, 'DualAttention', DualAttention)
_t.__dict__['DualAttention'] = DualAttention

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER
from musgd import MuSGD, build_musgd_param_groups


class SDD_YOLOTrainer(DetectionTrainer):
    """Custom Ultralytics trainer: MuSGD optimizer + ProgLoss scheduling.

    MuSGD: Newton-Schulz gradient orthogonalization for backbone weight
    matrices; prevents rank collapse under sparse G2A UAV supervision.

    ProgLoss (paper §4.6): dynamically re-weights box/cls loss components
    across epochs.  Early epochs emphasise classification (help the model
    find micro-targets before refining their boxes); later epochs shift
    weight to box regression for precise localisation.

    Schedule over the first PROG_RATIO fraction of total epochs:
      cls : PROG_CLS_INIT → PROG_CLS_FINAL  (ramps DOWN)
      box : PROG_BOX_INIT → PROG_BOX_FINAL  (ramps UP)
    Both are held at their final values for the remainder of training.
    """

    # ── ProgLoss hyperparameters (paper §4.6) ─────────────────────────────
    PROG_BOX_INIT  = 3.75   # 0.5× final — start with lower box weight
    PROG_BOX_FINAL = 7.5    # paper value
    PROG_CLS_INIT  = 1.5    # 3× final  — start with higher cls weight
    PROG_CLS_FINAL = 0.5    # paper value
    PROG_RATIO     = 0.5    # ramp completes at this fraction of total epochs

    # ── ProgLoss callback ─────────────────────────────────────────────────
    @staticmethod
    def _prog_loss_cb(trainer: "SDD_YOLOTrainer") -> None:
        """Linearly ramp box/cls loss weights; called at on_train_epoch_start."""
        prog_end = max(1, int(trainer.epochs * SDD_YOLOTrainer.PROG_RATIO))
        epoch    = trainer.epoch

        if epoch < prog_end:
            t = epoch / prog_end  # 0.0 → 1.0
            trainer.args.box = (SDD_YOLOTrainer.PROG_BOX_INIT
                                + t * (SDD_YOLOTrainer.PROG_BOX_FINAL
                                       - SDD_YOLOTrainer.PROG_BOX_INIT))
            trainer.args.cls = (SDD_YOLOTrainer.PROG_CLS_INIT
                                + t * (SDD_YOLOTrainer.PROG_CLS_FINAL
                                       - SDD_YOLOTrainer.PROG_CLS_INIT))
        else:
            trainer.args.box = SDD_YOLOTrainer.PROG_BOX_FINAL
            trainer.args.cls = SDD_YOLOTrainer.PROG_CLS_FINAL

        # Log every 10 epochs (and always at epoch 0)
        if epoch == 0 or (epoch + 1) % 10 == 0:
            LOGGER.info(
                f"  [ProgLoss] epoch {epoch + 1:>4d}/{trainer.epochs}"
                f"  box={trainer.args.box:.3f}  cls={trainer.args.cls:.3f}"
            )

    def build_optimizer(self, model, name="auto", lr=0.01, momentum=0.95,
                        decay=0.0005, iterations=None):
        """Build optimizer and register ProgLoss callback (idempotent)."""

        # ── Register ProgLoss once (guard against double-registration on resume)
        existing = [getattr(c, "__name__", "") for c in
                    self.callbacks.get("on_train_epoch_start", [])]
        if "_prog_loss_cb" not in existing:
            self.add_callback("on_train_epoch_start",
                              SDD_YOLOTrainer._prog_loss_cb)
            LOGGER.info(
                f"\n── ProgLoss registered ──\n"
                f"  cls : {SDD_YOLOTrainer.PROG_CLS_INIT:.2f} → "
                f"{SDD_YOLOTrainer.PROG_CLS_FINAL:.2f}  (ramps down over "
                f"first {int(SDD_YOLOTrainer.PROG_RATIO*100)}% of epochs)\n"
                f"  box : {SDD_YOLOTrainer.PROG_BOX_INIT:.2f} → "
                f"{SDD_YOLOTrainer.PROG_BOX_FINAL:.2f}  (ramps up)\n"
            )

        opt_name = (name or "auto").lower()

        if opt_name != "musgd":
            # Fall back to Ultralytics built-in (AdamW / SGD / auto)
            return super().build_optimizer(
                model, name=name, lr=lr, momentum=momentum,
                decay=decay, iterations=iterations,
            )

        # ── MuSGD construction ────────────────────────────────────────────
        LOGGER.info(
            "\n── Building MuSGD optimizer ──\n"
            "  backbone matrices (0-10): Newton-Schulz orthogonalized SGD\n"
            "  backbone vectors (bias/BN): Nesterov SGD\n"
            "  neck + head: Nesterov SGD\n"
        )

        param_groups = build_musgd_param_groups(
            model,
            backbone_layers=10,
            matrix_lr=lr,           # backbone matrix LR  (from hyp)
            vector_lr=lr * 0.1,     # backbone bias/BN    (10× smaller)
            neck_head_lr=lr,        # neck + P2 + head    (same as backbone)
            weight_decay=decay,
            momentum=momentum,
            ns_steps=5,
        )

        # Report param counts per group
        for pg in param_groups:
            n = sum(p.numel() for p in pg["params"])
            LOGGER.info(
                f"  {pg['label']:20s}  {len(pg['params']):4d} tensors"
                f"  {n/1e6:.2f}M params  lr={pg['lr']:.2e}"
                f"  NS={'ON ' if pg['use_ns'] else 'OFF'}"
            )

        optimizer = MuSGD(param_groups, lr=lr, momentum=momentum,
                          weight_decay=decay)

        LOGGER.info(
            f"\n  Total optimisable params: "
            f"{sum(p.numel() for g in param_groups for p in g['params'])/1e6:.2f} M\n"
        )
        return optimizer
