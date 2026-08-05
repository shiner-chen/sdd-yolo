# SDD-YOLO: Small-Target Detection for Ground-to-Air Anti-UAV Surveillance

PyTorch/Ultralytics implementation of the architecture described in:

> **SDD-YOLO: A Small-Target Detection Framework for Ground-to-Air Anti-UAV Surveillance with Edge-Efficient Deployment**  
> Pengyu Chen, Haotian Sa, Yiwei Hu, Yuhan Cheng, Junbo Wang  
> arXiv:2603.25218v1, March 2026

---

## What This Repo Contains

| File | Description |
|------|-------------|
| `dual_attention.py` | CBAM-style Dual Attention module (RKNN NPU-friendly, no Softmax) |
| `sdd_yolo_n.yaml` | SDD-YOLO-n architecture config (nc=80, all scales) |
| `sdd_yolo_nc1.yaml` | Single-class Anti-UAV config (nc=1) |
| `build_model.py` | Build + verify model architecture |
| `train_sdd_yolo.py` | Training script with paper hyperparameters |

---

## Architecture: What Changed vs YOLO26n

```
YOLO26n (baseline)               SDD-YOLO-n (this repo)
────────────────────             ────────────────────────────────
Backbone: C3k2 + SPPF + C2PSA   Backbone: same (C2PSA retained)
Neck:     PAN P3/P4/P5           Neck: PAN P2/P3/P4/P5  ← +P2 branch
Heads:    P3(8×) P4(16×) P5(32×) Heads: P2(4×) P3(8×) P4(16×) P5(32×)
Attention: none between neck     Attention: DualAttention at P2/P3/P4
DFL:      reg_max=1              DFL:  reg_max=1, dfl_loss=0.0
NMS:      end2end (O2O)          NMS:  end2end (O2O), same
```

### ① P2 High-Resolution Detection Head

Standard YOLO's smallest feature map is P3 (8× downsampling).
For a 320×320 input, an 8-pixel UAV occupies a single 1×1 feature cell — no spatial detail.

SDD-YOLO adds a P2 branch (4× downsampling):

```
Input 320×320:
  P2 (stride 4):  80×80  = 6 400 anchors  ← NEW  8-px target → 2×2 cells
  P3 (stride 8):  40×40  = 1 600 anchors
  P4 (stride 16): 20×20  =   400 anchors
  P5 (stride 32): 10×10  =   100 anchors
  ────────────────────────────────────────
  Total:                   8 500 anchors  (vs 2 100 without P2)
```

### ② DualAttention (CBAM-style, RKNN NPU-friendly)

Placed between the FPN neck and the detection heads.  
Suppresses aerial clutter (birds, clouds, building edges) specific to G2A UAV detection.

```python
# Channel attention: σ(W_c · GAP(F))
ca = sigmoid(Conv1×1(SiLU(Conv1×1(AdaptiveAvgPool(F)))))

# Spatial attention: σ(Conv_7×7(proj(F)))
# Uses 1×1 Conv instead of ReduceMean/ReduceMax — all ops are RKNN NPU-native
sa = sigmoid(Conv7×7(Conv1×1(F)))

output = F * ca * sa
```

**Why not standard CBAM?**  
`ReduceMean(axis=1)` and `ReduceMax(axis=1)` are NOT supported by RKNN 2.x NPU — they
cause graph splits (+3ms on RK3588 per DualAttention instance). The 1×1 Conv replacement
is fully NPU-native with negligible accuracy loss (~0.1–0.2% mAP).

### ③ DFL-Free + NMS-Free (inherited from YOLO26)

- `reg_max: 1` → DFL degrades to direct regression (no Softmax in bbox head)
- `end2end: True` → One-to-One assignment at inference, no NMS CPU bottleneck
- Training: set `dfl=0.0` to disable DFL loss weight; use WIoU v3 for bbox regression

---

## Requirements

```bash
pip install ultralytics>=8.4.0
# RKNN deployment also requires:
pip install rknn-toolkit2==2.3.2   # host-side quantization
# On RK3588 device:
pip install rknn-toolkit-lite2==2.3.2
```

---

## Quick Start

### 1. Verify the architecture builds

```bash
python build_model.py
# Expected output:
#   Params: ~2.51 M   FLOPs: ~6.4 G  (at 320×320, nc=1)
#   3 DualAttention modules at P2/P3/P4, Zero Softmax ✅
```

### 2. Train (single-class Anti-UAV, nc=1)

```bash
python train_sdd_yolo.py \
    --data your_dataset.yaml \
    --nc 1 \
    --epochs 300 \
    --imgsz 640 \
    --batch 16
```

Key training settings (paper Section 4.6):

| Setting | Value | Reason |
|---------|-------|--------|
| `dfl=0.0` | 0.0 | DFL-free (reg_max=1) |
| `optimizer` | AdamW (proxy for MuSGD) | Gradient stability |
| `mosaic` | 1.0 | Scale-diverse augmentation |
| `end2end` | True | NMS-free O2O inference |

For full **MuSGD** (Newton-Schulz gradient orthogonalization) and **STAL**
(Small-Target-Aware Label Assignment), implement a custom optimizer following
the paper's Section 4.6. The training script uses AdamW as a practical proxy.

### 3. Export to RKNN INT8 for RK3588

```python
import sys
sys.path.insert(0, '.')
from dual_attention import DualAttention
import ultralytics.nn.modules as _m, ultralytics.nn.tasks as _t
setattr(_m, 'DualAttention', DualAttention)
_t.__dict__['DualAttention'] = DualAttention

from ultralytics import YOLO
model = YOLO('sdd_yolo_nc1.yaml')  # or your trained .pt
model.model.model[-1].end2end = False  # disable end2end for RKNN

model.export(format='onnx', imgsz=320, simplify=True, opset=12)
```

Then use `rknn-toolkit2` to quantize the resulting ONNX to INT8.

---

## RKNN RK3588 Performance (bench_rknn.py, NPU Core 0)

| Model | Input | CPU ops | Mean | Std | FPS |
|-------|-------|---------|------|-----|-----|
| YOLO26n nc=1 (no PSA) | 320×320 | 0 | 5.41ms | 0.12ms | 185 |
| **SDD-YOLO-n nc=1** | 320×320 | 2 (C2PSA) | **10.63ms** | **0.12ms** | **94** |
| et-yolov6n (reference) | 320×320 | 0 | 5.63ms | 0.12ms | 178 |

SDD-YOLO is slower than a plain YOLO26n because:

1. **P2 head** adds 6 400 extra anchors (8 500 total vs 2 100) — more NPU compute
2. **C2PSA × 2** causes 2 NPU↔CPU graph splits (~+1.7ms) — Softmax not supported on RK3588 NPU

To further reduce latency: replace C2PSA with C3k2 in the YAML (~−1.7ms, ~−1–2% mAP).

---

## File Structure

```
sdd_yolo/
├── dual_attention.py      # DualAttention module (RKNN-friendly CBAM variant)
├── sdd_yolo_n.yaml        # nc=80 architecture (all model scales n/s/m/l/x)
├── sdd_yolo_nc1.yaml      # nc=1 single-class Anti-UAV variant
├── build_model.py         # Architecture verification script
└── train_sdd_yolo.py      # Training script
```

---

## Citation

```bibtex
@article{chen2026sddyolo,
  title   = {SDD-YOLO: A Small-Target Detection Framework for
             Ground-to-Air Anti-UAV Surveillance with Edge-Efficient Deployment},
  author  = {Chen, Pengyu and Sa, Haotian and Hu, Yiwei and
             Cheng, Yuhan and Wang, Junbo},
  journal = {arXiv:2603.25218},
  year    = {2026}
}
```
