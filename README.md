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
| `musgd.py` | MuSGD optimizer — Newton-Schulz gradient orthogonalization |
| `sdd_yolo_n.yaml` | SDD-YOLO-n architecture config (nc=80, all scales) |
| `sdd_yolo_nc1.yaml` | Single-class Anti-UAV config (nc=1) |
| `build_model.py` | Build + verify model architecture |
| `train_sdd_yolo.py` | Training script (MuSGD + pretrained weight transfer) |

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
Optimizer: —                     Optimizer: MuSGD (NS orthogonalization)
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

## Training

### Optimizer Choice: MuSGD vs AdamW

This repo implements the paper-native **MuSGD** optimizer alongside AdamW.
Choose based on your dataset and compute budget:

| | MuSGD (paper-native, default) | AdamW | SGD |
|--|-------------------------------|-------|-----|
| **Core mechanism** | Nesterov SGD + Newton-Schulz gradient orthogonalization | Adaptive per-param LR (Adam) + decoupled weight decay | Nesterov momentum |
| **Sparse G2A UAV data** | ✅ Best — prevents rank collapse of backbone filters under sparse supervision | ⚠️ Good, but risk of weight matrix rank collapse | ⚠️ Needs careful LR tuning |
| **Large dataset (COCO-scale)** | ✓ Competitive | ✓ Often faster convergence | ✓ Good with warmup |
| **Fine-tuning from YOLO26n** | ✓ Slightly better | ✓ Very usable (gap narrows) | ⚠️ May need lower LR |
| **Estimated mAP advantage** | Baseline | ~−0.5–1.5% on sparse G2A data | Similar to AdamW |
| **Convergence speed** | Similar | Slightly faster early epochs | Slower |

**Why MuSGD for sparse UAV detection:**  
G2A datasets have extreme foreground–background imbalance (~0.01% foreground pixels).
Sparse gradient signals cause backbone weight matrices to undergo **rank collapse** — multiple
filters learn identical features, losing the fine-grained texture diversity needed for
sub-16px targets. MuSGD's Newton-Schulz step rotates each gradient to an orthonormal
direction, keeping backbone filters maximally diverse:

```
Standard SGD:  W ← W − η · G          (direction: raw gradient)
MuSGD:         G' = NS(G)             (equalize singular values: std 2.99 → 0.15)
               W ← W − η · G'·(‖G‖/‖G'‖)  (direction: orthogonalized; norm: preserved)
```

MuSGD applies NS only to **2-D backbone weight matrices** (layers 0–10);
1-D params (bias, BN) and neck/head params use plain Nesterov SGD.

---

### Recommended: Two-Stage Training with Pretrained Weights

Using YOLO26n pretrained weights dramatically speeds up convergence.
The backbone (layers 0–10, Conv + C3k2 + SPPF + C2PSA) transfers 100%;
new P2 branch, DualAttention, and Detect(nc=1) are randomly initialised.

```
Weight transfer coverage:
  ✓  Backbone (layers 0-10): ~76% of total tensors — fully loaded
  ✓  Neck P3/P4/P5 C3k2 blocks — partially loaded
  ✗  P2 neck branch           — random init
  ✗  DualAttention × 3        — random init
  ✗  Detect head (nc=1)       — random init (nc mismatch with YOLO26n nc=80)
```

**Stage 1 — Train new layers, backbone frozen (fast convergence)**

```bash
python train_sdd_yolo.py \
    --data your_uav_data.yaml \
    --nc 1 \
    --pretrained yolo26n.pt \
    --optimizer musgd \
    --freeze 10 \
    --epochs 100 \
    --imgsz 640 \
    --batch 16
```

What happens:
- Backbone weights loaded from `yolo26n.pt`, frozen (`--freeze 10`)
- P2 branch, DualAttention, Detect head trained from scratch with full LR
- MuSGD applied to all unfrozen parameters
- Converges in ~50–80 epochs; backbone already provides strong features

**Stage 2 — Fine-tune all layers**

```bash
python train_sdd_yolo.py \
    --data your_uav_data.yaml \
    --nc 1 \
    --resume runs/sdd-yolo/exp/weights/last.pt \
    --optimizer musgd \
    --epochs 200
```

What happens:
- All layers unfrozen; backbone fine-tuned at the same LR
- MuSGD NS orthogonalization most impactful here — backbone filters stay diverse
  even under sparse UAV gradient signals
- Typical total training: ~150–250 epochs to convergence

**Train from scratch (no pretrained weights)**

```bash
python train_sdd_yolo.py \
    --data your_uav_data.yaml \
    --nc 1 \
    --optimizer musgd \
    --epochs 300
```

Use `--optimizer adamw` if MuSGD is too slow or dataset is large enough
that rank collapse is not a concern.

---

### Key Hyperparameters (paper Section 4.6)

| Setting | Value | Source | Reason |
|---------|-------|--------|--------|
| `dfl` | `0.0` | paper §4.3 | DFL-free: reg_max=1 makes DFL trivial; use WIoU v3 instead |
| `optimizer` | `musgd` | paper §4.6 | Newton-Schulz orthogonalization for sparse G2A data |
| `lr0` | `0.01` | paper | Backbone matrix LR; vector/neck LR auto-scaled |
| `momentum` | `0.95` | paper | Nesterov momentum coefficient |
| `mosaic` | `1.0` | paper §5.1 | Scale-diverse augmentation for small targets |
| `mixup` | `0.1` | paper §5.1 | Foreground mixing |
| `freeze` | `10` (stage 1) | this repo | Freeze backbone layers 0–10 in stage 1 |
| `end2end` | `True` (train) | YOLO26 | O2O label assignment (STAL-like behaviour) |
| `end2end` | `False` (RKNN export) | this repo | Disable TopK/Gather for NPU deployment |

---

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

Then quantize to RKNN INT8 using `rknn-toolkit2`.

---

## RKNN RK3588 Performance (bench_rknn.py, NPU Core 0)

| Model | Input | CPU ops | Mean | Std | FPS |
|-------|-------|---------|------|-----|-----|
| YOLO26n nc=1 (no PSA) | 320×320 | 0 | 5.41ms | 0.12ms | 185 |
| **SDD-YOLO-n nc=1** | 320×320 | 2 (C2PSA) | **10.63ms** | **0.12ms** | **94** |
| et-yolov6n (reference) | 320×320 | 0 | 5.63ms | 0.12ms | 178 |

SDD-YOLO is slower than plain YOLO26n because:

1. **P2 head** adds 6 400 extra anchors (8 500 total vs 2 100) — more NPU compute
2. **C2PSA × 2** causes 2 NPU↔CPU graph splits (~+1.7ms) — Softmax not supported on RK3588 NPU

To further reduce latency: replace C2PSA with C3k2 in the YAML (~−1.7ms, ~−1–2% mAP).

---

## Quantization-Aware Training (QAT)

Standard INT8 PTQ compresses all high-confidence scores into a single quantization
bucket, producing discrete outputs (all detections get the same score ≈ 1.29).
QAT fixes this by training with fake quantization so the model learns to maintain
continuous, discriminative confidence scores even after INT8 conversion.

### How it works

```
best.pt (FP32)
  │
  ├─ insert FakeQuantize hooks (simulates RKNN INT8 rounding during forward pass)
  │    • Activation: per-tensor symmetric INT8  (matches RKNN default)
  │    • Weights:    per-channel symmetric INT8 (matches RKNN channel-wise quant)
  │    • Skipped:    C2PSA (Softmax, CPU-only on RKNN — no benefit from QAT)
  │
  ├─ Epoch 1-2  (calibration): observer ON, fake_quant OFF
  │    → measure per-layer activation ranges without disturbing training
  │
  ├─ Epoch 3-20 (QAT active):  observer ON, fake_quant ON
  │    → model trains with INT8 rounding noise → learns to spread confidence scores
  │
  └─ Export as FP32 ONNX  →  RKNN-Toolkit2 INT8 PTQ
       QAT-regularized activations make RKNN's subsequent PTQ much more accurate
```

Expected improvement vs plain INT8 PTQ:

| Metric | INT8 PTQ | After QAT + PTQ |
|--------|---------|----------------|
| Score distribution | 1 discrete value (1.29) | continuous [0, 1] |
| mAP@0.5 | ~17% | ~35–45% (estimated) |
| Latency | unchanged | unchanged (~13ms) |

### Code structure (`qat_sdd_yolo.py`)

```python
make_act_fq()              # per-tensor symmetric INT8 FakeQuantize (activations)
make_wt_fq()               # per-channel symmetric INT8 FakeQuantize (weights)

QATHooks                   # manages all FakeQuantize forward hooks
  .attach(model)           # insert hooks on all eligible Conv layers
  .set_observer_enabled()  # control calibration phase
  .set_fake_quant_enabled()# control quantization phase

QATDetectionTrainer        # extends Ultralytics DetectionTrainer
  ._setup_train()          # auto-injects QATHooks after model→device
  .optimizer_step()        # enables fake_quant after calibration epochs

export_qat_onnx(pt, imgsz) # QAT .pt → FP32 ONNX (end2end=False for RKNN)
quantize_to_rknn(onnx, txt)# ONNX → RKNN INT8 via rknn-toolkit2 PTQ

main() / argparse          # full 3-stage pipeline with CLI
```

### Usage

**Full pipeline — train + export + RKNN INT8:**

```bash
python qat_sdd_yolo.py \
    --pretrained /path/to/best.pt \
    --data       your_uav_dataset.yaml \
    --epochs     20 \
    --lr         0.0001 \
    --calib-data /path/to/calib500_fixed.txt \
    --device     0
```

**Export only (skip training):**

```bash
python qat_sdd_yolo.py \
    --pretrained runs/qat/exp/weights/best_qat.pt \
    --export-only \
    --calib-data /path/to/calib500_fixed.txt
```

**Key training notes:**

| Setting | Value | Reason |
|---------|-------|--------|
| `--lr` | `1e-4` | ~1/100 of original LR — fine-tune, not re-train |
| `--epochs` | `20` | 2 calibration + 18 QAT-active epochs |
| `amp` | disabled | FakeQuantize STE gradient unstable in fp16 |
| `mosaic` | `0.5` | reduced — avoid disturbing fine-tune convergence |
| `dfl` | `0.0` | inherited from SDD-YOLO (reg_max=1) |
| `qat_calibration_epochs` | `2` | observer-only phase before fake_quant enabled |

---

## File Structure

```
sdd_yolo/
├── dual_attention.py      # DualAttention module (RKNN-friendly CBAM variant)
├── musgd.py               # MuSGD optimizer (Newton-Schulz gradient orthogonalization)
├── qat_sdd_yolo.py        # QAT fine-tuning script (INT8 score discretization fix)
├── sdd_yolo_n.yaml        # nc=80 architecture (all model scales n/s/m/l/x)
├── sdd_yolo_nc1.yaml      # nc=1 single-class Anti-UAV variant
├── build_model.py         # Architecture verification script
└── train_sdd_yolo.py      # Training script (MuSGD + pretrained transfer + 2-stage)
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
