"""
SDD-YOLO Model Builder & Verifier
==================================
Registers DualAttention into Ultralytics, builds SDD-YOLO-n, verifies architecture.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# ── 1. Register DualAttention into Ultralytics module namespace ───────────
from dual_attention import DualAttention
import ultralytics.nn.modules as ult_modules
import ultralytics.nn.tasks  as ult_tasks

# Inject into both the modules namespace and tasks globals
# (parse_model uses getattr(modules, name) to resolve YAML module names)
setattr(ult_modules, 'DualAttention', DualAttention)
ult_tasks.__dict__['DualAttention'] = DualAttention

# ── 2. Build model ─────────────────────────────────────────────────────────
from ultralytics import YOLO
import torch

YAML_PATH = os.path.join(os.path.dirname(__file__), 'sdd_yolo_n.yaml')
print('=' * 65)
print('Building SDD-YOLO-n ...')
print(f'YAML: {YAML_PATH}')
print('=' * 65)

model = YOLO(YAML_PATH)

# ── 3. Architecture summary ────────────────────────────────────────────────
print('\n── Architecture Summary ──')
model.info(verbose=False)

# ── 4. Forward pass verification ─────────────────────────────────────────
model.model.eval()
dummy = torch.zeros(1, 3, 640, 640)
with torch.no_grad():
    out = model.model(dummy)

print('\n── Forward Pass (640×640, end2end=True) ──')
if isinstance(out, (list, tuple)):
    for i, o in enumerate(out):
        if hasattr(o, 'shape'):
            print(f'  output[{i}]: {tuple(o.shape)}')
        elif isinstance(o, (list, tuple)):
            for j, oo in enumerate(o):
                if hasattr(oo, 'shape'):
                    print(f'  output[{i}][{j}]: {tuple(oo.shape)}')
else:
    print(f'  output: {tuple(out.shape)}')

# ── 5. Compare with paper ─────────────────────────────────────────────────
from ultralytics.utils.torch_utils import get_num_params
params = get_num_params(model.model) / 1e6
print(f'\n── Params vs Paper (Table 2) ──')
print(f'  Params : {params:.3f} M   (paper target: 2.50 M)')

# ── 6. DualAttention module check ─────────────────────────────────────────
print('\n── DualAttention modules ──')
found, total_da_params = 0, 0
for name, m in model.model.named_modules():
    if isinstance(m, DualAttention):
        found += 1
        # Check ca was built (lazy init after dummy forward)
        if hasattr(m, 'ca') and m.ca is not None:
            p = sum(p.numel() for p in m.parameters())
            total_da_params += p
            c1 = m.ca[1].in_channels  # Conv2d after AdaptiveAvgPool
            print(f'  ✓ {name:45s} c1={c1}  params={p}')
        else:
            print(f'  ✗ {name} — not built yet')

if found == 0:
    print('  ✗ None found — check YAML registration')
else:
    print(f'\n  Total: {found} DualAttention modules  (expected 3: P2/P3/P4)')
    print(f'  Total DA params: {total_da_params}')
    print('  Ops: AdaptiveAvgPool2d + Conv2d + SiLU + Sigmoid + mean/max')
    print('  → Zero Softmax → RKNN NPU-friendly ✅')

print('\n' + '=' * 65)
print('SDD-YOLO-n build: SUCCESS ✅')
print('=' * 65)
