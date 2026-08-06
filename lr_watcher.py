#!/usr/bin/env python3
"""
Stage 2 LR watcher.
Trigger: epoch >= 10, last 3 mAP50 strictly declining,
         latest mAP50 < best_mAP50 - DECLINE_THRESHOLD
Action: kill Stage 2, restart from best.pt with lr0=0.005
"""
import csv, os, signal, subprocess, sys, time

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_CSV  = "/home/adlink/chenx/sdd-yolo/runs/detect/runs/sdd-yolo/stage2_finetune/results.csv"
STAGE2_PID   = 911754
LOGDIR       = "/home/adlink/chenx/sdd-yolo"
PYTHON       = "/home/adlink/chenx/rknn-env/bin/python"

TRIGGER_EPOCH_MIN   = 10      # don't act before this epoch
DECLINE_CONSECUTIVE = 3       # need N strictly declining epochs
DECLINE_THRESHOLD   = 0.045   # drop below (best - this) to trigger
NEW_LR0             = 0.005
POLL_INTERVAL       = 90      # seconds between checks

def log(msg):
    print(msg, flush=True)

log(f"[lr-watcher] Started at PID={os.getpid()}. Monitoring Stage 2...")

triggered = False
while not triggered:
    time.sleep(POLL_INTERVAL)

    try:
        with open(RESULTS_CSV) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue

        data = [(int(float(r["epoch"])), float(r["metrics/mAP50(B)"]))
                for r in rows]
        best_map   = max(m for _, m in data)
        last_epoch, latest_map = data[-1]

        log(f"[lr-watcher] ep{last_epoch:>3d}  mAP50={latest_map:.4f}  "
            f"best={best_map:.4f}  trigger@ep>={TRIGGER_EPOCH_MIN} + "
            f"{DECLINE_CONSECUTIVE}↓ + drop>{DECLINE_THRESHOLD:.3f}")

        if last_epoch < TRIGGER_EPOCH_MIN:
            continue
        if len(data) < DECLINE_CONSECUTIVE:
            continue

        last_n = [m for _, m in data[-DECLINE_CONSECUTIVE:]]
        is_declining = all(last_n[i] < last_n[i-1]
                          for i in range(1, DECLINE_CONSECUTIVE))
        is_below_best = latest_map < (best_map - DECLINE_THRESHOLD)

        if is_declining and is_below_best:
            log(f"[lr-watcher] *** TRIGGER ***  "
                f"last {DECLINE_CONSECUTIVE} epochs declining "
                f"({', '.join(f'{v:.4f}' for v in last_n)}), "
                f"mAP50={latest_map:.4f} < best({best_map:.4f})-{DECLINE_THRESHOLD}")
            triggered = True

    except Exception as e:
        log(f"[lr-watcher] Warning reading CSV: {e}")

# ── Kill Stage 2 ───────────────────────────────────────────────────────────────
log(f"[lr-watcher] Stopping Stage 2 (PID {STAGE2_PID})...")
try:
    pgid = os.getpgid(STAGE2_PID)
    os.killpg(pgid, signal.SIGTERM)
    log(f"[lr-watcher] SIGTERM → process group {pgid}")
    time.sleep(6)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    log("[lr-watcher] Stage 2 terminated.")
except ProcessLookupError:
    log("[lr-watcher] Stage 2 already exited.")
except Exception as e:
    log(f"[lr-watcher] Kill error: {e}")

time.sleep(12)   # let GPU memory free

# ── Pick weights ───────────────────────────────────────────────────────────────
best_pt = f"{LOGDIR}/runs/detect/runs/sdd-yolo/stage2_finetune/weights/best.pt"
last_pt = f"{LOGDIR}/runs/detect/runs/sdd-yolo/stage2_finetune/weights/last.pt"
weights = best_pt if os.path.exists(best_pt) else last_pt
log(f"[lr-watcher] Using weights: {weights}")

# ── Restart Stage 2 ────────────────────────────────────────────────────────────
restart_log = f"{LOGDIR}/stage2_restart.log"
log(f"[lr-watcher] Launching Stage 2 restart  lr0={NEW_LR0}  log={restart_log}")

env = os.environ.copy()
env["PYTHONPATH"] = LOGDIR
os.chdir(LOGDIR)

with open(restart_log, "w") as flog:
    proc = subprocess.Popen(
        [PYTHON, "train_sdd_yolo.py",
         "--data",      "/home/adlink/data/ARD100_roi320/ard100_roi320.yaml",
         "--nc",        "1",
         "--resume",    weights,
         "--optimizer", "musgd",
         "--lr0",       str(NEW_LR0),
         "--epochs",    "200",
         "--imgsz",     "320",
         "--batch",     "128",
         "--device",    "0,1",
         "--workers",   "16",
         "--project",   "runs/sdd-yolo",
         "--name",      "stage2_lr005"],
        stdout=flog, stderr=flog, env=env,
        preexec_fn=os.setsid,
    )

log(f"[lr-watcher] Stage 2 restarted  PID={proc.pid}  log=stage2_restart.log")
log("[lr-watcher] Done. Exiting watcher.")
