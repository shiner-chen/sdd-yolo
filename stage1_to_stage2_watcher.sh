#!/usr/bin/env bash
# Watches results.csv for epoch 15, then kills Stage 1 and launches Stage 2.
set -euo pipefail

RESULTS_CSV="/home/adlink/chenx/sdd-yolo/runs/detect/runs/sdd-yolo/stage1_freeze/results.csv"
STAGE1_PID=906178
LOGDIR="/home/adlink/chenx/sdd-yolo"
PYTHON="/home/adlink/chenx/rknn-env/bin/python"

echo "[watcher] Started at $(date). Waiting for epoch 15 in results.csv..."

while true; do
    if [ -f "$RESULTS_CSV" ]; then
        # Check if epoch 15 row exists (first field == 15)
        if awk -F',' 'NR>1 && $1+0==15 {found=1} END {exit !found}' "$RESULTS_CSV" 2>/dev/null; then
            echo "[watcher] Epoch 15 confirmed in results.csv at $(date). Stopping Stage 1..."
            break
        fi
    fi
    sleep 30
done

# ── Kill Stage 1 process group ────────────────────────────────────────────────
PGID=$(ps -o pgid= -p "$STAGE1_PID" 2>/dev/null | tr -d ' ') || true
if [ -n "$PGID" ] && [ "$PGID" != "0" ]; then
    echo "[watcher] Sending SIGTERM to process group $PGID ..."
    kill -TERM -- "-$PGID" 2>/dev/null || true
    sleep 5
    # Force kill if still alive
    kill -KILL -- "-$PGID" 2>/dev/null || true
    echo "[watcher] Stage 1 processes terminated."
else
    echo "[watcher] Stage 1 PID $STAGE1_PID not found — may have already exited."
fi

# Wait for GPU memory to free
sleep 10

# ── Launch Stage 2 ────────────────────────────────────────────────────────────
WEIGHTS="$LOGDIR/runs/detect/runs/sdd-yolo/stage1_freeze/weights/last.pt"

if [ ! -f "$WEIGHTS" ]; then
    echo "[watcher] ERROR: Stage 1 weights not found at $WEIGHTS"
    exit 1
fi

echo "[watcher] Launching Stage 2 at $(date)"
echo "[watcher] Weights: $WEIGHTS"

cd "$LOGDIR"
PYTHONPATH="$LOGDIR" nohup "$PYTHON" train_sdd_yolo.py \
    --data /home/adlink/data/ARD100_roi320/ard100_roi320.yaml \
    --nc 1 \
    --resume "$WEIGHTS" \
    --optimizer musgd \
    --lr0 0.01 \
    --epochs 200 \
    --imgsz 320 \
    --batch 128 \
    --device 0,1 \
    --workers 16 \
    --project runs/sdd-yolo \
    --name stage2_finetune \
    > "$LOGDIR/stage2.log" 2>&1 &

STAGE2_PID=$!
echo "[watcher] Stage 2 launched, PID=$STAGE2_PID"
echo "[watcher] Log: $LOGDIR/stage2.log"
echo "[watcher] Done."
