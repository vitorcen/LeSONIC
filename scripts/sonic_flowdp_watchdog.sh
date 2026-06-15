#!/usr/bin/env bash
# Auto-resume watchdog for FlowDP-SONIC training.
#
# The local box (python 3.10.20 / kernel 6.17) throws INTERMITTENT heap-corruption
# crashes mid-run — e.g. `ValueError: too many values to unpack (expected 0)` deep in
# torch's named_parameters() — that have nothing to do with the training logic (it ran
# 900+ clean steps first). They can come in bursts. This wrapper relaunches training,
# resuming from the latest checkpoint, until it reaches STEPS or MAX_RETRIES is hit.
#
#   bash scripts/sonic_flowdp_watchdog.sh            # fresh start, auto-resume on crash
# Tunables via env: STEPS, SAVE_FREQ, BATCH_SIZE, MAX_RETRIES, OUTPUT_DIR.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/flowdp_sonic}"
STEPS="${STEPS:-2400}"               # ~40 epochs (batch64, 3815 frames -> ~60 steps/ep)
SAVE_FREQ="${SAVE_FREQ:-120}"        # ~2 epochs/ckpt -> dense low-epoch sweep coverage
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_RETRIES="${MAX_RETRIES:-40}"
LOG="${LOG:-/tmp/flowdp_sonic_train.log}"

export STEPS SAVE_FREQ BATCH_SIZE NUM_WORKERS OUTPUT_DIR

last_step() {
  local f="$OUTPUT_DIR/checkpoints/last/training_state/training_step.json"
  [ -f "$f" ] && grep -oE '[0-9]+' "$f" | head -1 || echo 0
}

echo "[watchdog] $(date +%H:%M:%S) FlowDP-SONIC | target=$STEPS save_freq=$SAVE_FREQ batch=$BATCH_SIZE" | tee -a "$LOG"

for attempt in $(seq 1 "$MAX_RETRIES"); do
  # completion check (log marker OR last ckpt step reached target)
  if grep -q "End of training" "$LOG" 2>/dev/null; then
    echo "[watchdog] training reported End of training — done." | tee -a "$LOG"; exit 0
  fi
  st="$(last_step)"
  if [ "$st" -ge "$STEPS" ] 2>/dev/null; then
    echo "[watchdog] last ckpt step=$st >= $STEPS — done." | tee -a "$LOG"; exit 0
  fi

  if [ -f "$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json" ]; then
    RES=1; echo "[watchdog] attempt $attempt: RESUME from step $st" | tee -a "$LOG"
  else
    RES=0; echo "[watchdog] attempt $attempt: fresh start" | tee -a "$LOG"
  fi

  RESUME="$RES" bash "$REPO_ROOT/scripts/sonic_flowdp_train.sh" >> "$LOG" 2>&1
  rc=$?
  echo "[watchdog] attempt $attempt exited rc=$rc (step now $(last_step))" | tee -a "$LOG"
  [ "$rc" = "0" ] && continue   # clean exit -> loop re-checks completion markers
  sleep 8                       # let any transient corruption settle before retry
done

echo "[watchdog] gave up after $MAX_RETRIES attempts (step $(last_step)/$STEPS)" | tee -a "$LOG"
exit 1
