#!/usr/bin/env bash
# Watchdog for local StarVLA SONIC training on the GPF-prone kernel-6.17 box.
# Bursty general-protection-fault crashes (import-time or mid-train) are a
# known machine-level issue — the cure is dense rolling checkpoints + relaunch
# with RESUME=1 until the target step lands. (feedback-training-save-policy /
# feedback-training-resume-chunks; atomic .tmp saves mean a mid-save crash
# never corrupts a real checkpoint.)
#
#   CONFIG=$PWD/scripts/starvla/configs/sonic_qwen3_5_4b_ce.yaml \
#     MAX_STEPS=6000 bash scripts/starvla_sonic_watchdog.sh
#
# Knobs: SAVE_INTERVAL (default 250 = lose <=250 steps per crash),
#        KEEP (rolling ckpts, default 3), MAX_ATTEMPTS (default 60),
#        SLEEP_S (between relaunches, default 60 — boot-crash bursts pass).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:?set CONFIG=<yaml>}"
MAX_STEPS="${MAX_STEPS:-6000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-250}"
KEEP="${KEEP:-3}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-60}"
SLEEP_S="${SLEEP_S:-60}"
RUN_ID="$(grep -E '^run_id:' "$CONFIG" | awk '{print $2}')"
CK="$REPO_ROOT/outputs/starvla/$RUN_ID/checkpoints"
LOG="${LOG:-/tmp/starvla_watchdog_$RUN_ID.log}"
TRAIN_LOG="${TRAIN_LOG:-/tmp/starvla_train_$RUN_ID.log}"

# MAX_STEPS must be a multiple of SAVE_INTERVAL or the final ckpt never saves.
(( MAX_STEPS % SAVE_INTERVAL == 0 )) || { echo "MAX_STEPS % SAVE_INTERVAL != 0" >&2; exit 2; }

echo "[watchdog] run=$RUN_ID target=steps_$MAX_STEPS save=$SAVE_INTERVAL keep=$KEEP" | tee -a "$LOG"
for i in $(seq 1 "$MAX_ATTEMPTS"); do
  if [[ -f "$CK/steps_${MAX_STEPS}_pytorch_model.pt" ]]; then
    echo "[watchdog] ✅ steps_$MAX_STEPS present — done (attempt $i)" | tee -a "$LOG"; exit 0
  fi
  if ps -eo cmd | grep -q "[t]rain_starvla.py"; then   # [t] = self-match-proof
    sleep "$SLEEP_S"; continue   # someone else (or a previous attempt) is training
  fi
  rm -f "$CK"/*.tmp              # atomic-save leftovers from a mid-save crash
  RESUME=1
  ls "$CK"/steps_*_pytorch_model.pt >/dev/null 2>&1 || RESUME=0   # fresh run
  echo "[watchdog] attempt $i $(date +%H:%M:%S) RESUME=$RESUME latest=$(ls -t "$CK"/steps_*_pytorch_model.pt 2>/dev/null | head -1 | xargs -r basename)" | tee -a "$LOG"
  CONFIG="$CONFIG" MAX_STEPS="$MAX_STEPS" SAVE_INTERVAL="$SAVE_INTERVAL" KEEP="$KEEP" RESUME="$RESUME" \
    bash "$REPO_ROOT/scripts/starvla_sonic_finetune.sh" > "$TRAIN_LOG" 2>&1
  echo "[watchdog] trainer exited (attempt $i)" | tee -a "$LOG"
  sleep "$SLEEP_S"
done
echo "[watchdog] 🔴 $MAX_ATTEMPTS attempts exhausted without steps_$MAX_STEPS" | tee -a "$LOG"
exit 1
