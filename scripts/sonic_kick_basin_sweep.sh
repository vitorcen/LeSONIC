#!/usr/bin/env bash
# Phase-1 closed-loop kick-basin sweep (doc/sonic_robustness_retrain.html §3, §6).
#
# For each checkpoint in a training output dir, run a LIVE kick rollout (bootstrap=0 — the policy
# generates every token, the deploy scenario that FREEZES on the un-augmented ckpt-8000) and score
# the swing-foot peak height + joint excursion from the VLA_DIAG_TRACE harness. This is the decisive
# metric — NOT open-loop token-MSE (which was 0.0011 while the closed loop failed).
#
# Baseline to beat (ckpt-8000, no noise): right-foot max z = 0.038 m (frozen), joint exc = 0.67.
# A widened basin = right-foot lifts (>0.30 m) and joint excursion approaches the GT replay (3.25).
#
#   bash scripts/sonic_kick_basin_sweep.sh outputs/gr00t_sonic_noise08
#   STEPS=200 LOOPS=2 bash scripts/sonic_kick_basin_sweep.sh <out_dir> [ckpt-glob]
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:?usage: sonic_kick_basin_sweep.sh <training_out_dir> [ckpt-glob]}"
OUT_DIR="$(cd "$OUT_DIR" 2>/dev/null && pwd || echo "$OUT_DIR")"   # absolute: GR00T server cd's into its venv dir
GLOB="${2:-checkpoint-*}"
PORT="${GR00T_PORT:-5555}"
STEPS="${STEPS:-200}"
LOOPS="${LOOPS:-2}"
MOTION="${MOTION:-kick}"
RESULT="${RESULT:-/tmp/kick_basin_sweep.tsv}"
GR00T_DIR="$REPO_ROOT/dependencies/Isaac-GR00T"

# right_ankle_roll_link z is the kick signal; analysis mirrors the Phase-0 trace scoring.
score() {  # $1 = trace npz
  python3 - "$1" <<'PY'
import sys, numpy as np
d = np.load(sys.argv[1], allow_pickle=True)
names = [str(x) for x in d['body_names']]
ri = names.index('right_ankle_roll_link')
bz, j, root = d['bodyz'], d['joints'], d['root']
exc = np.linalg.norm(j - j[0], axis=1)
rf = bz[:, ri]
print(f"{rf.max():.3f}\t{rf.max()-rf.min():.3f}\t{exc.max():.3f}\t{root[:,2].min():.3f}\t{(rf>0.30).mean():.2f}")
PY
}

printf 'ckpt\trfoot_max\trfoot_rng\tjoint_exc\trootz_min\tfrac_kick\n' | tee "$RESULT"
for ck in $(ls -d "$OUT_DIR"/$GLOB 2>/dev/null | sort -t- -k2 -n); do
  [[ -f "$ck/model.safetensors.index.json" || -f "$ck/model.safetensors" ]] || continue
  name="$(basename "$ck")"
  trace="/tmp/basin_${name}.npz"
  log="/tmp/basin_${name}.log"
  rm -f "$trace" "$log"
  # fresh server per ckpt (stop any prior), then LIVE kick rollout with the trace harness.
  bash "$REPO_ROOT/scripts/gear_sonic_stop.sh" >/dev/null 2>&1
  HEADLESS=1 SETTLE=0 BOOTSTRAP_STEPS=0 STEPS="$STEPS" LOOPS="$LOOPS" \
    VLA_DIAG_TRACE="$trace" GR00T_CKPT="$ck" \
    setsid bash "$REPO_ROOT/scripts/gear_sonic_live_demo.sh" "$MOTION" >"$log" 2>&1 &
  # wait for trace to fill (Isaac boot ~90s + rollout) or the run to end
  for _ in $(seq 1 120); do
    if [ -f "$trace" ]; then
      n=$(python3 -c "import numpy as np;print(len(np.load('$trace')['t']))" 2>/dev/null || echo 0)
      [ "${n:-0}" -ge $((STEPS*LOOPS - 40)) ] && break
    fi
    pgrep -f "python gear_sonic/eval_agent_trl.py" >/dev/null 2>&1 || { [ -f "$trace" ] && break; }
    sleep 3
  done
  for p in $(pgrep -f "python gear_sonic/eval_agent_trl.py"); do kill -TERM "$p" 2>/dev/null; done
  bash "$REPO_ROOT/scripts/gear_sonic_stop.sh" >/dev/null 2>&1
  if [ -f "$trace" ]; then
    printf '%s\t%s\n' "$name" "$(score "$trace")" | tee -a "$RESULT"
  else
    printf '%s\tNO_TRACE\n' "$name" | tee -a "$RESULT"
  fi
done
echo "[sweep] done -> $RESULT"
